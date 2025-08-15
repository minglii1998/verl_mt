# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single
GPU model. Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model
to perform generation.
"""

import contextlib
import numpy as np
import re, random
import torch
import torch.distributed
import ray
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.torch_functional import get_response_mask, postprocess_data, pad_sequence_to_length
from verl.utils.model import compute_position_id_with_mask

from .base import BaseRollout

__all__ = ["HFShardedMTRollout"]


class HFShardedMTRollout(BaseRollout):
    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module

        # Initialize tokenizer directly from the training model's config path.
        # `module` is the HF model being trained, whose `config.name_or_path` always points to the correct repository.
        self.tokenizer = AutoTokenizer.from_pretrained(self.module.config.name_or_path, trust_remote_code=True)

        # Maximum token length of each segment when splitting long queries
        self.segment_token_length = self.config.get("segment_token_length", 128)

        self.input_abstain_rate = config.get("input_abstain_rate", 0.0)
        self.max_shard_count = config.get("max_shard_count", 10)
        # Whether to respect curriculum-controlled segment count
        self.curriculum_segments_enabled = config.get("curriculum_segments_enabled", False)
        self.num_segments_required = config.get("num_segments_required", 1) if self.curriculum_segments_enabled else 999
        self.abstain_prompt = "If you believe the question is not solvable, reason step by step and output \\boxed{abstain}."

    def _sanity_check(self, dp: DataProto, pad_token_id: int, tokenizer):
        """
        Validate the DataProto structure after generation.
        Checks tensor shapes, padding consistency, and position_ids continuity.
        """
        td = dp.batch
        ids = td["input_ids"]
        attn = td["attention_mask"]
        pos = td["position_ids"]
        prm = td["prompts"]
        rsp = td["responses"]

        # Check tensor shape consistency
        assert ids.shape == attn.shape == pos.shape, "shape mismatch among ids/attn/pos"

        B, L = ids.shape

        # Check attention_mask corresponds to pad_token
        assert torch.all((ids == pad_token_id) == (attn == 0)), "pad-token and attention_mask mismatch"

        # Check prompt/response alignment in input_ids
        Lp = prm.size(1)
        Lr = rsp.size(1)
        assert Lp + Lr == L, "Lp+Lr != total length"
        assert torch.equal(ids[:, :Lp], prm), "prompt slice mismatch"
        assert torch.equal(ids[:, Lp:Lp+Lr], rsp), "response slice mismatch"

        # Check position_ids continuity for each sample
        for b in range(B):
            sample_attn = attn[b]
            sample_pos = pos[b]
            
            valid_mask = sample_attn == 1
            valid_positions = torch.where(valid_mask)[0]
            
            if len(valid_positions) > 1:
                valid_pos_ids = sample_pos[valid_positions]
                diff = valid_pos_ids[1:] - valid_pos_ids[:-1]
                    
                assert torch.all(diff == 1), f"Sample {b} position_ids not continuous"

        # Check response position_id continuity with prompt
        for b in range(B):
            prompt_valid = attn[b, :Lp] == 1
            response_valid = attn[b, Lp:] == 1
            
            if prompt_valid.any() and response_valid.any():
                last_prompt_pos = pos[b, :Lp][prompt_valid][-1]
                first_response_pos = pos[b, Lp:][response_valid][0]
                
                assert first_response_pos == last_prompt_pos + 1, \
                    f"Sample {b}: response start position_id({first_response_pos}) != prompt end({last_prompt_pos}) + 1"

        # Verify decoded content consistency
        decoded_prompt = tokenizer.batch_decode(prm, skip_special_tokens=True)
        decoded_slice = tokenizer.batch_decode(ids[:, :Lp], skip_special_tokens=True)
        assert decoded_prompt == decoded_slice, "Decoded prompt content mismatch"

        # Check padding token consistency
        pad_mask = attn == 0
        pad_tokens = ids[pad_mask]
        if pad_tokens.numel() > 0:
            assert torch.all(pad_tokens == pad_token_id), \
                f"Non-pad tokens found in padding positions: {torch.unique(pad_tokens).tolist()}"

        # Check padding continuity for each sample
        for b in range(B):
            sample_attn = attn[b]
            sample_ids = ids[b]
            
            valid_positions = torch.where(sample_attn == 1)[0]
            
            if len(valid_positions) > 0:
                first_valid = valid_positions[0]
                last_valid = valid_positions[-1]
                
                # Check left padding
                if first_valid > 0:
                    left_pad_tokens = sample_ids[:first_valid]
                    assert torch.all(left_pad_tokens == pad_token_id), \
                        f"Sample {b}: left padding contains non-pad tokens"
                
                # Check right padding
                if last_valid < L - 1:
                    right_pad_tokens = sample_ids[last_valid + 1:]
                    assert torch.all(right_pad_tokens == pad_token_id), \
                        f"Sample {b}: right padding contains non-pad tokens"
                
                # Check middle continuity
                if last_valid > first_valid:
                    middle_attn = sample_attn[first_valid:last_valid + 1]
                    assert torch.all(middle_attn == 1), \
                        f"Sample {b}: padding found in middle of valid tokens"
            else:
                # If no valid tokens, entire sequence should be padding
                assert torch.all(sample_ids == pad_token_id), \
                    f"Sample {b}: no valid tokens but contains non-pad tokens"

        # Check response mask continuity: if 0 appears, all following should be 0
        for b in range(B):
            resp_mask = attn[b, Lp:]
            if resp_mask.numel():
                zero_positions = torch.where(resp_mask == 0)[0]
                if len(zero_positions) > 0:
                    first_zero = zero_positions[0]
                    after_first_zero = resp_mask[first_zero:]
                    assert torch.all(after_first_zero == 0), \
                        f"Sample {b}: attention_mask has 1s after EOS"

    def _validate_format_consistency(self, dp: DataProto, expected_response_length: int):
        """Validate that returned DataProto format is consistent with standard HFRollout"""
        td = dp.batch
        
        # Check required keys exist
        required_keys = {"prompts", "responses", "input_ids", "attention_mask", "position_ids"}
        actual_keys = set(td.keys())
        assert required_keys.issubset(actual_keys), f"Missing required keys: {required_keys - actual_keys}"
        
        # Check dimensions
        batch_size = td.batch_size[0]
        prompts = td["prompts"]
        responses = td["responses"] 
        input_ids = td["input_ids"]
        attention_mask = td["attention_mask"]
        position_ids = td["position_ids"]
        
        prompt_length = prompts.size(1)
        response_length = responses.size(1)
        total_length = input_ids.size(1)
        
        # Validate shape consistency
        assert prompts.shape == (batch_size, prompt_length), f"prompts shape error: {prompts.shape}"
        assert responses.shape == (batch_size, response_length), f"responses shape error: {responses.shape}"
        assert input_ids.shape == (batch_size, total_length), f"input_ids shape error: {input_ids.shape}"
        assert attention_mask.shape == (batch_size, total_length), f"attention_mask shape error: {attention_mask.shape}"
        assert position_ids.shape == (batch_size, total_length), f"position_ids shape error: {position_ids.shape}"
        
        # Validate length relationships
        assert prompt_length + response_length == total_length, "prompt + response != total length"
        assert response_length == expected_response_length, f"response length {response_length} != expected {expected_response_length}"
        
        # Validate data type consistency
        assert prompts.dtype == responses.dtype == input_ids.dtype, "token tensors have inconsistent types"
        assert attention_mask.dtype == position_ids.dtype, "mask tensors have inconsistent types"
        
        # Validate tensor concatenation relationship
        reconstructed = torch.cat([prompts, responses], dim=1)
        assert torch.equal(input_ids, reconstructed), "input_ids != cat(prompts, responses)"

    def _split_question_into_segments(self, question: str, *, shuffle: bool = False, task_type: str = None):
        """
        Split question into segments by sentence delimiters and remove trailing commas/semicolons.
        If shuffle=True, randomly shuffle the order before returning.
        """

        if task_type == 'math':
            delimiter_pattern = (
                r"([。！？!?；;:]|"        # Full/half-width sentence endings
                r"(?<!\d)[,，](?!\d)|"    # Commas, excluding digit separators
                r"(?<!\d)\.(?!\d))"       # Periods, excluding decimals
            )
        elif task_type == 'code':
            delimiter_pattern = r"([。！？!?]|(?<!\d)\.(?!\d)|\n+)"

            # Used for code examples
            markers = [
                "\n\nInput", 
                "\n$Example:$", 
                "\n\nExample", 
                "\n-----Input-----", 
                "\n\n# Example",
                "\n\n# Input",
                "\n\n# Output",
                "\n\n-----Constraints-----",
                "\nExample",
                "\n\n For example:",
                "\n------ Input Format ------"
            ]
            cut_positions = [question.find(m) for m in markers if question.find(m) != -1]
            question = question[: min(cut_positions)].strip()
            examples = question[min(cut_positions):].strip()


        parts = re.split(delimiter_pattern, question)
        segments, buf = [], ""

        for frag in parts:
            if not frag:
                continue
            if re.match(delimiter_pattern, frag):
                if task_type == 'math':
                    buf += frag
                    seg = re.sub(r"[，,；;]+$", "", buf.strip())  # Remove trailing commas/semicolons
                elif task_type == 'code':
                    if not frag.startswith("\n"):
                        buf += frag
                    seg = buf.strip()
                if seg:
                    segments.append(seg)
                buf = ""
            else:
                buf += frag

        if buf.strip():
            if task_type == 'math':
                segments.append(re.sub(r"[，,；;]+$", "", buf.strip()))
            elif task_type == 'code':
                segments.append(buf.strip())

        if task_type == 'code':
            segments.append(examples)

        return segments or [question]

    def _split_question_into_n_segments(self, question: str, n: int, *, task_type: str = None):
        """
        Randomly select n-1 sentence boundary cut points to split question into exactly n segments.
        
        Uses same delimiters as _split_question_into_segments.
        If available cut points < n-1, falls back to _split_question_into_segments result
        and adjusts to ensure n segments.
        """

        if task_type == 'math':
            delimiter_pattern = (
                r"([。！？!?；;:]|"        # Sentence endings
                r"(?<!\d)[,，](?!\d)|"    # Commas excluding digit grouping
                r"(?<!\d)\.(?!\d))"       # Periods excluding decimals
            )
        elif task_type == 'code':
            delimiter_pattern = r"([。！？!?]|(?<!\d)\.(?!\d)|\n+)"

            # Used for code examples
            markers = [
                "\n\nInput", 
                "\n$Example:$", 
                "\n\nExample", 
                "\n-----Input-----", 
                "\n\n# Example",
                "\n\n# Input",
                "\n\n# Output",
                "\n\n-----Constraints-----",
                "\nExample",
                "\n\n For example:",
                "\n------ Input Format ------"
            ]
            cut_positions = [question.find(m) for m in markers if question.find(m) != -1]
            question_ori = question
            question = question[: min(cut_positions)].strip()
            examples = question_ori[min(cut_positions):].strip()

        # Collect valid cut points (after punctuation), excluding text end to avoid empty segments
        cut_positions = [
            m.end() for m in re.finditer(delimiter_pattern, question)
            if m.end() < len(question)
        ]

        # Fallback if insufficient cut points or n<=1
        if task_type == 'code':
            if n == 1:
                return [question_ori]
            elif n == 2:
                return [question, examples]

            # Select up to (n-2) cut points and sort. If not enough cut points,
            # we will duplicate the last segment to reach exactly n segments.
            k = min(n - 2, len(cut_positions))
            chosen = sorted(random.sample(cut_positions, k))

        else:
            if len(cut_positions) < n - 1 or n <= 1:
                return [question]

            # Select n-1 cut points and sort
            chosen = sorted(random.sample(cut_positions, n - 1))

        # Generate segments, removing trailing commas/semicolons
        segments, prev = [], 0
        for cut in chosen + [len(question)]:
            seg = question[prev:cut].strip()
            if task_type == 'math':
                seg = re.sub(r"[，,；;]+$", "", seg)
            elif task_type == 'code':
                seg = seg.strip()
            segments.append(seg)
            prev = cut

        # Ensure the number of segments matches expectation
        if task_type == 'code':
            # For code tasks with examples, append examples as the last segment
            # to make the total segments equal to n.
            while len(segments) < n - 1:
                # Duplicate last available segment to pad
                segments.append(segments[-1] if segments else question)
            segments.append(examples)
            return segments
        
        return segments

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_multi_turn(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    @torch.no_grad()
    def _generate_multi_turn(self, prompts: DataProto) -> DataProto:
        """Handle prompts with raw_prompt by performing multi-turn generation.

        For each sample, we split the original user question into segments and run
        inference turn-by-turn, accumulating assistant responses, then return the
        result built from the last turn for RLHF reward calculation.
        """
        self.module.eval()
        # param_ctx will be created fresh for each generate call to avoid reusing exhausted context
        with torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
            is_validate = prompts.meta_info.get("validate", False)

            temperature = prompts.meta_info.get("temperature", self.config.temperature)
            response_length = prompts.meta_info.get("response_length", self.config.response_length)
            top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
            top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))  # to be compatible with vllm

            if not do_sample:
                # do_sample==False -> greedy decoding
                kwargs = {
                    "do_sample": False,
                    "num_beams": 1,
                }
            elif is_validate:
                # do validate and do sample -> use val_kwargs
                kwargs = {
                    "do_sample": True,
                    "num_beams": 1,
                    "top_k": max(0, self.config.val_kwargs.top_k),  # to be compatible with vllm
                    "top_p": self.config.val_kwargs.top_p,
                    "temperature": self.config.val_kwargs.temperature,
                    "num_return_sequences": 1,  # if validate, already repeat in ray_trainer
                }
            else:
                # do_sample -> use rollout config
                kwargs = {
                    "do_sample": True,
                    "num_beams": 1,
                    "top_p": top_p,
                    "top_k": top_k,
                    "temperature": temperature,
                    "num_return_sequences": 1, # not diretly use num_return_sequences for n rollouts
                }

            # make config according to generate mode
            generation_config = GenerationConfig(**kwargs)

            # used to construct attention_mask
            eos_token_id = prompts.meta_info["eos_token_id"]
            pad_token_id = prompts.meta_info["pad_token_id"]

            # Curriculum parameter passed from trainer (if any)
            num_segments_required = prompts.meta_info.get("num_segments_required", self.num_segments_required)

            # If curriculum disabled, effectively ignore it by setting a large requirement
            if not self.curriculum_segments_enabled:
                num_segments_required = 999

            raw_prompts = prompts.non_tensor_batch["raw_prompt"]

            # Determine minimum number of segments across all samples
            min_num_segments = 1  # fallback
            num_segments_per_sample = []
            segments_per_sample = []
            for chat in raw_prompts:
                # convert ndarray to list when necessary
                chat_list = chat.tolist() if isinstance(chat, np.ndarray) else chat
                task_type = None
                if 'Python' in chat_list[0]['content']:
                    task_type = 'code'
                elif 'mathematical' in chat_list[0]['content']:
                    task_type = 'math'
                else:
                    task_type = 'math'

                # locate last user message within the conversation
                last_user_idx = max(i for i, m in enumerate(chat_list) if m["role"] == "user")
                user_question = chat_list[last_user_idx]["content"]
                segments_tmp = self._split_question_into_segments(user_question, task_type=task_type)
                segments_per_sample.append(segments_tmp)
                num_segments_per_sample.append(len(segments_tmp))

            if num_segments_per_sample:
                local_min_seg = max(1, min(num_segments_per_sample))

            # ray.util.pdb.set_trace()
            device = torch.device(get_device_name())
            if torch.distributed.is_initialized():
                min_seg_tensor = torch.tensor(local_min_seg, device=device, dtype=torch.long)
                torch.distributed.all_reduce(min_seg_tensor, op=torch.distributed.ReduceOp.MIN)
                min_num_segments = int(min_seg_tensor.item())
            else:
                min_num_segments = local_min_seg

            min_num_segments = min(num_segments_required, min_num_segments, self.max_shard_count)

            # print(f"min_num_segments: {min_num_segments}", flush=True)

            self.tokenizer.padding_side = "left"  # left pad for efficiency on causal LM

            if is_validate:
                config_n = 1
            else:
                config_n = self.config.n
            num_samples = len(raw_prompts) * config_n

            # Build conversations & segments list for each sample
            conversations = []  # list[list[dict]]
            segments_all = []   # list[list[str]]
            required_reward_types = []
            for j, chat in enumerate(raw_prompts):
                chat_list = chat.tolist() if isinstance(chat, np.ndarray) else chat
                task_type = None
                if 'Python' in chat_list[0]['content']:
                    task_type = 'code'
                elif 'mathematical' in chat_list[0]['content']:
                    task_type = 'math'
                else:
                    task_type = 'math' 

                for i in range(config_n): # n rollouts
                    chat_l = chat.tolist() if isinstance(chat, np.ndarray) else chat
                    last_user_idx = max(i for i, m in enumerate(chat_l) if m["role"] == "user")

                    conv = list(chat_l[:last_user_idx])
                    if self.input_abstain_rate > 0:
                        added = False
                        for msg in conv:
                            if msg.get("role") == "system":
                                if self.abstain_prompt not in msg.get("content", ""):
                                    msg["content"] = msg.get("content", "").rstrip() + " " + self.abstain_prompt
                                added = True
                                break
                    conversations.append(conv)

                    user_q = chat_l[last_user_idx]["content"]
                    # input abstain format
                    if random.random() < self.input_abstain_rate and len(segments_per_sample[j]) > min_num_segments:
                        segments_n = self._split_question_into_n_segments(user_q, min_num_segments+1, task_type=task_type)
                        # random.shuffle(segments_n)
                        segments_n = segments_n[:-1]
                        required_reward_types.append("abstain")
                    else:
                        segments_n = self._split_question_into_n_segments(user_q, min_num_segments, task_type=task_type)
                        # random.shuffle(segments_n)
                        required_reward_types.append("default")

                    segments_all.append(segments_n)

            # ray.util.pdb.set_trace()
            assert num_samples == len(conversations) == len(segments_all), f"num_samples: {num_samples}, len(conversations): {len(conversations)}, len(segments_all): {len(segments_all)}"

            # ray.util.pdb.set_trace()
            # placeholders for last turn tensors
            last_prompt_ids_lst, last_attention_mask_lst, last_position_ids_lst, last_seq_lst = [None]*num_samples, [None]*num_samples, [None]*num_samples, [None]*num_samples
            for t in range(min_num_segments):
                # Build prompts for this turn
                prompt_texts = []
                for s in range(num_samples):
                    conversations[s].append({"role": "user", "content": segments_all[s][t]})
                    if 'Qwen3' in self.module.config.name_or_path:
                        pt = self.tokenizer.apply_chat_template(
                            conversations[s], 
                            tokenize=False, 
                            add_generation_prompt=True, 
                            enable_thinking=False)
                    else:
                        pt = self.tokenizer.apply_chat_template(
                            conversations[s], 
                            tokenize=False, add_generation_prompt=True)
                    prompt_texts.append(pt)

                tok_batch = self.tokenizer(prompt_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
                input_ids, attention_mask = tok_batch["input_ids"], tok_batch["attention_mask"]
                position_ids_batch = compute_position_id_with_mask(attention_mask)

                # ray.util.pdb.set_trace()
                with (
                    FSDP.summon_full_params(self.module, writeback=False, recurse=False)
                    if isinstance(self.module, FSDP) else contextlib.nullcontext()
                ), torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                    out = self.module.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=do_sample,
                        max_new_tokens=response_length,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                        generation_config=generation_config,
                        output_scores=False,
                        return_dict_in_generate=True,
                        use_cache=True,
                    )

                # 立刻同步，保证所有 rank 完全跑完 un-shard
                torch.cuda.synchronize()
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()

                seq_batch = out.sequences  # (batch, prompt+new)

                # decode assistant text & update conversations
                prompt_len_const = input_ids.size(1)  # original prompt length (after left padding) is same across batch
                assistant_text_list = []
                for s in range(num_samples):
                    gen_tokens = seq_batch[s, prompt_len_const:]
                    assistant_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    conversations[s].append({"role": "assistant", "content": assistant_text})
                    assistant_text_list.append(assistant_text)

                # if assistant_text_list:
                #     max_text = max(assistant_text_list, key=len)
                #     print(f"t={t}, seq_batch: {seq_batch.shape}, longest_assistant_text: {[max_text]}", flush=True)
                    
                if t == min_num_segments - 1:
                    for s in range(num_samples):
                        last_prompt_ids_lst[s] = input_ids[s:s+1].clone()
                        last_attention_mask_lst[s] = attention_mask[s:s+1].clone()
                        last_position_ids_lst[s] = position_ids_batch[s:s+1].clone()
                        last_seq_lst[s] = seq_batch[s:s+1].clone()
                # ray.util.pdb.set_trace()

            # Build & left-pad outputs in a single pass
            seq_lengths = []
            adjusted_seq_list = []
            prompt_len_list = []

            for s in range(num_samples):
                seq = last_seq_lst[s]
                prompt_len = last_prompt_ids_lst[s].size(1)
                target_len = prompt_len + self.config.response_length
                if seq.size(1) < target_len:
                    pad_len = target_len - seq.size(1)
                    seq = torch.cat([
                        seq,
                        torch.full((1, pad_len), pad_token_id, dtype=seq.dtype, device=seq.device),
                    ], dim=1)
                adjusted_seq_list.append(seq)
                seq_lengths.append(seq.size(1))
                prompt_len_list.append(prompt_len)

            local_max_seq_len = max(seq_lengths)
            
            # Synchronize global maximum sequence length across all distributed workers
            # This ensures consistent tensor dimensions for DataProto.concat
            global_max_seq_len = local_max_seq_len
            if torch.distributed.is_initialized():
                # Create tensor for all_reduce operation
                max_len_tensor = torch.tensor(local_max_seq_len, dtype=torch.long, device=device)
                # All-reduce to get the global maximum across all workers
                torch.distributed.all_reduce(max_len_tensor, op=torch.distributed.ReduceOp.MAX)
                global_max_seq_len = max_len_tensor.item()

            sample_outputs = []
            for s in range(num_samples):
                seq = adjusted_seq_list[s]
                prompt_len = prompt_len_list[s]
                if seq.size(1) < global_max_seq_len:
                    pad_size = global_max_seq_len - seq.size(1)
                    seq = torch.cat([
                        torch.full((1, pad_size), pad_token_id, dtype=seq.dtype, device=seq.device),
                        seq,
                    ], dim=1)

                    prompt_len_list[s] += pad_size
                    last_prompt_ids_lst[s] = torch.cat(
                        [torch.full((1, pad_size), pad_token_id, device=seq.device, dtype=seq.dtype),
                        last_prompt_ids_lst[s]],
                        dim=1)
                        
                    # For prompt attention_mask: only pad with zeros for the prompt part
                    last_attention_mask_lst[s] = torch.cat([
                        torch.zeros((1, pad_size), dtype=last_attention_mask_lst[s].dtype, device=seq.device),
                        last_attention_mask_lst[s],
                    ], dim=1)

                    # For prompt position_ids: recompute from the padded prompt attention_mask
                    last_position_ids_lst[s] = compute_position_id_with_mask(last_attention_mask_lst[s])

                response = seq[:, prompt_len_list[s]:]
                resp_attn_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=last_attention_mask_lst[s].dtype)
                attn_mask_full = torch.cat([last_attention_mask_lst[s], resp_attn_mask], dim=-1)

                # Recompute position_ids from final attention_mask to guarantee monotonicity
                pos_ids_full = compute_position_id_with_mask(attn_mask_full)

                prompts_slice = seq[:, :prompt_len_list[s]]
                td = TensorDict(
                    {
                        "prompts": prompts_slice,
                        "responses": response,
                        "input_ids": seq,
                        "attention_mask": attn_mask_full,
                        "position_ids": pos_ids_full,
                    },
                    batch_size=1,
                )
                # Pass required reward type information downstream via non_tensor_batch
                sample_outputs.append(
                    DataProto(
                        batch=td,
                        non_tensor_batch={"rollout_info": np.array([required_reward_types[s]], dtype=object)},
                    )
                )

        self.module.train()
        get_torch_device().empty_cache()
        batch_data_proto = DataProto.concat(sample_outputs)
        self._validate_format_consistency(batch_data_proto, self.config.response_length)
        self._sanity_check(batch_data_proto, pad_token_id, self.tokenizer)
        # ray.util.pdb.set_trace()
        return batch_data_proto

    def update_num_segments_required(self, num_segments_required: int):
        """Dynamically update the minimum segment count used during generation."""
        self.num_segments_required = num_segments_required
