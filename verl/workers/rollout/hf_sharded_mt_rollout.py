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
import random
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

    def _split_question_into_segments(self, question: str):
        """
        按句末标点切分 *question*，并去掉每段末尾多余的逗号/分号。
        """
        import re

        # 把 . 也放进捕获分组，才能在 re.split 结果中被单独取出
        delimiter_pattern = r"([。！？!?；;:]|\.(?!\d))"
        parts = re.split(delimiter_pattern, question)

        segments, buf = [], ""
        for frag in parts:
            if not frag:
                continue

            # 如果是句末标点 → 输出一个 segment
            if re.match(delimiter_pattern, frag):
                buf += frag
                seg = re.sub(r"[，,；;]+$", "", buf.strip())  # 去尾逗号/分号
                if seg:
                    segments.append(seg)
                buf = ""
            else:
                buf += frag

        # 处理最后残留
        if buf.strip():
            segments.append(re.sub(r"[，,；;]+$", "", buf.strip()))

        return segments or [question]


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
        result built from the **last** turn (so RLHF can calculate reward on the
        final answer).
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
                    "num_return_sequences": self.config.n,
                }

            # make config according to generate mode
            generation_config = GenerationConfig(**kwargs)

            # used to construct attention_mask
            eos_token_id = prompts.meta_info["eos_token_id"]
            pad_token_id = prompts.meta_info["pad_token_id"]

            raw_prompts = prompts.non_tensor_batch["raw_prompt"]
            sample_outputs = []

            for idx_sample, chat in enumerate(raw_prompts):
                if isinstance(chat, np.ndarray):
                    chat = chat.tolist()

                # Identify the *last* user message as the target to split. Previous turns remain intact.
                last_user_idx = max(i for i, m in enumerate(chat) if m["role"] == "user")
                # Everything before that (inclusive previous assistant/tool/system messages) forms the running context.
                conversation = chat[:last_user_idx]  # shallow copy is fine (list of dicts)
                user_question = chat[last_user_idx]["content"]
                user_question = user_question.split('Let\'s think step by step and output the final answer after')[0]
                # segments = [user_question]

                segments = self._split_question_into_segments(user_question)
                random.shuffle(segments)

                last_prompt_ids = None  # will store tensor for last turn
                last_attention_mask = None
                last_position_ids = None
                last_seq = None

                # ray.util.pdb.set_trace()
                for seg in segments:
                    # 1) append user seg to conversation
                    conversation.append({"role": "user", "content": seg})

                    # 2) tokenize current conversation
                    if 'Qwen3' in self.module.config.name_or_path:
                        prompt_text = self.tokenizer.apply_chat_template(
                            conversation,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False 
                        )
                    else:
                        prompt_text = self.tokenizer.apply_chat_template(
                            conversation,
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                    tok = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
                    device = torch.device(get_device_name())
                    input_ids = tok["input_ids"].to(device)
                    attention_mask = tok["attention_mask"].to(device)
                    position_ids_current = compute_position_id_with_mask(attention_mask).to(device)

                    ctx = (
                        FSDP.summon_full_params(self.module, writeback=False, recurse=False)
                        if isinstance(self.module, FSDP)
                        else contextlib.nullcontext()
                    )

                    with ctx, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                        output = self.module.generate(
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
                    seq = output.sequences  # (1, prompt_len + new_tokens)

                    # 3) decode assistant response and append as assistant message for next round (except after last seg)
                    prompt_len = input_ids.size(1)
                    assistant_text_all = self.tokenizer.decode(seq[0], skip_special_tokens=True)
                    assistant_tokens = seq[0, prompt_len:]
                    assistant_text = self.tokenizer.decode(assistant_tokens, skip_special_tokens=True)
                    conversation.append({"role": "assistant", "content": assistant_text})
                    # ray.util.pdb.set_trace()

                    # Save tensors from this last turn (will be used to build output after loop)
                    last_prompt_ids = input_ids
                    last_attention_mask = attention_mask
                    last_position_ids = position_ids_current
                    last_seq = seq

                # Build tensors for last turn similar to single-shot rollout logic
                seq = last_seq  # (1, total_len)
                prompt_length = last_prompt_ids.size(1)
                generated_batch_size = 1

                # pad response to fixed length
                sequence_length = prompt_length + self.config.response_length
                delta_length = sequence_length - seq.shape[1]
                if delta_length > 0:
                    delta_tokens = torch.full((generated_batch_size, delta_length), pad_token_id, dtype=seq.dtype, device=seq.device)
                    seq = torch.cat((seq, delta_tokens), dim=1)

                response = seq[:, prompt_length:]

                # extend position_ids and attention_mask to response part
                response_length_cur = response.size(1)
                delta_position_id = torch.arange(1, response_length_cur + 1, device=last_position_ids.device)
                response_position_ids = last_position_ids[:, -1:] + delta_position_id.unsqueeze(0)
                position_ids_full = torch.cat([last_position_ids, response_position_ids], dim=-1)

                response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=last_attention_mask.dtype)
                attention_mask_full = torch.cat((last_attention_mask, response_attention_mask), dim=-1)

                prompts_ = seq[:, :prompt_length]
                batch_td = TensorDict(
                    {
                        "prompts": prompts_,
                        "responses": response,
                        "input_ids": seq,
                        "attention_mask": attention_mask_full,
                        "position_ids": position_ids_full,
                    },
                    batch_size=generated_batch_size,
                )
                sample_outputs.append(DataProto(batch=batch_td))

            # Calculate target length (prompt_max + response_len)
            max_total_seq_len = 0
            for dp in sample_outputs:
                max_total_seq_len = max(max_total_seq_len, dp.batch["input_ids"].shape[1])

            if max_total_seq_len > 0:
                padded_outputs = []
                for dp in sample_outputs:
                    td = dp.batch

                    cur_len = td["input_ids"].shape[1]
                    if cur_len < max_total_seq_len:
                        # pad input_ids / attention_mask / position_ids on the **left**
                        pad_size = max_total_seq_len

                        input_ids = pad_sequence_to_length(td["input_ids"], pad_size, pad_token_id, left_pad=True)
                        attention_mask = pad_sequence_to_length(td["attention_mask"], pad_size, 0, left_pad=True)
                        position_ids = pad_sequence_to_length(td["position_ids"], pad_size, 0, left_pad=True)

                        # regenerate prompts slice because prompt length expanded
                        prompt_len_new = pad_size - self.config.response_length
                        prompts_new = input_ids[:, :prompt_len_new]

                        # rebuild TensorDict (responses keep the same fixed length)
                        td = TensorDict(
                            {
                                "prompts": prompts_new,
                                "responses": td["responses"],
                                "input_ids": input_ids,
                                "attention_mask": attention_mask,
                                "position_ids": position_ids,
                            },
                            batch_size=td.batch_size,
                        )
                    # else keep original td
                    padded_outputs.append(DataProto(batch=td))

                sample_outputs = padded_outputs

        # ray.util.pdb.set_trace()
        self.module.train()
        get_torch_device().empty_cache()
        return DataProto.concat(sample_outputs)
