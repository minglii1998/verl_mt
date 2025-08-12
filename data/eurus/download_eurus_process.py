import re
import random
import json
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
import pathlib
from tqdm import tqdm

def _split_question_into_segments(question: str, *, shuffle: bool = False):
        """
        Split a question string into coarse-grained sentence segments.

        Delimiters considered:
        1. Sentence-ending punctuation: full/half-width 。 ！ ？ . ! ?
           (the period is excluded when it looks like a decimal point).
        2. One or more newline characters ("\n+"). Consecutive newlines are
           treated as a single delimiter.

        Commas、semicolons等弱分隔符将不再产生分段。
        If *shuffle* is True, the resulting segments are randomly shuffled.
        """

        # Sentence-ending punctuation or newline group
        delimiter_pattern = r"([。！？!?]|(?<!\d)\.(?!\d)|\n+)"

        parts = re.split(delimiter_pattern, question)
        segments, buf = [], ""

        for frag in parts:
            if not frag:
                continue
            if re.match(delimiter_pattern, frag):
                # If punctuation (not newline), keep it at the end of the segment
                if not frag.startswith("\n"):
                    buf += frag
                seg = buf.strip()
                if seg:
                    segments.append(seg)
                buf = ""
            else:
                buf += frag

        if buf.strip():
            segments.append(buf.strip())

        if shuffle:
            random.shuffle(segments)

        return segments or [question]

def main():
    """
    Download Eurus-2-RL dataset, filter for `code` ability and post-process prompts.

    Post-processing rules:
    1. Remove system prompts from the `prompt` field.
    2. For the remaining user prompt, truncate everything that appears after one of
       the following markers (if present): "\n\nInput", "\n$Example:$",
       "\n\nExample", "\n-----Input-----".
    Finally, randomly sample 10 processed examples and save them to a JSON file.
    """

    # For deterministic sampling
    random.seed(42)

    # Initialize tokenizer once (Qwen3 family)
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True
    )

    # Load the training split of the dataset
    dataset = load_dataset("PRIME-RL/Eurus-2-RL-Data", split="train")

    # Keep only examples where the 'ability' field is exactly 'code'.
    code_examples = dataset.filter(lambda example: example.get("ability") == "code")

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
    processed_samples = []

    for example in tqdm(code_examples):
        prompt_field = example.get("prompt")

        # Step 1: Remove system prompt and extract user prompt content.
        user_prompt = ""
        if isinstance(prompt_field, list):
            # Filter out system messages
            non_system_msgs = [msg for msg in prompt_field if msg.get("role") != "system"]
            # Prefer the first user message if available
            for msg in non_system_msgs:
                if msg.get("role") == "user":
                    user_prompt = msg.get("content", "")
                    break
            # Fallback: concatenate all non-system message contents
            if not user_prompt:
                user_prompt = "\n".join(m.get("content", "") for m in non_system_msgs).strip()
        elif isinstance(prompt_field, str):
            # If prompt is a plain string, assume it already only contains the user prompt
            user_prompt = prompt_field
        else:
            # Unsupported format; skip this example
            continue

        # Preserve the original (untruncated) user prompt for final output
        raw_user_prompt = user_prompt

        # Step 2: Truncate user prompt at the earliest marker occurrence
        cut_positions = [user_prompt.find(m) for m in markers if user_prompt.find(m) != -1]
        if cut_positions:
            user_prompt = user_prompt[: min(cut_positions)].strip()
        else:
            # If no marker found to locate example, skip this sample
            continue
            
        # Step 3: Skip samples whose user prompt token length > 500 (by Qwen3 tokenizer)
        tokenized = tokenizer(user_prompt, add_special_tokens=False)
        if len(tokenized.get("input_ids", [])) > 500:
            continue

        new_sample = dict(example)
        SYSTEM_PROMPT_CONTENT = (
            "You are an expert Python programmer. You will be given a question (problem specification) "
            "and will generate a correct Python program that matches the specification and passes all tests."
            # "Format:\n- [Standalone] Make sure that your answer consists of only one Python function at the top level. "
            # "Do not wrap with a class or split into multiple functions."
        )
        new_sample["prompt"] = [
            {"role": "system", "content": SYSTEM_PROMPT_CONTENT},
            # Use original, untruncated user prompt in saved sample
            {"role": "user", "content": raw_user_prompt},
        ]

        # Compute segment count for the (cleaned) user prompt
        segments = _split_question_into_segments(user_prompt)
        # Filter out samples where any segment has fewer than 5 words
        if any(len(seg.strip().split()) < 5 for seg in segments):
            continue

        segment_count = len(segments)

        # Ensure extra_info is a dict and inject segment_count
        extra_info = new_sample.get("extra_info", {})
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except Exception:
                extra_info = {}
        if not isinstance(extra_info, dict):
            extra_info = {}
        extra_info["segment_count"] = segment_count
        new_sample["extra_info"] = extra_info

        processed_samples.append(new_sample)

    # Shuffle once to create reproducible splits
    random.shuffle(processed_samples)

    # Determine split sizes
    train_size = min(2000, len(processed_samples))
    eval_size = min(100, max(0, len(processed_samples) - train_size))

    train_samples = processed_samples[:train_size]
    eval_samples = processed_samples[train_size : train_size + eval_size]

    # Output files
    train_path = "data/eurus/eurus_code_examples_2000.json"
    eval_path = "data/eurus/eurus_code_examples_eval_100.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_samples, f, ensure_ascii=False, indent=2)

    # Also save Parquet files with matching names
    train_parquet = pathlib.Path(train_path).with_suffix(".parquet")
    eval_parquet = pathlib.Path(eval_path).with_suffix(".parquet")
    pd.DataFrame(train_samples).to_parquet(train_parquet, index=False)
    pd.DataFrame(eval_samples).to_parquet(eval_parquet, index=False)

    print(f"Total processed examples: {len(processed_samples)}")
    print(f"Saved {len(train_samples)} training examples to {train_path}")
    print(f"Saved {len(eval_samples)} evaluation examples to {eval_path}")
    print(f"And Parquet to {train_parquet} / {eval_parquet}")


if __name__ == "__main__":
    main()