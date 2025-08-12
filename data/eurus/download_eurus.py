import random
import json
from datasets import load_dataset
import pandas as pd
import pathlib


def main():
    """Download Eurus-2-RL dataset, per-category sampling for train/test, save JSON and Parquet."""
    random.seed(42)

    # Load the training split of the dataset.
    dataset = load_dataset("PRIME-RL/Eurus-2-RL-Data", split="train")

    # Keep only examples where the 'ability' field is exactly 'code'.
    code_examples = dataset.filter(lambda example: example.get("ability") == "code")

    # Group by category (use data_source as category)
    source_to_examples = {}
    for example in code_examples:
        source = example.get("data_source")
        if source is None:
            continue
        source_to_examples.setdefault(source, []).append(example)

    train_samples_all = []
    test_samples_all = []

    # For each category, sample up to 1000 for train and 20 for test (non-overlapping)
    for src, examples in source_to_examples.items():
        examples_list = list(examples)
        random.shuffle(examples_list)

        train_n = min(250, len(examples_list))
        train_samples = examples_list[:train_n]

        remaining = examples_list[train_n:]
        test_n = min(20, len(remaining))
        test_samples = remaining[:test_n]

        train_samples_all.extend(train_samples)
        test_samples_all.extend(test_samples)

    # Output paths
    train_json = "data/eurus/eurus_code_by_datasource_train_250.json"
    test_json = "data/eurus/eurus_code_by_datasource_test_20.json"

    # Save JSON
    with open(train_json, "w", encoding="utf-8") as f:
        json.dump(train_samples_all, f, ensure_ascii=False, indent=2)
    with open(test_json, "w", encoding="utf-8") as f:
        json.dump(test_samples_all, f, ensure_ascii=False, indent=2)

    # Save Parquet with matching names
    train_parquet = pathlib.Path(train_json).with_suffix(".parquet")
    test_parquet = pathlib.Path(test_json).with_suffix(".parquet")
    pd.DataFrame(train_samples_all).to_parquet(train_parquet, index=False)
    pd.DataFrame(test_samples_all).to_parquet(test_parquet, index=False)

    print(f"Saved train: {len(train_samples_all)} to {train_json} and {train_parquet}")
    print(f"Saved test:  {len(test_samples_all)} to {test_json} and {test_parquet}")


if __name__ == "__main__":
    main()