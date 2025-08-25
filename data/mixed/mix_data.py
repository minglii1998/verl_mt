import json
import os
import random
from pathlib import Path

import pandas as pd


def concat_parquet(input_files, output_path):
    """Concatenate multiple parquet files and save the result.

    Parameters
    ----------
    input_files : list[Path]
        Parquet files to concatenate.
    output_path : Path
        Destination parquet file.
    """
    dfs = []
    for file in input_files:
        if not file.exists():
            raise FileNotFoundError(f"Parquet file not found: {file}")
        dfs.append(pd.read_parquet(file))

    merged_df = pd.concat(dfs, ignore_index=True)
    # Randomly shuffle rows
    merged_df = merged_df.sample(frac=1).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(output_path, index=False)



def concat_json(input_files, output_path):
    """Concatenate JSON arrays stored in files and save the result.

    Each input JSON file is expected to contain a single JSON array
    (list) at the top level. The arrays from all input files are
    concatenated and written as one array to *output_path*.
    """
    def _load_json_list(path: Path):
        """Load a list of JSON objects from *path*.

        Supports two formats:
        1. A single JSON array (standard).
        2. Newline-delimited JSON (one JSON object per line / NDJSON).
        If the first method fails due to JSONDecodeError, it falls back to
        NDJSON parsing."""
        try:
            with path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
                if isinstance(data, list):
                    return data
                # if it's a dict or other, wrap in list
                return [data]
        except json.JSONDecodeError:
            # Fallback: read line by line
            items = []
            with path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Skip invalid lines or raise – here choose to skip with warning
                        # but still continue parsing others.
                        continue
            return items

    combined = []
    for file in input_files:
        if not file.exists():
            raise FileNotFoundError(f"JSON file not found: {file}")
        combined.extend(_load_json_list(file))

    if combined:
        random.shuffle(combined)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(combined, fp, ensure_ascii=False, indent=2)



def main():
    # Locate directories relative to this script file
    data_dir = Path(__file__).resolve().parents[1]  # .../data
    eurus_dir = data_dir / "eurus"
    gsm_dir = data_dir / "gsm8k"
    mixed_dir = data_dir / "mixed"

    # Train files (1000 + 1000 -> 2000)
    train_parquets = [
        eurus_dir / "eurus_code_examples_1000.parquet",
        gsm_dir / "gsm_math_train_1000.parquet",
    ]
    train_jsons = [
        eurus_dir / "eurus_code_examples_1000.json",
        gsm_dir / "gsm_math_train_1000.json",
    ]
    train_parquet_out = mixed_dir / "mixed_train_2000.parquet"
    train_json_out = mixed_dir / "mixed_train_2000.json"

    # Test files (100 + 100 -> 200)
    test_parquets = [
        eurus_dir / "eurus_code_examples_eval_100.parquet",
        gsm_dir / "gsm_math_test_100.parquet",
    ]
    test_jsons = [
        eurus_dir / "eurus_code_examples_eval_100.json",
        gsm_dir / "gsm_math_test_100.json",
    ]
    test_parquet_out = mixed_dir / "mixed_test_200.parquet"
    test_json_out = mixed_dir / "mixed_test_200.json"

    # Perform concatenations
    concat_parquet(train_parquets, train_parquet_out)
    concat_parquet(test_parquets, test_parquet_out)
    concat_json(train_jsons, train_json_out)
    concat_json(test_jsons, test_json_out)

    print("Merging complete.")


if __name__ == "__main__":
    main()
