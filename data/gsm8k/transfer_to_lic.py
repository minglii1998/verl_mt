import pandas as pd
import json
import random
import re

def _split_question_into_segments(question: str, *, shuffle: bool = False):
        """
        Split question into segments by sentence delimiters and remove trailing commas/semicolons.
        If shuffle=True, randomly shuffle the order before returning.
        """

        delimiter_pattern = (
            r"([。！？!?；;:]|"        # Full/half-width sentence endings
            r"(?<!\d)[,，](?!\d)|"    # Commas, excluding digit separators
            r"(?<!\d)\.(?!\d))"       # Periods, excluding decimals
        )

        parts = re.split(delimiter_pattern, question)
        segments, buf = [], ""

        for frag in parts:
            if not frag:
                continue
            if re.match(delimiter_pattern, frag):
                buf += frag
                seg = re.sub(r"[，,；;]+$", "", buf.strip())  # Remove trailing commas/semicolons
                if seg:
                    segments.append(seg)
                buf = ""
            else:
                buf += frag

        if buf.strip():
            segments.append(re.sub(r"[，,；;]+$", "", buf.strip()))

        if shuffle:
            random.shuffle(segments)

        return segments or [question]

def process_split(input_path: str):
        df = pd.read_parquet(input_path)

        # orient="records" 让每行是一个 JSON 对象；lines=True 方便下游流式处理
        # 使用 df.to_json 再 json.loads，可自动处理日期等类型为 ISO 字符串
        records = json.loads(
            df.to_json(orient="records", date_format="iso", force_ascii=False)
        )

        # 修改 prompt
        system_message = {
            "role": "system",
            "content": "As an expert problem solver solve step by step the following mathematical questions. Put your final answer within \\boxed{}."
        }

        remove_phrase = "Let's think step by step and output the final answer after \"####\"."

        for record in records:

            record["data_source"] = "lic_math"

            prompt_list = record.get("prompt", [])
            # 若 prompt 被序列化为字符串，尝试反序列化
            if isinstance(prompt_list, str):
                try:
                    prompt_list = json.loads(prompt_list)
                except Exception:
                    prompt_list = []

            # 在开头插入 system prompt
            prompt_list.insert(0, system_message)

            # 移除 user content 中的指定短语
            for msg in prompt_list:
                if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    msg["content"] = msg["content"].replace(remove_phrase, "").strip()

            record["prompt"] = prompt_list

            # 计算最后一个 user message 的分段数量，并存入 extra_info.segment_count
            user_msgs = [m for m in prompt_list if m.get("role") == "user" and isinstance(m.get("content"), str)]
            if user_msgs:
                last_user_msg = user_msgs[-1]
                segments = _split_question_into_segments(last_user_msg["content"])
                segment_count = len(segments)
            else:
                segment_count = 0

            # 更新或创建 extra_info 字段
            extra_info = record.get("extra_info", {})
            if isinstance(extra_info, str):
                try:
                    extra_info = json.loads(extra_info)
                except Exception:
                    extra_info = {}
            if not isinstance(extra_info, dict) or extra_info is None:
                extra_info = {}
            extra_info["segment_count"] = segment_count
            record["extra_info"] = extra_info

        # 判断是train还是test，根据文件名
        if "test" in input_path:
            sample_size = 100
        else:
            sample_size = 2000

        if len(records) > sample_size:
            # 先过滤出 segment 数量大于 1 的样本
            filtered_records = []
            for rec in records:
                # 找到最后一个 user message
                prompt_list = rec.get("prompt", [])
                if isinstance(prompt_list, str):
                    try:
                        prompt_list = json.loads(prompt_list)
                    except Exception:
                        prompt_list = []
                user_msgs = [m for m in prompt_list if m.get("role") == "user" and isinstance(m.get("content"), str)]
                if user_msgs:
                    last_user_msg = user_msgs[-1]
                    segments = _split_question_into_segments(last_user_msg["content"])
                    if len(segments) > 1:
                        filtered_records.append(rec)

            # 如果过滤后数量不足，补充剩余的
            if len(filtered_records) >= sample_size:
                records = random.sample(filtered_records, sample_size)
            else:
                # 补充 segment 数量 <=1 的样本
                remaining = [rec for rec in records if rec not in filtered_records]
                needed = sample_size - len(filtered_records)
                records = filtered_records + random.sample(remaining, min(needed, len(remaining)))

        # 按 segment_count 从大到小排序
        records.sort(key=lambda rec: rec.get("extra_info", {}).get("segment_count", 0), reverse=False)

        # 保存修改后的数据
        output_path = f"data/gsm8k/{input_path.split('/')[-1].split('.')[0]}_lic_format_{sample_size}.json"
        # 保存 JSONLines
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False, indent=2))
                f.write("\n")

        # 保存为新的 Parquet 文件
        parquet_output_path = f"data/gsm8k/{input_path.split('/')[-1].split('.')[0]}_lic_format_{sample_size}.parquet"
        pd.DataFrame(records).to_parquet(parquet_output_path, index=False)

if __name__ == "__main__":
        for split in ["train", "test"]:
            input_path = f"data/gsm8k/{split}.parquet"
            process_split(input_path)
