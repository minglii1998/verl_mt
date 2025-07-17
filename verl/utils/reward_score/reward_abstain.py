
def compute_score_reward_abstain(solution_str, ground_truth=None) -> float:
    """
    若模型在最后一个 \\boxed… 片段内显式包含 'abstain'（大小写不敏感），返回 1.0；否则返回 0.0。
    ground_truth 参数仅为兼容旧接口，当前不使用。
    """
    try:
        boxed = last_boxed_only_string(solution_str)
        if boxed is None:
            return 0.0
        inner = remove_boxed(boxed).strip().lower()
        return 1.0 if "abstain" in inner else 0.0
    except Exception as e:
        # print(e)  
        return 0.0


def last_boxed_only_string(string: str):
    """
    提取 **最后一个** \\boxed… 或 \\boxed … 片段（若不存在则返回 None）。
    支持两种 LaTeX 写法：
      1. \\boxed{...}
      2. \\boxed ...   （空格后直接内容，无大括号）
    """
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        # 处理带空格形式
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]

    if idx < 0:
        idx = string.rfind("\\fbox")  # 兼容 \fbox{} 写法
        if idx < 0:
            return None

    # 寻找对应右花括号
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    """
    去掉 \\boxed 包装，返回内部字符串。
    支持两种写法：
      1. \\boxed {...}
      2. \\boxed ...   （空格后直接内容）
    """
    if "\\boxed " in s:
        prefix = "\\boxed "
        assert s.startswith(prefix)
        return s[len(prefix) :]

    prefix = "\\boxed{"
    assert s.startswith(prefix) and s.endswith("}")
    return s[len(prefix) : -1]
