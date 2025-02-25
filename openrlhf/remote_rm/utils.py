import os
import jsonlines
import re


def standardize_value_str(x: str) -> str:
    """Standardize numerical values"""
    y = x.replace(",", "")
    if '.' in y:
        y = y.rstrip('0')
        if y[-1] == '.':
            y = y[:-1]
    if not len(y):
        return "[INVALID]"
    if y[0] == '.':
        y = '0' + y
    if y[-1] == '%':
        y = str(eval(y[:-1]) / 100)
    return y.rstrip('.')

def extract_gsm8k_numbers(text):
    ANS_RE = re.compile(r"(\-?[0-9\.\,]+)")
    match = ANS_RE.search(text)
    try:
        return standardize_value_str(match.group(1).strip())
    except:
        return "[INVALID]"

def extract_answer(text, data_type):
    if "he answer is:" in text:
        text_split = text.split("he answer is:")
    elif "he answer is" in text:
        text_split = text.split("he answer is")
    else:
        return "[INVALID]"

    if len(text_split) == 2:
        content = text_split[-1].strip()
        if len(content) > 0 and content[-1] == ".":
            content = content[:-1]
        if data_type == "gsm8k":
            content = extract_gsm8k_numbers(content)
        if data_type == "aime2024":
            content = extract_gsm8k_numbers(content)
        return content
    return "[INVALID]"


if os.path.exists("/apdcephfs_sh2/share_300000800/user/antewang/Qwen2.5-Math/evaluation/data/gsm8k/train.jsonl"):
    data_fpath_list = [
        "/apdcephfs_sh2/share_300000800/user/antewang/Qwen2.5-Math/evaluation/data/gsm8k/train.jsonl",
        "/apdcephfs_sh2/share_300000800/user/antewang/Qwen2.5-Math/evaluation/data/math/train.jsonl",
        "/apdcephfs_sh2/share_300000800/user/antewang/Qwen2.5-Math/evaluation/data/gsm8k/test.jsonl",
        "/apdcephfs_sh2/share_300000800/user/antewang/Qwen2.5-Math/evaluation/data/math/test.jsonl"
    ]
else:
    data_fpath_list = [
        "/apdcephfs_qy3/share_301812049/oakyu/exp.tencent_chat/Qwen2.5-Math-evaluation/data/gsm8k/train.jsonl",
        "/apdcephfs_qy3/share_301812049/oakyu/exp.tencent_chat/Qwen2.5-Math-evaluation/data/math/train.jsonl",
        "/apdcephfs_qy3/share_301812049/oakyu/exp.tencent_chat/Qwen2.5-Math-evaluation/data/gsm8k/test.jsonl",
        "/apdcephfs_qy3/share_301812049/oakyu/exp.tencent_chat/Qwen2.5-Math-evaluation/data/math/test.jsonl"
    ]

dataset = []
for data_fpath in data_fpath_list:
    with jsonlines.open(data_fpath, 'r') as reader:
        for item in reader:
            if "gsm8k" in data_fpath:
                question = item["question"]
                answer = extract_gsm8k_numbers(item["answer"].split("####")[-1].strip())
                dataset.append({"question": question, "answer": answer, "type": "gsm8k"})
            else:
                question = item["problem"]
                answer = item["answer"]
                dataset.append({"question": question, "answer": answer, "type": "math"})

answer_dict = {item["question"].strip(): {"ref": item["answer"], "type": item["type"]} for item in dataset}
