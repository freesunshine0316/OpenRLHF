import os
import traceback
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from collections import defaultdict
import jsonlines
import re
from openrlhf.remote_rm.grader import grade_answer
from openrlhf.remote_rm.utils import extract_gsm8k_numbers, extract_answer, answer_dict
from openrlhf.remote_rm.qwen_math_eval_toolkit.grader import math_equal as qwen_math_equal
from openrlhf.remote_rm.qwen_math_eval_toolkit.parser import extract_answer as qwen_extract_answer
from multiprocessing import Process, Queue


def get_reward_r1_zero(sequences, queries, responses):
    rewards = []
    for sequence in sequences:
        try:
            q = sequence.split("\nUser: ", 1)[1]
            q = q.split(" Show your work in great details.", 1)[0].strip()
            answer_pattern = r'<answer>(.*?)</answer>'
            match = re.search(answer_pattern, sequence)
            if not match:
                rewards.append(-1.0)
                continue
            a = match.group(1).strip()
            if a == "":
                print(f"!!! Empty answer: {sequence}")
                rewards.append(-1.0)
                continue
            v = answer_dict.get(q, None)
            if v is None:
                print(f"!!! Unmatched question: {q}")
                rewards.append(-1.0)
                continue
            ref, data_type = v["ref"], v["type"]
            hyp = extract_answer(a, data_type)
            if hyp == "[INVALID]":
                print(f"!!! Fail to extract answer from {a} for data_type {data_type}")
                rewards.append(-1.0)
                continue
            if grade_answer(hyp, ref):
                rewards.append(1.0)
            else:
                rewards.append(-0.1)
            print((a, ref, rewards[-1]))
        except Exception as e:
            print(e)
            rewards.append(-1.0)
    return rewards


def qwen_math_equal_subprocess(prediction, reference,  timeout_seconds=10):
    def worker(q, prediction, reference):
        result = qwen_math_equal(prediction=prediction, reference=reference, timeout=False)
        q.put(result)

    q = Queue()
    p = Process(target=worker, args=(q, prediction, reference))
    p.start()

    # 添加超时处理
    p.join(timeout=timeout_seconds)  # 等待进程完成，最多等待 timeout_seconds 秒

    # 如果进程还在运行，则终止它并返回 False
    if p.is_alive():
        p.terminate()
        p.join()  # 确保进程被完全清理
        return False

    # 如果进程正常完成，获取结果
    try:
        return q.get_nowait()
    except:
        return False


def get_reward_qwen_math(sequences, queries, responses):
    rewards = []
    for sequence in sequences:
        try:
            query, model_output = sequence.split("<|im_end|>\n<|im_start|>assistant")
            question = query.split("<|im_start|>user")[1].strip()
            model_output = model_output.strip()
            if question not in answer_dict:
                print(f"!!! Unmatched question: {question}")
                rewards.append(-1.0)
                continue

            stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
            for stop_word in stop_words:
                if stop_word in model_output:
                    model_output = model_output.split(stop_word)[0].strip()

            if "boxed" not in model_output:
                print(f"!!! No 'boxed' found: {model_output}")
                box_match = -1.0
            else:
                extract_answer = qwen_extract_answer(model_output, data_name="math")
                answer = answer_dict[question]["ref"]
                if qwen_math_equal_subprocess(prediction=extract_answer, reference=answer):
                    box_match = 1.0
                else:
                    box_match = -0.5
            rewards.append(box_match)
        except Exception as e:
            print(f"!!!Exception: {e}")
            rewards.append(-1.0)
    rewards_dict = defaultdict(int)
    for r in rewards:
        rewards_dict[r] += 1
    print(f"!!! Reward Mean: {sum(rewards) / (len(rewards) + 1e-5)}, Distribution: {rewards_dict}")
    return rewards



def get_reward(sequences, queries, responses):
    """
    reward: 1 if the answer is correct, -1 if the answer is incorrect, -100 if the answer is invalid
    """
    rewards = []
    for q, response in zip(queries, responses):
        try:
            q = q.split('Answer:')[0].strip().split('Question:')[-1].strip()
            v = answer_dict.get(q, None)
            if v is None:
                rewards.append(-1)
                continue
            ref, data_type = v["ref"], v["type"]
            hyp = extract_answer(response, data_type)
            if hyp == "[INVALID]":
                rewards.append(-1)
                print((hyp, ref, rewards[-1]))
                continue
            if grade_answer(hyp, ref):
                rewards.append(1)
            else:
                rewards.append(-1)
            print((hyp, ref, rewards[-1]))
        except Exception as e:
            print(e)
            rewards.append(-1)
    return rewards

app = FastAPI()

class InputText(BaseModel):
    sequence: List[str]
    query: List[str]
    response: List[str]

class OutputPrediction(BaseModel):
    rewards: List[float]

@app.post("/predict", response_model=OutputPrediction)
async def predict(input_text: InputText):
    return {"rewards": get_reward(input_text.sequence, input_text.query, input_text.response)}

@app.post("/predict_r1_zero", response_model=OutputPrediction)
async def predict(input_text: InputText):
    return {"rewards": get_reward_r1_zero(input_text.sequence, input_text.query, input_text.response)}

@app.post("/predict_qwen_math", response_model=OutputPrediction)
async def predict(input_text: InputText):
    return {"rewards": get_reward_qwen_math(input_text.sequence, input_text.query, input_text.response)}
