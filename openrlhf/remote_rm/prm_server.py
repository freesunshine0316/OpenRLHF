import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F

model_name = "/apdcephfs_sh2/share_300000800/user/antewang/Qwen2.5-Math-PRM-7B"

def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).eval().cuda()

def split_query_and_response(text):
    q, a = text.split("\n\nAnswer: Let's think step by step\n")
    query = q.split("Question:")[1].strip()
    if a.endswith("<|endoftext|>"):
        a = a[:-len("<|endoftext|>")].strip()
    response = [step.strip() for step in a.split("\n")]
    return query, response

def compute_rewards(texts):
    sys_content = "Please reason step by step, and put your final answer within \\boxed{}."
    conversation_strs = []
    for text in texts:
        query, response = split_query_and_response(text)
        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": query},
            {"role": "assistant", "content": "<extra_0>".join(response) + "<extra_0>"},
        ]
        conversation_str = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        print((conversation_str,))
        conversation_strs.append(conversation_str)

    inputs = tokenizer(conversation_strs, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model(**inputs)

    step_sep_id = tokenizer.encode("<extra_0>")[0]
    token_masks = (inputs["input_ids"] == step_sep_id)
    step_rewards = make_step_rewards(outputs[0], token_masks)
    return [rewards[-1] for rewards in step_rewards]


app = FastAPI()

class InputText(BaseModel):
    texts: List[str]

class OutputPrediction(BaseModel):
    rewards: List[float]

@app.post("/predict", response_model=OutputPrediction)
async def predict(input_text: InputText):
    outputs = compute_rewards(input_text.texts)
    return {"rewards": outputs}