import os
import re
import torch
import random
from openrlhf.remote_rm.utils import extract_gsm8k_numbers, extract_answer, answer_dict
from openrlhf.remote_rm.grader import grade_answer

strategy = "voting"

def get_full_traj(traj, tokenizer, actor, n_samples_per_prompt, prompt_max_len, enable_test_memory_mode, **generate_kwargs):
    generate_kwargs["num_return_sequences"] = n_samples_per_prompt
    if enable_test_memory_mode:
        input_ids = torch.randint(4, 100, (1, prompt_max_len)).to("cuda")
        attention_mask = torch.ones_like(input_ids)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        generate_kwargs["min_new_tokens"] = generate_kwargs["max_new_tokens"]
    else:
        inputs = tokenizer(traj, max_length=prompt_max_len, truncation=True, add_special_tokens=False, return_tensors="pt")
        inputs = {k: v.to(actor.model.device) for k, v in inputs.items()}
    sequences, _, _ = actor.generate(**inputs, **generate_kwargs)
    sequences_str = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    is_ends = [tokenizer.eos_token_id in o for o in sequences]
    return sequences, sequences_str, is_ends
    # return sequences

def get_scores(trajs, is_ends, tokenizer, critic, prompt_max_len, enable_test_memory_mode): # TODO: prompt_max_len, enable_test_memory_mode, remove sequence conversion
    trajs = [traj + (" " + tokenizer.eos_token if is_end else '') for traj, is_end in zip(trajs, is_ends)]
    inputs = tokenizer(trajs, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(critic.device) for k, v in inputs.items()}
    outputs = critic.compute_value(**inputs, return_dict=False)
    return outputs[0].squeeze()


def search(query, tokenizer, actor, critic, n_samples_per_prompt, prompt_max_len, enable_test_memory_mode, **generate_kwargs):
    trajs, trajs_str, is_ends = get_full_traj(query, tokenizer, actor, n_samples_per_prompt, prompt_max_len, enable_test_memory_mode, **generate_kwargs)
    scores = get_scores(trajs_str, is_ends, tokenizer, critic, prompt_max_len, enable_test_memory_mode)
    if strategy == "bestofn":
        best_idx = torch.argmax(scores)
        return [trajs[best_idx]]
    elif strategy == "bestandworst":
        best_idx = torch.argmax(scores)
        worse_idx = torch.argmin(scores)
        return [trajs[best_idx], trajs[worse_idx]]
    elif strategy == "random":
        return random.sample(trajs, 2)
    elif strategy == "voting":
        # question
        question = query[len("Question:"): -len("Answer: Let's think step by step\n")].strip()
        v = answer_dict.get(question, None)
        if v is None:
            return []
        predictions = {}
        ref, data_type = v["ref"], v["type"]
        for traj, traj_str, score in zip(trajs, trajs_str, scores.tolist()):
            hyp = extract_answer(traj_str, data_type)
            if hyp == "[INVALID]":
                continue
            if grade_answer(hyp, ref):
                score += 1
            flag = False
            for k in predictions:
                if grade_answer(k, hyp):
                    flag = True
                    predictions[k]["score"] += score
                    predictions[k]["trajs"].append(traj)
                    break
            if not flag:
                predictions[hyp] = {"score": score, "trajs": [traj]}
        sorted_trajs = sorted(predictions.values(), key=lambda x: x["score"], reverse=True)
        if len(sorted_trajs) == 0:
            return [random.choice(trajs), random.choice(trajs)], trajs
        
        selected_trajs = [random.choice(sorted_trajs[0]["trajs"]), random.choice(sorted_trajs[-1]["trajs"])]
        return selected_trajs, trajs
