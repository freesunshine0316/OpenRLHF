import re
import torch
import random
import ray
import jsonlines
import numpy as np
from openrlhf.remote_rm.grader import grade_answer
from openrlhf.search_algorithm.search_utils import DEFAULT_N, DEFAULT_TEMPERATURE, initialize_question_answer_map

answer_dict = initialize_question_answer_map()

def clean_pad_token(text, pad_token):
    return re.sub(pad_token, "", text)

def get_full_traj(traj, tokenizer, actor, **kwargs):
    input_ids = tokenizer(traj, return_tensors="pt")
    input_ids = {k: v.to(actor.model.device) for k, v in input_ids.items()}
    outputs = actor.model.generate(**input_ids,
        do_sample=True,
        max_new_tokens=1024,
        temperature=kwargs.get("temperature", DEFAULT_TEMPERATURE),
        tokenizer=tokenizer,
        num_return_sequences=kwargs.get("search_budget", DEFAULT_N),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True)
    sequences = tokenizer.batch_decode(outputs.sequences, keep_special_tokens=True)
    sequences = [clean_pad_token(seq, tokenizer.pad_token) for seq in sequences]
    cumulative_logprobs = outputs.scores
    avg_logprobs = None
    return sequences, cumulative_logprobs, avg_logprobs

def get_scores(trajs, tokenizer, critic):
    input_ids = tokenizer(trajs, return_tensors="pt", padding=True, truncation=True)
    input_ids = {k: v.to(critic.device) for k, v in input_ids.items()}
    outputs = critic.compute_value(**input_ids, return_dict=False)
    return (torch.clamp(outputs[0].squeeze(), min=-1, max=1) + 1) / 2

def get_full_trajs_vllm(all_prompts, tokenizer, llms, **kwargs):
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=kwargs.get("temperature", DEFAULT_TEMPERATURE),
        top_p=1,
        top_k=-1,
        max_tokens=1024,
        min_tokens=1,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
        logprobs = 0,
    )
    all_prompt_token_ids = tokenizer(all_prompts, add_special_tokens=False, max_length=1024, truncation=True,)["input_ids"]

    # Distribute requests to engines and collect responses to outputs
    all_output_refs = []
    batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
    for i, llm in enumerate(llms):
        prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
        if prompt_token_ids:
            all_output_refs.append(
                llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
            )

    # Retrieve and combine results from all outputs
    all_outputs = sum(ray.get(all_output_refs), [])
    sequences = [output.outputs[0].text for output in all_outputs]
    cumulative_logprobs = [output.outputs[0].cumulative_logprob for output in all_outputs]
    avg_logprobs = [sum(list(item.values())[0].logprob for item in output.outputs[0].logprobs) / len(output.outputs[0].logprobs) for output in all_outputs] # if the same logprob, prefer longer sequence
    return sequences, cumulative_logprobs, avg_logprobs

def extract_numbers(text):
    ANS_RE = re.compile(r"(\-?[0-9\.\,]+)")
    match = ANS_RE.search(text)
    try:
        return match.group(1).strip()
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
        if len(content) == 0:
            return "[INVALID]"
        if content[-1] == ".":
            content = content[:-1]
        if data_type == "gsm8k":
            content = extract_numbers(content)
        return content
    return "[INVALID]"

def clean_eos(traj, eos_token):
    if traj.endswith(eos_token):
        return traj[:-len(eos_token)]
    else:
        return traj

def search(query, tokenizer, actor, critic, **kwargs):
    trajs = get_full_traj(query, tokenizer, actor)
    scores = get_scores(trajs, tokenizer, critic)
    strategy = kwargs["search_strategy"]
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
        has_gold = False
        for traj, score in zip(trajs, scores.tolist()):
            hyp = extract_answer(clean_eos(traj, tokenizer.eos_token), data_type)
            if hyp == "[INVALID]":
                continue
            if grade_answer(hyp, ref):
                score += 1
                has_gold = True
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
        if len(sorted_trajs) == 1 or not has_gold:
            return [random.choice(sorted_trajs[0]["trajs"])]
        else:
            return [random.choice(_sorted_trajs["trajs"]) for _sorted_trajs in sorted_trajs[:2]]

def search_vllm(queries, tokenizer, actor, critic=None, search_args=None):
    if not isinstance(queries, list):
        queries = [queries]
    # Expand prompt list based on the number of samples per prompt
    search_budget = search_args.get("search_budget", DEFAULT_N)
    all_prompts = sum([[prompt] * search_budget for prompt in queries], [])
    trajs, cumulative_logprobs, avg_logprobs = get_full_trajs_vllm(all_prompts, tokenizer, actor)
    chosen_trajs = []
    for i in range(0, len(all_prompts), search_budget):
        if search_args["compute_reward_strategy"] == "cumulative":
            _rewards = cumulative_logprobs[i:i+search_budget]
        elif search_args["compute_reward_strategy"] == "average":
            _rewards = avg_logprobs[i:i+search_budget]
        else:
            raise Exception("Invalid compute_reward_strategy")

        _trajs = trajs[i:i+search_budget]
        if search_args["voting"]:
            predictions = {} # we do not remove [INVALID], as the model has to learn good format
            for traj, reward in zip(_trajs, _rewards):
                hyp = extract_answer(clean_eos(traj, tokenizer.eos_token), "math")
                found_cluster = False
                for k in predictions:
                    if grade_answer(k, hyp):
                        found_cluster = True
                        predictions[k]["score"] += 1 # hard voting
                        predictions[k]["trajs"].append({"text": traj, "reward": reward})
                        break
                if not found_cluster:
                    predictions[hyp] = {"score": 1, "trajs": [{"text": traj, "reward": reward}]}
            sorted_trajs = sorted(predictions.values(), key=lambda x: x["score"], reverse=True)
            best_trajs = sorted_trajs[0]["trajs"]
            best_idx = np.argmax(np.array([_traj["reward"] for _traj in best_trajs]))
            chosen_trajs.append(best_trajs[best_idx]["text"])
        else:
            best_idx = np.argmax(np.array(_rewards))
            chosen_trajs.append(_trajs[best_idx])
    return chosen_trajs
