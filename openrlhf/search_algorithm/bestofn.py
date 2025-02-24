import os
import re
import torch
import random
from openrlhf.remote_rm.utils import extract_gsm8k_numbers, extract_answer, answer_dict
from openrlhf.remote_rm.grader import grade_answer



def search(prompts, sampling_fn, compute_score_block_fn, tokenizer, n_samples_per_prompt, strategy):
    """
    Generation: if not vllm, with a batch size of `args.micro_rollout_batch_size`
    Compute rewards (forward): with a batch size of `args.micro_rollout_batch_size`

    Selection and return in a list of data, each data:
        {
            "sequence": tensor of shape (seq_len,),
            "prompt_ids": tensor of shape (prompt_len,),
            "response_ids": tensor of shape (response_len,),
            "all_sequences": list [args.n_samples_per_prompt], each item is a tensor of shape (seq_len,),
            "all_response_ids": list [args.n_samples_per_prompt], each item is a tensor of shape (response_len,)
        }
    """
    n_prompt = len(prompts)

    all_prompts = [prompt for _ in range(n_samples_per_prompt) for prompt in prompts] # e.g. [p1, p2, p3, p4, p1, p2, p3, p4, ...]
    all_sequences_list, all_prompt_ids_list, all_response_ids_list = sampling_fn(all_prompts) # shape of each item in all_sequences_list: micro_rollout_batch_size

    all_scores_list = []
    for sequences, prompt_ids in zip(all_sequences_list, all_prompt_ids_list):
        rewards = compute_score_block_fn(sequences, len(prompt_ids[0]))
        all_scores_list.append(rewards.tolist())

    micro_rollout_batch_size = all_sequences_list[0].shape[0]
    n_micro_block = n_prompt * n_samples_per_prompt // micro_rollout_batch_size
    assert len(all_sequences_list) == len(all_scores_list) == n_micro_block

    # resize to [n_prompt, n_samples_per_prompt]
    all_sequences_list = [seq for micro_sequences in all_sequences_list for seq in micro_sequences]
    all_sequences_list = [[all_sequences_list[i + j * n_prompt] for j in range(n_samples_per_prompt)] for i in range(n_prompt)]

    all_prompt_ids_list = [seq for micro_prompt_ids in all_prompt_ids_list for seq in micro_prompt_ids]
    all_prompt_ids_list = [[all_prompt_ids_list[i + j * n_prompt] for j in range(n_samples_per_prompt)] for i in range(n_prompt)]

    all_response_ids_list = [seq for micro_response_ids in all_response_ids_list for seq in micro_response_ids]
    all_response_ids_list = [[all_response_ids_list[i + j * n_prompt] for j in range(n_samples_per_prompt)] for i in range(n_prompt)]

    all_scores_list = [score for micro_scores in all_scores_list for score in micro_scores]
    all_scores_list = [[all_scores_list[i + j * n_prompt] for j in range(n_samples_per_prompt)] for i in range(n_prompt)]

    # selection
    examples = []
    for question, sequences, prompt_ids, response_ids, scores in zip(prompts, all_sequences_list, all_prompt_ids_list, all_response_ids_list, all_scores_list):
        if strategy == "best":
            best_idx = torch.argmax(torch.tensor(scores))
            examples.append({
                "sequence": sequences[best_idx],
                "prompt_ids": prompt_ids[best_idx],
                "response_ids": response_ids[best_idx],
                "all_sequences": sequences,
                "all_response_ids": response_ids,
            })

        elif strategy == "best2":
            indices = torch.topk(torch.tensor(scores), k=2, dim=0, largest=True).indices
            examples.append({
                "sequence": sequences[indices[0]],
                "prompt_ids": prompt_ids[indices[0]],
                "response_ids": response_ids[indices[0]],
                "all_sequences": sequences,
                "all_response_ids": response_ids,
            })
            examples.append({
                "sequence": sequences[indices[1]],
                "prompt_ids": prompt_ids[indices[1]],
                "response_ids": response_ids[indices[1]],
                "all_sequences": sequences,
                "all_response_ids": response_ids,
            })

        elif strategy == "bestandworst":
            best_idx = torch.argmax(torch.tensor(scores))
            worse_idx = torch.argmin(torch.tensor(scores))
            examples.append({
                "sequence": sequences[best_idx],
                "prompt_ids": prompt_ids[best_idx],
                "response_ids": response_ids[best_idx],
                "all_sequences": sequences,
                "all_response_ids": response_ids,
            })
            examples.append({
                "sequence": sequences[worse_idx],
                "prompt_ids": prompt_ids[worse_idx],
                "response_ids": response_ids[worse_idx],
                "all_sequences": sequences,
                "all_response_ids": response_ids,
            })

        elif strategy == "random":
            indices = random.sample(range(n_samples_per_prompt), 1)
            examples.append({
                "sequence": sequences[indices[0]],
                "prompt_ids": prompt_ids[indices[0]],
                "response_ids": response_ids[indices[0]],
                "all_sequences": sequences,
                "all_response_ids": response_ids,
            })

        elif strategy == "random2":
            indices = random.sample(range(n_samples_per_prompt), 2)
            examples.append({
                "sequence": sequences[indices[0]],
                "prompt_ids": prompt_ids[indices[0]],
                "response_ids": response_ids[indices[0]],
                "all_sequences": all_sequences,
                "all_response_ids": all_response_ids,
            })
            examples.append({
                "sequence": sequences[indices[1]],
                "prompt_ids": prompt_ids[indices[1]],
                "response_ids": response_ids[indices[1]],
                "all_sequences": all_sequences,
                "all_response_ids": all_response_ids,
            })

        elif strategy == "softvoting":
            question = question[len("Question:"): -len("Answer: Let's think step by step\n")].strip()
            v = answer_dict.get(question, None)
            if v is None:
                return []
            predictions = {}
            ref, data_type = v["ref"], v["type"]

            sequences_str = [tokenizer.decode(s, skip_special_tokens=True) for s in sequences]
            for i, (traj_str, score) in enumerate(zip(sequences_str, scores)):
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
                        predictions[k]["indices"].append(i)
                        break
                if not flag:
                    predictions[hyp] = {"score": score, "indices": [i]}

            sorted_predictions = sorted(predictions.values(), key=lambda x: x["score"], reverse=True)
            if len(sorted_predictions) == 0:
                indices = random.sample(range(n_samples_per_prompt), 2)
            else:
                indices = [random.choice(sorted_predictions[0]["indices"]), random.choice(sorted_predictions[-1]["indices"])]
            examples.append({
                "sequence": sequences[indices[0]],
                "prompt_ids": prompt_ids[indices[0]],
                "response_ids": response_ids[indices[0]],
                "all_sequences": sequences,
                "all_response_ids": response_ids,
            })
            examples.append({
                "sequence": sequences[indices[1]],
                "prompt_ids": prompt_ids[indices[1]],
                "response_ids": response_ids[indices[1]],
                "all_sequences": sequences,
                "all_response_ids": response_ids,
            })
    
    return examples
