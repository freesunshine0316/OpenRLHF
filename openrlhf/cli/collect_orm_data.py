import argparse
import os
import numpy as np
import json
from openrlhf.models.ray_distributed_actors import DistributedSamplerVLLM
from openrlhf.remote_rm.grader import grade_answer
from openrlhf.remote_rm.utils import extract_answer, extract_gsm8k_numbers
from collections import defaultdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_workers_per_gpu", type=int, default=1)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--input_key", type=str, default="question", help="JSON dataset key")
    parser.add_argument("--answer_key", type=str, default="gt_answer", help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    data_list = [json.loads(line) for line in open(args.data_path).read().strip().split('\n')]
    prompt_list = [args.input_template.format(item[args.input_key]) for item in data_list]

    sampler = DistributedSamplerVLLM(
        num_gpus=args.num_gpus, 
        num_workers_per_gpu=args.num_workers_per_gpu,
        model_path=args.model_path,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
    )
    responses = sampler.generate(
        prompt_list,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        num_samples=args.num_samples,
    )

    samples = []
    for instance, hyp_list in zip(data_list, responses):
        question = instance[args.input_key]
        ref = instance[args.answer_key]
        hyp_list = list(set(hyp_list))
        for hyp in hyp_list:
            ex_hyp = extract_answer(hyp, 'math')
            if ex_hyp == "[INVALID]":
                label = -1
            if grade_answer(ex_hyp, ref):
                label = 1
            else:
                label = -1
            samples.append({
                'input': args.input_template.format(question),
                'question': question,
                'answer': hyp,
                'gt': ref,
                'label': label,
            })

    with open(args.output_path, 'w') as f:
        f.writelines([json.dumps(sample) + '\n' for sample in samples])
    print(f"Save to {args.output_path}")
