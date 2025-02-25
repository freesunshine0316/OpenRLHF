import os
import sys
import ray
import math
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from openrlhf.models import get_llm_for_sequence_regression
from openrlhf.datasets.utils import zero_pad_sequences
from openrlhf.utils import get_tokenizer


@ray.remote(num_gpus=1)
class SamplerVLLM:
    def __init__(
        self,
        model_path: str,
        dtype: str,
        max_model_len: int,
    ):
        self.llm = LLM(model=model_path, dtype=dtype, tensor_parallel_size=1, max_model_len=max_model_len)
        self.tokenizer = self.llm.get_tokenizer()

    def generate(
        self, 
        prompt_list,
        temperature=1,
        top_p=1,
        max_new_tokens=None,
        num_samples=1,
    ):
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, n=num_samples)

        input_ids = self.tokenizer([prompt for prompt in prompt_list], add_special_tokens=False).input_ids
        results = self.llm.generate(
            prompt_token_ids=input_ids, 
            sampling_params=sampling_params,
            use_tqdm=True,
        )
        responses = [[o.text.strip() for o in result.outputs] for result in results]
        return responses

class DistributedSamplerVLLM:
    def __init__(
        self, 
        num_gpus, 
        num_workers_per_gpu,
        model_path: str,
        dtype: str,
        max_model_len: int,
    ):
        vllm_port = int(os.environ.get("VLLM_PORT", 0))

        num_cpus = os.environ.get("RAYLET_CPUS", None)
        num_cpus = int(num_cpus) if num_cpus is not None else None
        ray.init(num_cpus=num_cpus)

        num_workers_per_gpu = 1 if num_workers_per_gpu is None else num_workers_per_gpu
        num_workers = num_workers_per_gpu * num_gpus
        num_gpus_per_worker = 1 / num_workers_per_gpu

        samplers = []
        from ray.runtime_env import RuntimeEnv
        for i in range(num_workers):
            runtime_env = RuntimeEnv(env_vars={"VLLM_PORT": str(vllm_port + 2*i)})
            sampler = SamplerVLLM.options(num_gpus=num_gpus_per_worker, runtime_env=runtime_env).remote(
                model_path=model_path,
                dtype=dtype,
                max_model_len=max_model_len,
            )
            samplers.append(sampler)
        ray.wait([sampler.__ray_ready__.remote() for sampler in samplers])

        self.num_workers = len(samplers)
        self.samplers = samplers
        self.sampler_pool = ray.util.actor_pool.ActorPool(samplers)

    def shutdown(self):
        ray.shutdown()

    def generate(
        self, 
        prompt_list,
        temperature=0,
        top_p=1,
        max_new_tokens=None,
        num_samples=1,
    ):
        try:
            piece = math.ceil(len(prompt_list) / self.num_workers)
            split_prompts = [
                prompt_list[i * piece: min((i+1) * piece, len(prompt_list))] 
                for i in range(self.num_workers)
            ]
            split_prompts = [piece for piece in split_prompts if piece != []]

            results = list(
                self.sampler_pool.map(
                    lambda p, x: p.generate.remote(
                        x,
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_new_tokens,
                        num_samples=num_samples,
                    ), 
                    split_prompts
                )
            )
            results = [item for res in results for item in res]

        except ray.exceptions.RayActorError as ex:
            print(ex)
            sys.exit(1)

        return results

@ray.remote(num_gpus=1)
class Scorer:
    def __init__(
        self,
        model_path: str,
        flash_attn: bool,
        bf16: bool,
        load_in_4bit: bool = False,
        lora_rank: bool = False,
        lora_alpha: int = 16,
        lora_dropout: float = 0,
        packing_samples: bool = False,
    ):
        self.model = get_llm_for_sequence_regression(
            model_path,
            "critic",
            use_flash_attention_2=flash_attn,
            bf16=bf16,
            load_in_4bit=load_in_4bit,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            packing_samples=packing_samples,
            value_head_prefix="score",
            init_value_head=False,
        )
        self.model.eval().cuda()
        self.tokenizer = get_tokenizer(model_path, self.model, padding_side="right")

    def collate_fn(self, instances):
        seq_ids, attention_mask = tuple([instance[key] for instance in instances] for key in ("seq_ids", "attention_mask"))
        
        padding_side = "right"
        seq_ids = zero_pad_sequences(seq_ids, side=padding_side, value=self.tokenizer.pad_token_id)
        attention_mask = zero_pad_sequences(attention_mask, side=padding_side)
        return seq_ids, attention_mask

    def scoring(
        self, 
        prompt_list,
        response_list,
        batch_size,
    ):
        input_ids = self.tokenizer(prompt_list, add_special_tokens=False).input_ids
        output_ids = self.tokenizer(response_list, add_special_tokens=False).input_ids

        seq_ids = [torch.concat([torch.tensor(inp), torch.tensor(oup)]) for inp, oup in zip(input_ids, output_ids)]
        attention_mask = [torch.ones_like(x) for x in seq_ids]
        samples = [
            {
                "seq_ids": s, 
                "attention_mask": a
            }
            for s, a in zip(seq_ids, attention_mask)
        ]

        dataloader = DataLoader(
            samples,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

        all_scores = []
        for data in tqdm(dataloader, total=len(dataloader)):
            seq_ids, attention_mask = data
            seq_ids = seq_ids.to("cuda")
            attention_mask = attention_mask.to("cuda")
            scores = self.model.compute_reward(
                input_ids=seq_ids,
                attention_mask=attention_mask,
            )
            all_scores.append(scores.detach().cpu())
        return torch.concat(all_scores)

class DistributedScorer:
    def __init__(
        self, 
        num_gpus, 
        num_workers_per_gpu,
        model_path: str,
        flash_attn: bool,
        bf16: bool,
        load_in_4bit: bool = False,
        lora_rank: bool = False,
        lora_alpha: int = 16,
        lora_dropout: float = 0,
        packing_samples: bool = False,
    ):
        vllm_port = int(os.environ.get("VLLM_PORT", 0))

        num_cpus = os.environ.get("RAYLET_CPUS", None)
        num_cpus = int(num_cpus) if num_cpus is not None else None
        ray.init(num_cpus=num_cpus)

        num_workers_per_gpu = 1 if num_workers_per_gpu is None else num_workers_per_gpu
        num_workers = num_workers_per_gpu * num_gpus
        num_gpus_per_worker = 1 / num_workers_per_gpu

        scorers = []
        from ray.runtime_env import RuntimeEnv
        for i in range(num_workers):
            runtime_env = RuntimeEnv(env_vars={"VLLM_PORT": str(vllm_port + 2*i)})
            scorer = Scorer.options(num_gpus=num_gpus_per_worker, runtime_env=runtime_env).remote(
                model_path=model_path,
                flash_attn=flash_attn,
                bf16=bf16,
                load_in_4bit=load_in_4bit,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                packing_samples=packing_samples,
            )
            scorers.append(scorer)
        ray.wait([scorer.__ray_ready__.remote() for scorer in scorers])

        self.num_workers = len(scorers)
        self.scorers = scorers
        self.scorer_pool = ray.util.actor_pool.ActorPool(scorers)

    def shutdown(self):
        ray.shutdown()
        
    def scoring(
        self, 
        prompt_list,
        response_list,
        batch_size,
    ):
        try:
            piece = math.ceil(len(prompt_list) / self.num_workers)
            split_prompts = [
                prompt_list[i * piece: min((i+1) * piece, len(prompt_list))] 
                for i in range(self.num_workers)
            ]
            split_prompts = [piece for piece in split_prompts if piece != []]
            split_responses = [
                response_list[i * piece: min((i+1) * piece, len(response_list))] 
                for i in range(self.num_workers)
            ]
            split_responses = [piece for piece in split_responses if piece != []]

            split_tuples = list(zip(split_prompts, split_responses))
            results = list(
                self.scorer_pool.map(
                    lambda p, x: p.scoring.remote(
                        *x,
                        batch_size=batch_size,
                    ), 
                    split_tuples
                )
            )
            results = torch.concat(results)

        except ray.exceptions.RayActorError as ex:
            sys.exit(1)

        return results
