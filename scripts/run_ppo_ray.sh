
if [ -e /apdcephfs_qy3 ]; then
    WORKSPACE=/apdcephfs_qy3/share_301812049/oakyu/exp.tencent_chat
else
    WORKSPACE=/apdcephfs/share_300000800/user/oakyu/exp.tencent_chat
fi

POLICY=${WORKSPACE}/models/Qwen2.5-Math-7B_star_MATH_1k

NAME=naive_bf16_t0.6_tp0.95_grpo
OUTPUT=${WORKSPACE}/OpenRLHF/output/Qwen2.5-Math-7B_MATH_ppo/${NAME}
TENSORBOARD=${WORKSPACE}/OpenRLHF/tensorboard/Qwen2.5-Math-7B_MATH_ppo/${NAME}

set -x 

MEGATRON_REPO=${WORKSPACE}/OpenRLHF/
export PYTHONPATH=${MEGATRON_REPO}:$PYTHONPATH


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "./ray_workdir"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --colocate_actor_ref \
   --reward_num_nodes 0 \
   --reward_num_gpus_per_node 0 \
   --critic_num_nodes 0 \
   --critic_num_gpus_per_node 0 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 1 \
   --pretrain ${POLICY} \
   --remote_rm_url http://localhost:1234/predict \
   --save_path ${OUTPUT} \
   --micro_train_batch_size 2 \
   --train_batch_size 256 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 256 \
   --temperature 0.6 \
   --top_p 0.95 \
   --max_epochs 1 \
   --num_episodes 4 \
   --prompt_max_len 512 \
   --generate_max_len 1024 \
   --zero_stage 3 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 1e-6 \
   --init_kl_coef 0.01 \
   --prompt_data ${WORKSPACE}/OpenRLHF/dataset/train_math_level3to5.jsonl \
   --input_key question \
   --input_template $'Question: {}\n\nAnswer: Let\'s think step by step\n' \
   --max_samples 100000 \
   --bf16 \
   --grad_accum_dtype fp32 \
   --adam_offload \
   --gradient_checkpointing \
   --advantage_estimator grpo \
   --search_algo sampling \
   --n_samples_per_prompt 1 \
   --save_steps 25  \
   --use_tensorboard ${TENSORBOARD} \
   2>&1 | tee -a ${WORKSPACE}/OpenRLHF/logs/train/Qwen2.5-Math-7B_MATH_ppo/${NAME}.log



python3 /apdcephfs_qy3/share_301812049/oakyu/exp.tencent_chat/debug_anydev/occupy_heavy.py
python3 /jizhi/jizhi2/worker/trainer/occupy_heavy.py
