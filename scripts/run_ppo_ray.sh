POLICY=/apdcephfs_sh2/share_300000800/user/antewang/Qwen2.5-Math-7B
# POLICY=/apdcephfs_sh2/share_300000800/user/oakyu/Qwen2.5-7B
OUTPUT=/apdcephfs/share_300000800/user/oakyu/exp.tencent_chat/OpenRLHF/output/r1_zero_qwen25_math_7b

set -x 

MEGATRON_REPO=/apdcephfs/share_300000800/user/oakyu/exp.tencent_chat/OpenRLHF/
export PYTHONPATH=${MEGATRON_REPO}:$PYTHONPATH

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "./ray_workdir"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ckpt_path ${OUTPUT}  \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --colocate_actor_ref \
   --reward_num_nodes 0 \
   --reward_num_gpus_per_node 0 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 2 \
   --vllm_num_engines 1 \
   --vllm_tensor_parallel_size 2 \
   --pretrain ${POLICY} \
   --remote_rm_url http://localhost:1234/predict_qwen_math \
   --save_path ${OUTPUT} \
   --micro_train_batch_size 1 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size 128 \
   --temperature 0.6 \
   --n_samples_per_prompt 8 \
   --max_samples 100000 \
   --max_epochs 1 \
   --num_episodes 500 \
   --max_len 2048 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data /apdcephfs/share_300000800/user/lfsong/exp.tencent_chat/math/train_gsm8k_math_add_r1_zero.jsonl \
   --apply_chat_template \
   --input_key question \
   --flash_attn \
   --adam_offload \
   --gradient_checkpointing \
   --gradient_checkpointing_use_reentrant \
   --save_steps 25 

   # --search_algo sampling \
   # --load_checkpoint \
   # https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/OpenRLHF/develop-log.md

   # --input_template "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {}. Assistant:" \

python3 /jizhi/jizhi2/worker/trainer/occupy_heavy.py
