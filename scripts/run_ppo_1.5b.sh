POLICY=/apdcephfs_qy3/share_301812049/oakyu/exp.tencent_chat/models/Qwen2.5-Math-1.5B_sft_star
CRITIC=/apdcephfs_qy3/share_301812049/oakyu/exp.tencent_chat/models/Qwen2.5-Math-1.5B_sft_value

NAME=naive_fp32
OUTPUT=/apdcephfs_qy3/share_301812049/oakyu/exp.tencent_chat/OpenRLHF/output/Qwen2.5-Math-1.5B_gsm8k_math_ppo/${NAME}
TENSORBOARD=/apdcephfs_qy3/share_301812049/oakyu/exp.tencent_chat/OpenRLHF/tensorboard/Qwen2.5-Math-1.5B_gsm8k_math_ppo/${NAME}

deepspeed --module openrlhf.cli.train_ppo \
  --pretrain ${POLICY} \
  --critic_pretrain ${CRITIC} \
  --remote_rm_url http://localhost:1234/predict \
  --save_path ${OUTPUT} \
  --save_steps -1 \
  --logging_steps 1 \
  --eval_steps -1 \
  --micro_train_batch_size 2 \
  --train_batch_size 256 \
  --micro_rollout_batch_size 2 \
  --rollout_batch_size 256 \
  --max_epochs 1 \
  --num_episodes 4 \
  --prompt_max_len 512 \
  --generate_max_len 1024 \
  --zero_stage 2 \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 1e-6 \
  --init_kl_coef 0.01 \
  --prompt_data dataset/train_gsm8k_math.jsonl \
  --input_key input \
  --max_samples 100000 \
  --adam_offload \
  --gradient_checkpointing \
  --search_algo sampling \
  --n_samples_per_prompt 1 \
  --use_tensorboard ${TENSORBOARD}

python3 occupy_heavy.py

# --search_algo bestofn

# --bf16 \
# --grad_accum_dtype fp32 \
