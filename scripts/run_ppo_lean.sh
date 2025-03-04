#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

export TRITON_PTXAS_PATH=/usr/local/cuda-12.4/bin/ptxas                                                                      
export TRITON_CUOBJDUMP_PATH=/usr/local/cuda-12.4/bin/cuobjdump                                                              
export TRITON_NVDISASM_PATH=/usr/local/cuda-12.4/bin/nvdisasm  

# 启动 Lean 远程服务器
uvicorn --app-dir /app/qi/backup/data/RPROVER/OpenRLHF openrlhf.remote_rm.lean_rm_server:app --host 0.0.0.0 --port 1234 > lean_rm.log 2>&1 &

#等待服务器启动
sleep 5

# # 首先处理数据
# python scripts/prepare_lean_data.py

deepspeed --module openrlhf.cli.train_ppo \
  --pretrain /app/qi/backup/models/Goedel-Prover-SFT \
  --reward_pretrain /app/qi/backup/models/Goedel-Prover-SFT \
  --save_path ./checkpoint/goedal-rlhf \
  --save_steps -1 \
  --logging_steps 1 \
  --eval_steps -1 \
  --micro_train_batch_size 4 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 4 \
  --rollout_batch_size 1024 \
  --max_epochs 1 \
  --prompt_max_len 1024 \
  --generate_max_len 1024 \
  --bf16 \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 9e-6 \
  --init_kl_coef 0.01 \
  --prompt_data /app/qi/backup/data/RPROVER/lean_proofs_data \
  --input_key context_messages \
  --apply_chat_template \
  --max_samples 100000 \
  --normalize_reward \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --use_wandb f3b175fa54df63e7b0592b1bf157744eba49ef44 \
  --remote_rm_url http://localhost:1234/predict