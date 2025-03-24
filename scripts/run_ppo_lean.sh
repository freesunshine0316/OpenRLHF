# filepath: [run_ppo_lean.sh](http://_vscodecontentref_/0)
#!/bin/bash
set -x
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

# for CUDA ENV
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH

export TRITON_PTXAS_PATH=/usr/local/cuda-12.2/bin/ptxas                                                                      
export TRITON_CUOBJDUMP_PATH=/usr/local/cuda-12.2/bin/cuobjdump                                                              
export TRITON_NVDISASM_PATH=/usr/local/cuda-12.2/bin/nvdisasm  

# More secure process cleanup
echo "===== Cleaning up related processes ====="

# First check and record the processes to be cleaned
echo "Checking for processes on port 1239..."
PORT_PIDS=$(lsof -t -i:1239 2>/dev/null)
if [ -n "$PORT_PIDS" ]; then
    echo "Will terminate the following port 1239 processes: $PORT_PIDS"
fi

echo "Checking for processes on port 29500..."
PORT_PIDS=$(lsof -t -i:29500 2>/dev/null)
if [ -n "$PORT_PIDS" ]; then
    echo "29500: $PORT_PIDS"
fi

echo "Checking lean related processes..."
# Use more precise pattern matching to avoid accidental killing
LEAN_PIDS=$(ps aux | grep -E '[l]ean4|[l]ake.*lean' | awk '{print $2}')
if [ -n "$LEAN_PIDS" ]; then
    echo "Will terminate the following lean processes: $LEAN_PIDS"
fi

echo "Checking uvicorn processes..."
UVICORN_PIDS=$(ps aux | grep -E '[u]vicorn.*lean_rm_server' | awk '{print $2}')
if [ -n "$UVICORN_PIDS" ]; then
    echo "Will terminate the following uvicorn processes: $UVICORN_PIDS"
fi

# Then terminate the processes one by one instead of using a broad pkill
if [ -n "$PORT_PIDS" ]; then
    echo "Terminating port 1239 processes..."
    for pid in $PORT_PIDS; do
        kill $pid 2>/dev/null || true
    done
fi

if [ -n "$LEAN_PIDS" ]; then
    echo "Terminating lean processes..."
    for pid in $LEAN_PIDS; do
        kill $pid 2>/dev/null || true
    done
fi

if [ -n "$UVICORN_PIDS" ]; then
    echo "Terminating uvicorn processes..."
    for pid in $UVICORN_PIDS; do
        kill $pid 2>/dev/null || true
    done
fi

# Wait for the processes to terminate completely
echo "Waiting for processes to terminate..."
sleep 2

# Check if there are any remaining processes, use SIGKILL only when necessary
REMAINING_PIDS=$(ps aux | grep -E '[l]ean4|[l]ake.*lean|[u]vicorn.*lean_rm_server' | awk '{print $2}')
if [ -n "$REMAINING_PIDS" ]; then
    echo "Some processes are still running, attempting to force termination..."
    for pid in $REMAINING_PIDS; do
        echo "Forcefully terminating PID $pid"
        kill -9 $pid 2>/dev/null || true
    done
    sleep 1
fi

echo "Starting lean verification server..."
# Use nohup to ensure the process continues to run in the background
nohup gunicorn -w 16 -k uvicorn.workers.UvicornWorker "openrlhf.remote_rm.lean_rm_server:app" -b 0.0.0.0:1239 > lean_rm.log 2>&1  &  

# Wait for the server to start completely
echo "Waiting for server to start..."
sleep 5

# Check if the server started successfully
MAX_RETRIES=5
RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:1239/status > /dev/null; then
        echo "Server started successfully"
        break
    else
        RETRY=$((RETRY+1))
        if [ $RETRY -lt $MAX_RETRIES ]; then
            echo "Attempt $RETRY/$MAX_RETRIES: Server not yet started, waiting longer..."
            sleep 5
        else
            echo "Error: Server failed to start successfully, please check the lean_rm.log file"
            cat lean_rm.log
            exit 1
        fi
    fi
done

echo "===== Starting PPO training ====="

deepspeed --module openrlhf.cli.train_ppo \
  --pretrain /app/qi/backup/models/Goedel-Prover-SFT \
  --critic_pretrain /app/qi/backup/models/Goedel-Prover-SFT \
  --save_path ./checkpoint/goedal-rlhf-v3  \
  --save_steps  100 \
  --logging_steps 1 \
  --eval_steps 20 \
  --micro_train_batch_size 1 \
  --train_batch_size 64 \
  --micro_rollout_batch_size 1 \
  --rollout_batch_size 64 \
  --max_epochs 1 \
  --prompt_max_len 1024 \
  --generate_max_len 2048 \
  --bf16 \
  --actor_learning_rate 5e-7 \
  --critic_learning_rate 9e-6 \
  --init_kl_coef 0.01 \
  --prompt_data /app/qi/backup/data/RPROVER/lean_proofs_data \
  --input_key context_messages \
  --apply_chat_template \
  --max_samples 5000 \
  --normalize_reward \
  --gradient_checkpointing \
  --zero_stage 2 \
  --use_wandb f3b175fa54df63e7b0592b1bf157744eba49ef44 \
  --remote_rm_url http://localhost:1239/predict_lean \
  --flash_attn  \
  --adam_offload \



   