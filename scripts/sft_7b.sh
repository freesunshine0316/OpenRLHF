
if [ -e /apdcephfs_qy3 ]; then
    WORKSPACE=/apdcephfs_qy3/share_301812049/oakyu/exp.tencent_chat
else
    WORKSPACE=/apdcephfs/share_300000800/user/oakyu/exp.tencent_chat
fi

DATASET=${WORKSPACE}/OpenRLHF/dataset/Qwen2.5-Math-7B_star_MATH_1k.jsonl
MODEL=${WORKSPACE}/models/Qwen2.5-MATH-7B
OUTPUT=${WORKSPACE}/models/Qwen2.5-Math-7B_star_MATH_1k
TENSORBOARD=${WORKSPACE}/OpenRLHF/tensorboard/sft/Qwen2.5-Math-7B_star_MATH_1k

deepspeed --module openrlhf.cli.train_sft \
   --max_len 2048 \
   --dataset ${DATASET} \
   --input_key question \
   --output_key answer \
   --input_template $'Question: {}\n\nAnswer: Let\'s think step by step\n' \
   --train_batch_size 128 \
   --micro_train_batch_size 1 \
   --max_samples 500000 \
   --pretrain ${MODEL} \
   --save_path ${OUTPUT} \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 2 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --use_tensorboard ${TENSORBOARD}

python /apdcephfs_qy3/share_301812049/oakyu/exp.tencent_chat/debug_anydev/occupy_heavy.py
python /jizhi/jizhi2/worker/trainer/occupy_heavy.py
