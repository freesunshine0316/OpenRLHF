
if [ -e /apdcephfs_qy3 ]; then
    WORKSPACE=/apdcephfs_qy3/share_301812049/oakyu/exp.tencent_chat
else
    WORKSPACE=/apdcephfs/share_300000800/user/oakyu/exp.tencent_chat
fi

DATASET=${WORKSPACE}/OpenRLHF/dataset/critic/Qwen2.5-Math-7B_sft_star_MATH_3k_n20.jsonl
MODEL=${WORKSPACE}/models/Qwen2.5-Math-7B_star_MATH_3k
N=20

NAME=Qwen2.5-Math-7B_star_MATH_3k_orm_n${N}_lr5e-6
OUTPUT=${WORKSPACE}/models/${NAME}
TENSORBOARD=${WORKSPACE}/OpenRLHF/tensorboard/critic/${NAME}

deepspeed --module openrlhf.cli.train_orm \
   --max_len 2048 \
   --dataset ${DATASET} \
   --input_key question \
   --output_key answer \
   --label_key label \
   --input_template $'Question: {}\n\nAnswer: Let\'s think step by step\n' \
   --max_n_samples ${N} \
   --train_batch_size 128 \
   --micro_train_batch_size 1 \
   --max_samples 500000 \
   --pretrain ${MODEL} \
   --save_path ${OUTPUT} \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --adam_offload \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing



python /apdcephfs_qy3/share_301812049/oakyu/exp.tencent_chat/debug_anydev/occupy_heavy.py
python /jizhi/jizhi2/worker/trainer/occupy_heavy.py
