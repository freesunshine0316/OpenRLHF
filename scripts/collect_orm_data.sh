
if [ -e /apdcephfs_qy3 ]; then
    WORKSPACE=/apdcephfs_qy3/share_301812049/oakyu/exp.tencent_chat
else
    WORKSPACE=/apdcephfs/share_300000800/user/oakyu/exp.tencent_chat
fi

MODEL=${WORKSPACE}/models/Qwen2.5-Math-7B_star_MATH_3k
DATASET=${WORKSPACE}/OpenRLHF/dataset/train_math_level3to5.jsonl
OUTPUT=${WORKSPACE}/OpenRLHF/dataset/critic/Qwen2.5-Math-7B_sft_star_MATH_3k_n20.jsonl
K=20

export VLLM_PORT=43543
python -m openrlhf.cli.collect_orm_data \
    --num_gpus 4 \
    --num_workers_per_gpu 1 \
    --model_path ${MODEL} \
    --data_path ${DATASET} \
    --input_key question \
    --answer_key gt_answer \
    --input_template $'Question: {}\n\nAnswer: Let\'s think step by step\n' \
    --num_samples ${K} \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_new_tokens 1024 \
    --output_path ${OUTPUT}


python3 /apdcephfs_qy3/share_301812049/oakyu/exp.tencent_chat/debug_anydev/occupy_heavy.py
python3 /jizhi/jizhi2/worker/trainer/occupy_heavy.py
