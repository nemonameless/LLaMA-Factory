#llamafactory-cli train examples/train_lora/qwen2vl_lora_sft.yaml

#llamafactory-cli train examples/train_lora/qwen2vl_lora_dpo.yaml

#llamafactory-cli train examples/train_full/qwen2vl_full_sft.yaml

GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=$((GPUS / GPUS_PER_NODE))

BATCH_SIZE=${BATCH_SIZE:-32}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1} ###
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS)) # 32/1/8=4

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export LAUNCHER=pytorch

DISTRIBUTED_ARGS="
    --nproc_per_node=$GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 
    --master_port=$MASTER_PORT
    "

LR=1e-8
#MODEL_PATH='Qwen/Qwen2-VL-2B-Instruct'
MODEL_PATH='Qwen2-VL-2B-Instruct'
OUTPUT_DIR='work_dirs/basline_2b_bs32_1e8_a100'
META_PATH="configs/baseline_6data_330k.json"


if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

torchrun $DISTRIBUTED_ARGS src/train.py \
    --model_name_or_path $MODEL_PATH \
    --do_train \
    --stage sft \
    --finetuning_type full \
    --freeze_vision_tower True \
    --deepspeed examples/deepspeed/ds_z2_config.json \
    --dataset mllm_demo \
    --dataset_dir . \
    --meta_path $META_PATH \
    --template qwen2_vl \
    --cutoff_len 8192 \
    --max_samples 1000000000 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --output_dir $OUTPUT_DIR \
    --logging_steps 1 \
    --save_steps 2000 \
    --save_total_limit 1 \
    --plot_loss \
    --overwrite_output_dir \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACC \
    --learning_rate $LR \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --bf16 \
    --ddp_timeout 9000 \
    --eval_steps -1 \
    2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
    #--resume_from_checkpoint "${OUTPUT_DIR}/checkpoint-5000" \
