timestamp=$(date +%Y%m%d%H%M)
logfile="train_${timestamp}.log"

# change CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=3 nohup python train.py \
    --model_name_or_path=hfl/chinese-macbert-large \
    --output_dir=output \
    --train_data_path=data/train.txt \
    --eval_data_path=data/dev.txt \
    --num_train_epochs=3 \
    --save_total_limit=5 \
    --learning_rate=2e-5 \
    --weight_decay=0.01 \
    --warmup_ratio=0.01 \
    --bf16=True \
    --eval_strategy=epoch \
    --save_strategy=epoch \
    --per_device_train_batch_size=64 \
    --report_to="none" \
    --remove_unused_columns=False \
    --max_length=128 \
    > "$logfile" 2>&1 &

