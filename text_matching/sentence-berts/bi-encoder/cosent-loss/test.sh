# change CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=2 python test.py \
    --model_name_or_path=output/hfl-chinese-macbert-large-2025-01-13_15-18-07 \
    --test_data_path=data/test.txt

