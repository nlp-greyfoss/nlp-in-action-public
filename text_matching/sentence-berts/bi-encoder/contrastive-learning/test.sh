# change CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0 python test.py \
    --model_name_or_path=output/hfl-chinese-macbert-large \
    --test_data_path=data/test.txt
   

