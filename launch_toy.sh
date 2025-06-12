export CUDA_VISIBLE_DEVICES=7

dataset_root="/data_jhchen/huggingface/datasets"
model_root="/data_jhchen/huggingface/pretrained_model"

python score.py \
    --chunked_data_path "$dataset_root/UltraRonin/pile-LlamaTokenizerFast-32k-truncated-toy" \
    --model_path "$model_root/UltraRonin/Long-Attn-Calculator" \
    --model_tag "Long-Attn-Calculator" \
    --attn_chunk_size 128 \
    --d_afs 4 \
    --d_cds 4 \
    --num_shards 8 \
    --ablation 'No' \
    --part_idx 0 \
    --test