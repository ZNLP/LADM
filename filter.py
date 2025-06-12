import os
import pdb
import math
import glob
import json
import argparse
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from datasets import load_from_disk, DatasetDict
import matplotlib.pyplot as plt
from collections import defaultdict
from transformers import AutoTokenizer

from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_data_path", type=str, default="/public/zhangjiajun/huggingface/datasets/monology/pile-LlamaTokenizerFast-32k")
    parser.add_argument("--target_path", type=str, default="/public/zhangjiajun/jhchen/huggingface/datasets/monology/pile-LlamaTokenizerFast-32k-truncated/TinyLlama_v1.1/pile-LlamaTokenizerFast-32k_seed42/checkpoint-5000/stride512-chunk128-interval512")

    return parser.parse_args()


if __name__ == "__main__":
    # pdb.set_trace()
    args = parse_args()
    logger.info(args)
    target_file = f"{args.target_path}/part*-score.json"
    
    all_score_list = []
    for file_path in sorted(glob.glob(target_file)):
        with open(file_path, 'r') as file:
            for line in file:
                obj = json.loads(line)
                all_score_list.append(obj)

    sorted_by_score_list = sorted(all_score_list, key=lambda x: x['score'], reverse=True)
    for i, obj in enumerate(sorted_by_score_list):
        obj['rank'] = i
    print(sorted_by_score_list[0])

    with open(f"{args.target_path}/sorted_by_score.jsonl", "w") as f:
        for obj in sorted_by_score_list:
            f.write(json.dumps(obj) + "\n")


    # pdb.set_trace()
    original_dataset = load_from_disk(args.original_data_path)['train']

    num_b = 1
    max_tokens = 32768 * 32 * 1024 * num_b

    meta_list = [item['meta']['pile_set_name'] for item in sorted_by_score_list]
    meta_set = sorted(set(meta_list))
    print(meta_set)
    print(sorted_by_score_list[0])
    print(sorted_by_score_list[-1])

    meta_dict = {}
    for meta in meta_set:
        meta_sorted_data_list = [x for x in sorted_by_score_list if x['meta']['pile_set_name'] == meta]
        meta_dict[meta] = meta_sorted_data_list
    
    for percent in tqdm(range(1, 101)):
        total_tokens = 0
        all_meta_top = []
        for meta in meta_set:
            top = [x['index'] for x in meta_dict[meta][:int(len(meta_dict[meta]) * percent * 0.01)]]
            all_meta_top.extend(top)
            tokens = sum([len(original_dataset[index]['input_ids']) for index in top])
            total_tokens += tokens
        if total_tokens > max_tokens:
            break
    selected_dataset = original_dataset.select(indices=all_meta_top)
    selected_dataset = DatasetDict({'train': selected_dataset})
    print(selected_dataset)
    selected_dataset.save_to_disk(f"{args.target_path}/top-{num_b}B-field")