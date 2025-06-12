import os
import pdb
import torch
import json
import argparse
import copy
import numpy as np
from tqdm import tqdm
from itertools import chain
from multiprocessing import Pool
from collections import defaultdict
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from loguru import logger
from modeling_llama2 import LlamaForCausalLM, LlamaModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunked_data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_tag", type=str, required=True)
    parser.add_argument("--attn_chunk_size", type=int, default=128)
    parser.add_argument("--d_afs", type=int, default=4)
    parser.add_argument("--d_cds", type=int, default=4)
    parser.add_argument("--num_shards", type=int, default=8)
    parser.add_argument("--part_idx", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--ablation", type=str, default='No', choices=['No', 'no_std', 'no_len_weight'])

    return parser.parse_args()


def main(args):
    chunked_datasets = load_from_disk(args.chunked_data_path)['train']
    outputs_dir = f"{args.chunked_data_path}/{args.model_tag}/chunk{args.attn_chunk_size}-d_afs{args.d_afs}-d_cds{args.d_cds}"
    args.interval = args.attn_chunk_size * args.d_afs
    args.stride = args.attn_chunk_size * args.d_cds
    if args.ablation != 'No':
        outputs_dir = outputs_dir + f"-{args.ablation}"
    if args.test:
        outputs_dir = outputs_dir + "-test"
    os.makedirs(outputs_dir, exist_ok=True)
    print(chunked_datasets)
    
    if args.test:
        part_chunked_datasets = chunked_datasets.select(range(32))
    else:
        part_chunked_datasets = chunked_datasets.shard(num_shards=args.num_shards, index=args.part_idx)
    print(part_chunked_datasets)

    if args.test:
        # pdb.set_trace()
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            attn_implementation="flash_attention_2"
        ).eval()
        
        with torch.no_grad():
            input_ids = torch.tensor([part_chunked_datasets[0]['input_ids']]).to(model.device)
            outputs = model(input_ids, labels=input_ids)
            print(outputs.loss)

    
    config = AutoConfig.from_pretrained(args.model_path)
    config.stride = args.stride
    config.attn_chunk_size = args.attn_chunk_size
    config.interval = args.interval
    config.ablation = args.ablation
    config._attn_implementation = "flash_attention_2"
    print(config)
    batch_size = 16
    model = LlamaModel.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="flash_attention_2"
    ).eval()
    print(model)


    with torch.no_grad():
        for start_idx in tqdm(range(0, len(part_chunked_datasets), batch_size)):
            data_batch = part_chunked_datasets[start_idx: min(start_idx + batch_size, len(part_chunked_datasets))]
            index = data_batch['index']
            metas = data_batch['meta']
            chunked_index = data_batch.get('chunked_index', [None] * len(index))
            input_ids = torch.tensor(data_batch['input_ids']).to(model.device)

            all_attn = model(input_ids, return_dict=False, use_cache=False)[-1]
            
            all_attn = torch.stack(all_attn, dim=0) # layers, bs
            avg_attn = all_attn.mean(dim=0).cpu().numpy().tolist() # bs
            
            
            with open(f"{outputs_dir}/part{args.part_idx}-score.json", "a") as f:
                for idx, meta, chunked_idx, attn in zip(index, metas, chunked_index, avg_attn):
                    f.write(json.dumps({"index": idx, "chunked_index": chunked_idx, "score": attn, "meta": meta}) + "\n")



if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    main(args)