import jsonlines
import os 
import argparse, sys
import numpy as np
from matplotlib import pyplot as plt
import re
from langdetect import detect


# args
parser=argparse.ArgumentParser()
parser.add_argument("--task", help="xgpqa or xmgsm")
parser.add_argument("--cands", help="candidates", default="en1357,en2468,en1234,en2345")
parser.add_argument("--model", help="name of the evaluated model", default="qwen")
parser.add_argument("--exp", help="experiment number, e.g. 2_1")
parser.add_argument("--log_type", help="exp or judge", default="exp")
parser.add_argument("--lang_setting", help="en or multilingual", default="multilingual")
args=parser.parse_args()

cands = args.cands.split(",")
task = args.task # xgpqa, xmgsm
if "gt-" not in task:
    task_prefix = 'samples_xgpqa_main_native_cot_zeroshot' if 'xgpqa' in task else 'samples_xmgsm_native_cot'
else:
    task_prefix = 'samples_xgpqa_main_google_native_cot_zeroshot' if 'xgpqa' in task else 'samples_xmgsm_native_cot_google'

if args.model == 'qwen':
    model_size = '72B'
    model = f'Qwen2.5-{model_size}-Instruct'
elif args.model == 'llama':
    model_size = '70B'
    model = f'Llama-3.1-{model_size}-Instruct'
elif args.model == 'r1-llama':
    model_size = '70B'
    model = f'DeepSeek-R1-Distill-Llama-{model_size}'
else:
    raise NotImplementedError(f"Unknown model: {args.model}")
task_suffix = f'_{model}_trans' if "st-" in task else ""

if args.log_type == "exp":
    log_dir = f'log/{task}/exp{args.exp}/{model}'
else:
    log_dir = f'judge-log/{task}/{args.lang_setting}/{model}'


chosen_dict = {}
n_samples = 0
for cand in cands:
    if args.log_type =="exp":
        log_filename = f'{task_prefix}_{cand}_exp{args.exp}.jsonl'
    else:
        log_filename = f'{task_prefix}_{cand}.jsonl'

    with jsonlines.open(os.path.join(log_dir, log_filename)) as reader:
        for obj in reader:
            if obj["filter"] == "flexible-extract":
                model_output = (obj['resps'][0][0])
                n_samples += 1
                lang_used = detect(model_output)
                if lang_used not in chosen_dict:
                    chosen_dict[lang_used] = 1
                else:
                    chosen_dict[lang_used] += 1

for k, v in chosen_dict.items():
    print(f"{k}: {v / n_samples:.4f}")


