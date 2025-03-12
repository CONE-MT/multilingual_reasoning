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
parser.add_argument("--langs", help="languages to choose")
parser.add_argument("--model", help="name of the evaluated model", default="qwen")
parser.add_argument("--lang_setting", help="en or multilingual", default="multilingual")
args=parser.parse_args()

langs = args.langs.split(",")
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

log_dir = f'judge-log/{task}/{args.lang_setting}/{model}'

# rule-based judging
use_cot = True

def extract(output):
    answer_prefix = "Answer"
    if use_cot:
        if f'{answer_prefix}: ' in output:
            output_tail = output.split(f'{answer_prefix}: ')[-1]
        else:
            output_tail = output[-32:]
    else:
        output_tail = output
    return output_tail


def judge(question, output, answer, multiple=False):
    eval_result = None
    output_tail = extract(output)
    
    option_pattern = r'[\(\{]([ABCD])[\)\}]'
    valid_options = ['A', 'B', 'C', 'D', 'N']
    
    number_pattern = r'\d+(?:[.,]\d{3})*(?:[.,]\d+)?'
    
    if "xgpqa" in task:
        output_option_list = re.findall(option_pattern, output_tail)
        if output_option_list:
            output_norm = output_option_list[-1]
        else:
            sub_pattern = r'\b([ABCD])\b'
            sub_option_list = re.findall(sub_pattern, output_tail)
            if sub_option_list:
                output_norm = sub_option_list[-1]
            else:
                output_norm = 'N'
        assert output_norm in valid_options, f"Invalid option: {output_norm}\nTail: {output_tail}"
        a = answer.strip('(').strip(')')
    elif "xmgsm" in task:
        output_number_list = re.findall(number_pattern, output_tail)
        output_norm = output_number_list[-1].replace(',', '').replace('.', '') if output_number_list else 'None'
        a = answer
        
    # assert len(a) == 1
    eval_result = a == output_norm
                
    # breakpoint()
    return eval_result


chosen_filename = f"{args.model}_{task}_chosen_lang.npy"
model_chosen_langs = np.load(f"self_judge/{chosen_filename}", allow_pickle=True)
dict_model_chosen_per_lang = model_chosen_langs.item()  # de: [1, 0, 1, ...]

correct_dict = {lang: [] for lang in langs}  # de: [0, 1, 1, ...]

for lang in langs:
    log_filename = f'{task_prefix}_{lang}.jsonl'

    with jsonlines.open(os.path.join(log_dir, log_filename)) as reader:
        for obj in reader:
            if obj["filter"] == "flexible-extract":
                question = obj['arguments']['gen_args_0']['arg_0']
                output = obj['resps'][0][0]
                reference = obj['target']
                
                if judge(question, output, reference):
                    correct_dict[lang].append(1)
                else:
                    correct_dict[lang].append(0)
    
    # breakpoint()
    right_chosen_counts = []
    wrong_chosen_counts = []
    for chosen, correct in zip(dict_model_chosen_per_lang[lang], correct_dict[lang]):
        if correct == 1:
            right_chosen_counts.append(chosen)
        else:
            wrong_chosen_counts.append(chosen)

    print(f"{lang}:\nright: {np.mean(right_chosen_counts)*100:.2f}, wrong: {np.mean(wrong_chosen_counts)*100:.2f}")
        
        
