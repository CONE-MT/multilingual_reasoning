import jsonlines
import os 
import argparse, sys
import numpy as np
from matplotlib import pyplot as plt
import re


# args
parser=argparse.ArgumentParser()
parser.add_argument("--task", help="xgpqa or xmgsm")
parser.add_argument("--langs", help="languages", default="en1357,en2468,en1234,en2345")
parser.add_argument("--model", help="name of the evaluated model", default="qwen")
parser.add_argument("--exp", help="experiment number")
parser.add_argument("--log_type", help="exp or judge", default="exp")
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

if args.log_type == "exp":
    log_dir = f'log/{task}/exp{args.exp}/{model}'
else:
    log_dir = f'judge-log/{task}/{args.lang_setting}/{model}'


# rule-based judging
use_cot = True
if args.log_type == "exp":
    f_judge_log = open(os.path.join(log_dir, 'judge_log.txt'), 'w')
else:
    f_judge_log = open(os.path.join(log_dir, f'judge_log_{args.exp}.txt'), 'w')


def extract(output):
    answer_prefix = "Final Answer" if args.exp == '1' else "Answer"
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
                
    print(f"Q: {question}\nA: {output}\n{'=' * 30}\nNormalized: {output_norm}\nRef: {answer}\nEval: {eval_result}\n\n", file=f_judge_log)
    # breakpoint()
    return eval_result


correct_dict = {lang: [] for lang in langs}
for lang in langs:
    if args.log_type =="exp":
        log_filename = f'{task_prefix}_{lang}_exp{args.exp}.jsonl'
    else:
        log_filename = f'{task_prefix}_{lang}.jsonl'
    questions = []
    outputs = []
    references = []
    with jsonlines.open(os.path.join(log_dir, log_filename)) as reader:
        for obj in reader:
            if obj["filter"] == "flexible-extract":
                questions.append(obj['arguments']['gen_args_0']['arg_0'])
                outputs.append(obj['resps'][0][0])
                references.append(obj['target'])

    correct_list = []
    for q, o, r in zip(questions, outputs, references):
        correct_int = 1 if judge(q, o, r) else 0
        correct_list.append(correct_int)

    correct_dict[lang] = correct_list
    acc = sum(correct_list) / len(correct_list)
    print(f'{lang} Accuracy: {acc:.5f}')

f_judge_log.close()        

# passk
all_lang_correct_ints = np.array([correct_dict[lang] for lang in langs])  # n_lang, n_sample
any_correct_ints = (np.sum(all_lang_correct_ints, axis=0) > 0).astype(int)
acc_all = sum(any_correct_ints) / len(any_correct_ints)
# acc_dict['all'] = acc_all
print(f'Pass@k: {acc_all:.3%}')
