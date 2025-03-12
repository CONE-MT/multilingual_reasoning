import jsonlines
from datasets import load_dataset
import os 
import argparse, sys
import numpy as np
from matplotlib import pyplot as plt
import re


# args
parser=argparse.ArgumentParser()
parser.add_argument("--langs", help="languages", default="ar,bn,cs,de,en,es,fr,hu,ja,ko,ru,sr,sw,te,th,vi,zh")
parser.add_argument("--task", help="xgpqa or xmgsm")
parser.add_argument("--model", help="name of the evaluated model", default="qwen")
parser.add_argument("--log_type", default="eval")
args=parser.parse_args()

langs = args.langs.split(",")

task = args.task # xgpqa, xmgsm
if "gt-" not in task:
    task_prefix = 'samples_xgpqa_main_native_cot_zeroshot' if 'xgpqa' in task else 'samples_xmgsm_native_cot'
else:
    task_prefix = 'samples_xgpqa_main_google_native_cot_zeroshot' if 'xgpqa' in task else 'samples_xmgsm_native_cot_google'

if args.model == 'qwen':
    model = f'Qwen2.5-72B-Instruct'
elif args.model == 'llama':
    model = f'Llama-3.1-70B-Instruct'
elif args.model == 'r1-llama':
    model = f'DeepSeek-R1-Distill-Llama-70B'
else:
    raise NotImplementedError(f"Unknown model: {args.model}")
task_suffix = f'_{model}_trans' if "st-" in task else ""

log_dir = f'log/{task}/{model}'


# save scoring results, 1 as correct, 0 as incorrect
correct_dict = {lang: [] for lang in langs}


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



dict_difficulties = {}  # key: record ID, value: difficulty
dict_diff_counts = {}  # key: difficulty, value: question count in each difficulty
en_data_filename = "Idavidrein/gpqa"  # path to the original GPQA dataset on Huggingface
ds = load_dataset(en_data_filename, "gpqa_main")
for obj in ds["train"]:
    # breakpoint()
    rec_id = obj["Record ID"]
    
    diff_vote = {}
    difficulty1 = obj["Writer's Difficulty Estimate"]
    difficulty2 = obj["Question Difficulty_EV_1"]
    difficulty3 = obj["Question Difficulty_EV_2"]
    
    for difficulty in [difficulty1, difficulty2, difficulty3]:
        if not difficulty:
            difficulty = "NULL"
        if difficulty not in diff_vote:
            diff_vote[difficulty] = 1
        else:
            diff_vote[difficulty] += 1
    
    major_diff = max(diff_vote, key=diff_vote.get)
    assert rec_id not in dict_difficulties
    dict_difficulties[rec_id] = major_diff
    
    # add difficulty count
    diff_label = major_diff.split(' (')[0]
    if diff_label not in dict_diff_counts:
        dict_diff_counts[diff_label] = 1
    else:
        dict_diff_counts[diff_label] += 1
            

correct_difficulty_dict = {lang: [] for lang in langs}  # value list element: difficulty label
wrong_difficulty_dict = {lang: [] for lang in langs}
results_dict = {}

for lang in langs:
    if args.log_type =="exp":
        log_filename = f'{task_prefix}_{lang}_exp{args.exp}.jsonl'
    else:
        log_filename = f'{task_prefix}_{lang}.jsonl'

    with jsonlines.open(os.path.join(log_dir, log_filename)) as reader:
        for obj in reader:
            if obj["filter"] == "flexible-extract":
                question = obj['arguments']['gen_args_0']['arg_0']
                output = obj['resps'][0][0]
                reference = obj['target']
                rec_id = obj["doc"]["Record ID"]
                difficulty = dict_difficulties[rec_id]
                
                if not difficulty:
                    difficulty = "NULL"
                
                if judge(question, output, reference):
                    correct_difficulty_dict[lang].append(difficulty)
                else:
                    wrong_difficulty_dict[lang].append(difficulty)

    lang_correct_difficulty_counts = {}  # key: difficulty, value: count of correct answers in each difficulty, in this language
    lang_n_correct = 0  # summed count of correct answers in all difficulties, in this language
    for diff in correct_difficulty_dict[lang]:
        # breakpoint()
        diff_label = diff.split(' (')[0]
        if diff_label not in lang_correct_difficulty_counts:
            lang_correct_difficulty_counts[diff_label] = 1
        else:
            lang_correct_difficulty_counts[diff_label] += 1
        lang_n_correct += 1
    
    # merge easy undergrad and hard undergrad
    # lang_correct_difficulty_counts["Undergraduate level"] = lang_correct_difficulty_counts["Easy undergraduate level"] + lang_correct_difficulty_counts["Hard undergraduate level"]
    # dict_diff_counts["Undergraduate level"] = dict_diff_counts["Easy undergraduate level"] + dict_diff_counts["Hard undergraduate level"]
    
    print(f"Language: {lang}")
    for diff in ["Easy undergraduate level", "Hard undergraduate level", "Hard graduate level", "Post-graduate level or harder", "NULL"]:
        print(f"{diff}: {lang_correct_difficulty_counts[diff] / dict_diff_counts[diff]:.3%} ({lang_correct_difficulty_counts[diff]} out of {dict_diff_counts[diff]})")
    print("=" * 30)
    
#     # print(f"{lang} correct:")
#     for diff, count in lang_correct_difficulty_counts.items():
#         if diff not in results_dict:
#             # results_dict[diff] = [count / lang_n_correct]
#             results_dict[diff] = [count / dict_diff_counts[diff]]
#         else:
#             # results_dict[diff].append(count / lang_n_correct)
#             results_dict[diff].append(count / dict_diff_counts[diff])

# for diff, rates in results_dict.items():
#     print(f"{diff}: Max ({max(rates):.4f}); Min ({min(rates):.4f})")

# print(results_dict)