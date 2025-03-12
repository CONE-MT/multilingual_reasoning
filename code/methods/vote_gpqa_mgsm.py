import jsonlines
import os 
import argparse, sys
import numpy as np
from matplotlib import pyplot as plt
import re


# args
parser=argparse.ArgumentParser()
parser.add_argument("--task", help="xgpqa, st-xgpqa, xmgsm or st-mgsm")
parser.add_argument("--langs", help="languages", default="en1357,en2468,en1234,en2345")
parser.add_argument("--model", help="name of the evaluated model", default="qwen")
parser.add_argument("--log", help="whether to save a judge_log.txt", default="False")
parser.add_argument("--exp", help="experiment number")
parser.add_argument("--log_type", help="exp or judge", default="exp")
parser.add_argument("--lang_setting", help="en or multilingual", default="multilingual")
args=parser.parse_args()

langs = args.langs.split(",")
use_log = args.log == "True"
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
    

# save scoring results, 1 as correct, 0 as incorrect
correct_dict = {lang: [] for lang in langs}


# rule-based judging
use_cot = True
if args.log_type == "exp":
    f_judge_log = open(os.path.join(log_dir, 'judge_log.txt'), 'w')
else:
    f_judge_log = open(os.path.join(log_dir, f'judge_log_{args.exp}.txt'), 'w')

def extract(output):
    answer_prefix = "Final Answer" if args.exp == 'exp1' else "Answer"
    if use_cot:
        if f'{answer_prefix}: ' in output:
            output_tail = output.split(f'{answer_prefix}: ')[-1]
        else:
            output_tail = output[-32:]
    else:
        output_tail = output
    return output_tail


def to_choices(lang_raw_outputs):
    # given a dict of raw model outputs in different languages, output ABCD choices for each language
    # lang_raw_outputs: {lang1: output1, lang2: output2, ... }
    valid_options = ['A', 'B', 'C', 'D', 'N']  # 'N' for None
    lang_processed_outputs = {}  # same size as outputs
    
    for lang, ro in lang_raw_outputs.items():
        output_tail = extract(ro)
        option_pattern = r'[\(\{]([ABCD])[\)\}]'
        
        output_option_list = re.findall(option_pattern, output_tail)
        if output_option_list:
            output_option = output_option_list[-1]
        else:
            sub_pattern = r'\b([ABCD])\b'
            sub_option_list = re.findall(sub_pattern, output_tail)
            if sub_option_list:
                output_option = sub_option_list[-1]
            else:
                output_option = 'N'
                
        assert output_option in valid_options, f"Invalid option: {output_option}"
        lang_processed_outputs[lang] = output_option
        
        # process_result = {x: 0 for x in valid_options}
        # for option in valid_options:
        #     if option in output_tail:
        #         process_result[option] += 1
        # lang_processed_outputs[lang] = max(process_result, key=process_result.get)
    
    return lang_processed_outputs  # a dict of ABCD choices for each question in that language


def vote_xgpqa(docid, lang_processed_outputs):
    # lang_processed_outputs: a dict of processed outputs to a same question from different languages
    # lang_processed_outputs: {lang1: 'A', lang2: 'C', ... }
    
    actual_options = ['CA', 'IA1', 'IA2', 'IA3', 'NONE']
    actual_map = actual_choice_dict[docid]  # {lang1: {'A': 'CA', 'B': 'IA2', 'C': 'IA3', 'D': 'IA1'}, lang2: { ... }}
    
    # vote the majority option
    vote_result = {x: 0 for x in actual_options}
    for lang, po in lang_processed_outputs.items():
        assert po in actual_map[lang], f"{docid}, {lang}, {actual_map[lang]}, {po}"
        vote_result[actual_map[lang][po]] += 1
    
    voted_option = max(vote_result, key=vote_result.get)
    maximum_consistency = vote_result[voted_option] / len(lang_processed_outputs)
    return voted_option, maximum_consistency  # a choice among CA and IA1-3


def vote_xmgsm(docid, lang_raw_outputs):
    # lang_raw_outputs: a dict of outputs to a same question from different languages
    # lang_raw_outputs: {lang1: 'answer: 122', lang2: '121', ... }
    number_pattern = r'\d+(?:[.,]\d{3})*(?:[.,]\d+)?'
    
    # vote the majority answer
    vote_result = {}
    for lang, ro in lang_raw_outputs.items():
        output_tail = extract(ro)
        output_number_list = re.findall(number_pattern, output_tail)
        output_number = output_number_list[-1].replace(',', '').replace('.', '') if output_number_list else 'None'
        
        if output_number not in vote_result:
            vote_result[output_number] = 1
        else:
            vote_result[output_number] += 1
    
    voted_answer = max(vote_result, key=vote_result.get)
    maximum_consistency = vote_result[voted_answer] / len(lang_raw_outputs)
    return voted_answer, maximum_consistency  # a numeric choice


def judge(docid, lang_raw_outputs, answer=None):
    # lang_raw_outputs: a dict of raw outputs to a same question from different languages
    # lang_raw_outputs: {lang1: output1, lang2: output2, ... }
    # answer: reference answer
    
    if "xgpqa" in task:
        lang_processed_outputs = to_choices(lang_raw_outputs)
        voted_output, consist = vote_xgpqa(docid, lang_processed_outputs)
        eval_result = voted_output == 'CA'
    elif "xmgsm" in task:
        voted_output, consist = vote_xmgsm(docid, lang_raw_outputs)
        eval_result = voted_output == answer
                
    if use_log:
        print(f"DocID: {docid}\nOutputs: {lang_raw_outputs}\nVoted Output: {voted_output}\n\nEval: {eval_result}\n\n", file=f_judge_log)
    # breakpoint()
    return eval_result, consist


def get_actual_map(actuals, surficials):
    # e.g. {'A': 'CA', 'B': 'IA2', 'C': 'IA3', 'D': 'IA1'}
    surficial_options = ['A', 'B', 'C', 'D', 'N']
    actual_options = ['CA', 'IA1', 'IA2', 'IA3', 'NONE']
    actual_map = {'N': 'NONE'}
    for s_text, s_opt in zip(surficials, surficial_options):
        for a_text, a_opt in zip(actuals, actual_options):
            if s_text.strip().lower() == a_text.strip().lower():
                actual_map[s_opt] = a_opt
                break
    assert len(actual_map) == 5
    return actual_map
    

all_langs_outputs = {}  # {docid: {lang1: output1, lang2: output2, ... }}
actual_choice_dict = {}  # for xgpqa, {docid: {lang1: {'A': 'CA', 'B': 'IA2', 'C': 'IA3', 'D': 'IA1'}, lang2: { ... }}, ... }
all_docs_answers = {}  # for xmgsm, {docid: answer}
all_docs_eval = []  # a list of 0 and 1
all_docs_consistency = []  # a list of consistency values between 0-1

for lang in langs:
    if args.log_type =="exp":
        log_filename = f'{task_prefix}_{lang}_exp{args.exp}.jsonl'
    else:
        log_filename = f'{task_prefix}_{lang}.jsonl'
    with jsonlines.open(os.path.join(log_dir, log_filename)) as reader:
        for obj in reader:
            if obj["filter"] == "flexible-extract":
                docid = obj['doc_id']
                
                if "xgpqa" in task:
                    obj_ca = obj['doc']['Correct Answer']
                    obj_ia1 = obj['doc']['Incorrect Answer 1']
                    obj_ia2 = obj['doc']['Incorrect Answer 2']
                    obj_ia3 = obj['doc']['Incorrect Answer 3']
                    obj_actuals = [obj_ca, obj_ia1, obj_ia2, obj_ia3]
                    
                    obj_a = obj['doc']['choice1']
                    obj_b = obj['doc']['choice2']
                    obj_c = obj['doc']['choice3']
                    obj_d = obj['doc']['choice4']
                    obj_suficials = [obj_a, obj_b, obj_c, obj_d]
                    
                    try:
                        obj_actual_map = get_actual_map(obj_actuals, obj_suficials)  # e.g. {'A': 'CA', 'B': 'IA2', 'C': 'IA3', 'D': 'IA1'}
                    except AssertionError:
                        continue

                    if docid not in actual_choice_dict:
                        actual_choice_dict[docid] = {}
                    actual_choice_dict[docid][lang] = obj_actual_map
                
                elif "xmgsm" in task:
                    doc_answer = obj['target']
                    all_docs_answers[docid] = doc_answer
                
                else:
                    raise NotImplementedError(f"Unknown task: {task}")
                    
                obj_output = obj['resps'][0][0]
                if docid not in all_langs_outputs:
                    all_langs_outputs[docid] = {}
                all_langs_outputs[docid][lang] = obj_output
                
for docid, raw_output in all_langs_outputs.items():
    if 'xgpqa' in task:
        doc_eval, doc_consist = judge(docid, raw_output)
    elif 'xmgsm' in task:
        doc_eval, doc_consist = judge(docid, raw_output, all_docs_answers[docid])
    else:
        raise NotImplementedError(f"Unknown task: {task}")
    all_docs_eval.append(doc_eval)
    all_docs_consistency.append(doc_consist)

if use_log:
    f_judge_log.close()

# print majority voting result
majority_acc = sum(all_docs_eval) / len(all_docs_eval)
majority_consistency = sum(all_docs_consistency) / len(all_docs_consistency)
# print(len(all_docs_eval))
# print(f'Majority voting accuracy: {majority_acc:.5f}')
# print(f'Majority consistency: {majority_consistency:.5f}')

print(f'{majority_acc:.5f};{majority_consistency:.5f}')