import jsonlines
import os 
import argparse, sys
import numpy as np
from matplotlib import pyplot as plt
import re


# args
parser=argparse.ArgumentParser()
parser.add_argument("--langs", help="languages", default="ar,bn,cs,de,en,es,fr,hu,ja,ko,ru,sr,sw,te,th,vi,zh")
# multi: "ar,bn,cs,de,en,es,fr,hu,ja,ko,ru,sr,sw,te,th,vi,zh"
# repeat: "en1234,en1235,en1236,en1237,en1357,en1358,en1359,en1360,en2048,en2345,en2346,en2347,en2348,en2468,en2469,en2470,en2471"
# paraphrase: "en_p1,en_p2,en_p3,en_p4,en_p5,en_p6,en_p7,en_p8,en_p9,en_p10,en_p11,en_p12,en_p13,en_p14,en_p15,en_p16,en_p17"
parser.add_argument("--task", help="xgpqa or xmgsm")
parser.add_argument("--model_size", help="Parameter size in billions")
parser.add_argument("--mode", help="scoring mode")
parser.add_argument("--firstk", help="first k languages in mode 4")
parser.add_argument("--model", help="name of the evaluated model", default="qwen")
args=parser.parse_args()

langs = args.langs.split(",")

task = args.task # xgpqa, xmgsm
if "gt-" not in task:
    task_prefix = 'samples_xgpqa_main_native_cot_zeroshot' if 'xgpqa' in task else 'samples_xmgsm_native_cot'
else:
    task_prefix = 'samples_xgpqa_main_google_native_cot_zeroshot' if 'xgpqa' in task else 'samples_xmgsm_native_cot_google'

model_size = args.model_size
if args.model == 'qwen':
    model = f'Qwen2.5-{model_size}-Instruct'
elif args.model == 'llama':
    model = f'Llama-3.1-{model_size}-Instruct'
elif args.model == 'r1-llama':
    model = f'DeepSeek-R1-Distill-Llama-{model_size}'
elif args.model == 'gpt-4o':
    model = 'gpt-4o'
else:
    raise NotImplementedError(f"Unknown model: {args.model}")
task_suffix = f'_{model}_trans' if "st-" in task else ""

log_dir = f'log/{task}/{model}'

mode = eval(args.mode)

# save scoring results, 1 as correct, 0 as incorrect
correct_dict = {lang: [] for lang in langs}


# rule-based judging
use_cot = True
f_judge_log = open(os.path.join(log_dir, 'judge_log.txt'), 'w')

def extract(output):
    if use_cot:
        if '\nAnswer: ' in output:
            output_tail = output.split('\nAnswer: ')[-1]
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


for lang in langs:
    if 'en' in lang:
        log_filename = f'{task_prefix}_{lang}.jsonl'
    else:
        log_filename = f'{task_prefix}_{lang}{task_suffix}.jsonl'
    questions = []
    outputs = []
    references = []
    with jsonlines.open(os.path.join(log_dir, log_filename)) as reader:
        for obj in reader:
            if obj["filter"] == "flexible-extract":
                questions.append(obj['arguments']['gen_args_0']['arg_0'])
                outputs.append(obj['resps'][0][0])
                references.append(obj['target'])
    
    for q, o, r in zip(questions, outputs, references):
        correct_int = 1 if judge(q, o, r) else 0
        correct_dict[lang].append(correct_int)
f_judge_log.close()        

# calculate accuracies in each languages and all-as-one language
acc_dict = {lang: 0 for lang in langs}

for lang in langs:
    correct_ints = correct_dict[lang]
    acc = sum(correct_ints) / len(correct_ints)
    acc_dict[lang] = acc
# print(acc_dict)
if mode == 1:
    for k, v in sorted(acc_dict.items(), key=lambda item: item[1], reverse=True):
        print(f'{k}: {v:.5f}')
    print('='*20)

# breakpoint()
all_lang_correct_ints = np.array([correct_dict[lang] for lang in langs])  # n_lang, n_sample

if mode == 1:
    # best of all
    for threshold in range(0, 10):
        any_correct_ints = (np.sum(all_lang_correct_ints, axis=0) > threshold).astype(int)
        # breakpoint()
        acc_all = sum(any_correct_ints) / len(any_correct_ints)
        # acc_dict['all'] = acc_all
        print(f'>={threshold + 1} correct: {acc_all:.3%}')
elif mode == 2:
    # random choice
    for trial in range(5):
        random_choices = np.random.choice(len(langs), all_lang_correct_ints.shape[1], replace=True)
        rand_correct_ints = [all_lang_correct_ints[random_choices[i], i] for i in range(all_lang_correct_ints.shape[1])]
        # breakpoint()
        acc_all = sum(rand_correct_ints) / len(rand_correct_ints)
        # acc_dict['all'] = acc_all
        print(f'trial {trial}: {acc_all:.3%}')
elif mode == 3:
    # how many languages are correct on each sample
    sum_correct_ints = np.sum(all_lang_correct_ints, axis=0)
    plt.hist(sum_correct_ints)
    plt.savefig(f'fig/hist_{task}_{model}.png')
elif mode == 4:
    # when correct lang <= firstk, find the most frequent languages
    four_correct_ints = (np.sum(all_lang_correct_ints, axis=0) > 0).astype(int) * (np.sum(all_lang_correct_ints, axis=0) <= eval(args.firstk)).astype(int)
    # breakpoint()
    n_fci = 0
    dict_fci_langs = {lang: 0 for lang in langs}
    for idx, fci in enumerate(four_correct_ints.tolist()):
        if fci:
            n_fci += 1
            for lang in langs:
                if correct_dict[lang][idx]:
                    dict_fci_langs[lang] += 1
    for k, v in sorted(dict_fci_langs.items(), key=lambda item: item[1], reverse=True)[:8]:
        print(f'{k}: {v}/{n_fci}')
elif mode == 5:
    # when correct lang >= firstk, find the most frequent languages
    four_correct_ints = (np.sum(all_lang_correct_ints, axis=0) > 0).astype(int) * (np.sum(all_lang_correct_ints, axis=0) >= eval(args.firstk)).astype(int)
    # breakpoint()
    n_fci = 0
    dict_fci_langs = {lang: 0 for lang in langs}
    for idx, fci in enumerate(four_correct_ints.tolist()):
        if fci:
            n_fci += 1
            for lang in langs:
                if correct_dict[lang][idx]:
                    dict_fci_langs[lang] += 1
    for k, v in sorted(dict_fci_langs.items(), key=lambda item: item[1], reverse=True)[:8]:
        print(f'{k}: {v}/{n_fci}')
