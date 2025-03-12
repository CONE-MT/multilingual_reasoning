import jsonlines
import os 
import argparse, sys
import numpy as np
import re
from itertools import combinations
import subprocess
from tqdm import tqdm
import multiprocessing
from multiprocessing import Process, Lock, Queue, Semaphore, Manager


# args
parser=argparse.ArgumentParser()
parser.add_argument("--setting", help="repeat, paraphrase", default="repeat")
parser.add_argument("--ens", help="en runs", default="en1357,en2468,en1234,en2345")
# parser.add_argument("--ens", help="en runs", default="en_p1,en_p2,en_p3,en_p4")
parser.add_argument("--non_ens", help="languages", default="ar,bn,cs,de,es,fr,hu,ja,ko,ru,sr,sw,te,th,vi,zh")
parser.add_argument("--task", help="xgpqa or xmgsm")
parser.add_argument("--model", help="name of the evaluated model", default="qwen")
parser.add_argument("--k_en", help="how many en runs to aggregate", default="2")
parser.add_argument("--k_non_en", help="how many non-en languages to aggregate", default="2")
parser.add_argument("--vote", help="whether to vote", default="True")
parser.add_argument("--save", default="False")
args=parser.parse_args()


en_runs = args.ens.split(",")
non_en_langs = args.non_ens.split(",")
langs = en_runs + non_en_langs

task = args.task # xgpqa, xmgsm
do_vote = eval(args.vote)
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

log_dir = f'log/{task}/{model}'

# save scoring results, 1 as correct, 0 as incorrect
correct_dict = {lang: [] for lang in langs}


# rule-based judging
use_cot = True
# f_judge_log = open(os.path.join(log_dir, 'judge_log.txt'), 'w')

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
                
    # print(f"Q: {question}\nA: {output}\n{'=' * 30}\nNormalized: {output_norm}\nRef: {answer}\nEval: {eval_result}\n\n", file=f_judge_log)
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
# f_judge_log.close()

# calculate accuracies in each languages
acc_dict = {lang: 0 for lang in langs}
for lang in langs:
    correct_ints = correct_dict[lang]
    acc = sum(correct_ints) / len(correct_ints)
    acc_dict[lang] = acc
print(f"Mean acc: {np.mean(list(acc_dict.values())):.3%}")
# exit()
# prepare for calculating pass@k
all_lang_correct_ints = np.array([correct_dict[lang] for lang in langs])  # n_lang, n_sample
# breakpoint()


# calculate pass@k for each combination, and take the highest and lowest and the average
# for row_combo in tqdm(combinations_of_rows, desc="combinations", total=n_combinations):
def process_combo(row_combo_list, pkd, vad, vcd, instance_lock, result_queue):
    en_combo = ','.join([en_runs[rc] for rc in row_combo_list["en"]])
    non_en_combo = ','.join([non_en_langs[rc] for rc in row_combo_list["non_en"]])
    lang_combo = ','.join([en_combo, non_en_combo])
    
    selected_rows = all_lang_correct_ints[list(row_combo_list['en']) + list(row_combo_list['non_en']), :]
    any_correct_ints = (np.sum(selected_rows, axis=0) > 0).astype(int)
    acc_all = sum(any_correct_ints) / len(any_correct_ints)
    with instance_lock:
        assert lang_combo not in pkd
        pkd[lang_combo] = acc_all
    
    if do_vote:
        result = subprocess.run(
            ['python', 'vote_gpqa_mgsm.py', '--task', args.task, '--model', args.model, '--model_size', model_size, '--langs', lang_combo],
            capture_output=True,  # Capture the output
            text=True  # Return output as a string
        )
        
        with instance_lock:
            vote_acc, vote_const = [eval(r) for r in result.stdout.strip().split(';')]
            
            vad[lang_combo] = vote_acc
            vcd[lang_combo] = vote_const

    result_queue.put(row_combo_list)
    

def process_all_combos(batchsize):
    en_row_indices = range(len(en_runs))  # rows of all_lang_correct_ints
    k_en = eval(args.k_en)  # k candidates in the voting, and pass@k
    en_combinations_of_rows = combinations(en_row_indices, k_en)
    en_combo_tuples = [c for c in en_combinations_of_rows]
    
    non_en_row_indices = range(len(non_en_langs))  # rows of all_lang_correct_ints
    k_non_en = eval(args.k_non_en)  # k candidates in the voting, and pass@k
    non_en_combinations_of_rows = combinations(non_en_row_indices, k_non_en)
    non_en_combo_tuples = [c for c in non_en_combinations_of_rows]
    
    combo_tuples = []
    for en_combo in en_combo_tuples:
        for non_en_combo in non_en_combo_tuples:
            combo_tuples.append({'en': en_combo, 'non_en': non_en_combo})
    n_combinations = len(combo_tuples)  # total number of combos
    # breakpoint()
    
    n_batches = n_combinations // batchsize if n_combinations % batchsize == 0 else n_combinations // batchsize + 1
    for i_batch in tqdm(range(n_batches), desc="batchs"):
    # for i_batch in range(n_batches):
        # breakpoint()
        # print(f"Batch {i_batch + 1} out of {n_batches}")
        idx_start = i_batch * batchsize
        
        instance_lock = Lock()
        queue = Queue()
        processes = []
        
        # breakpoint()
        combo_tuples_batch = combo_tuples[idx_start: idx_start + batchsize]
        for combo in combo_tuples_batch:
            process = Process(target=process_combo, args=(combo, passk_dict, vote_acc_dict, vote_const_dict, instance_lock, queue))
            processes.append(process)
            process.start()
        
        # with tqdm(total=len(combo_tuples_batch), desc='combo batch') as pbar:
        #     for _ in combo_tuples:
        #         result = queue.get()
        #         pbar.update(1)
        #         # print('\n')
        
        for process in processes:
            process.join()
    

if __name__ == "__main__":
    multiprocessing.set_start_method('fork')
    
    with Manager() as manager:
        passk_dict = manager.dict()
        vote_acc_dict = manager.dict()
        vote_const_dict = manager.dict()
        
        process_all_combos(batchsize=32)
    
        best_k_langs = max(passk_dict, key=passk_dict.get)
        best_passk = passk_dict[best_k_langs]
        accs = [f"{acc:.3%}" for acc in [acc_dict[lang] for lang in best_k_langs.split(',')]]
        if do_vote:
            print(f"Best languages: {best_k_langs}\nAcc: {', '.join(accs)}\nPass@k: {best_passk:.3%}\nVoted acc: {vote_acc_dict[best_k_langs]:.3%}\nConsistency: {vote_const_dict[best_k_langs]:.3%}")
        else:
            # print(f"Best languages: {best_k_langs}\nAcc: {', '.join(accs)}\nPass@k: {best_passk:.3%}")
            print(f"Best Pass@k: {best_passk:.3%}")
            # pass

        worst_k_langs = min(passk_dict, key=passk_dict.get)
        worst_passk = passk_dict[worst_k_langs]
        accs = [f"{acc:.3%}" for acc in [acc_dict[lang] for lang in worst_k_langs.split(',')]]
        if do_vote:
            print(f"\nWorst languages: {worst_k_langs}\nAcc: {', '.join(accs)}\nPass@k: {worst_passk:.3%}\nVoted acc: {vote_acc_dict[worst_k_langs]:.3%}\nConsistency: {vote_const_dict[worst_k_langs]:.3%}")
        else:
            print(f"\nWorst languages: {worst_k_langs}\nAcc: {', '.join(accs)}\nPass@k: {worst_passk:.3%}")
            # pass

        avg_passk = np.mean(list(passk_dict.values()))
        print(f"Average pass@k: {avg_passk:.3%}")
        # print(f"{best_passk:.3%};{avg_passk:.3%}")
        
        if do_vote:
            avg_vote_acc = np.mean(list(vote_acc_dict.values()))
            avg_vote_const = np.mean(list(vote_const_dict.values()))
            print(f"Average voted acc: {avg_vote_acc:.3%}")
            print(f"Average consistency: {avg_vote_const:.3%}")

        passk_scores_array = np.array(list(passk_dict.values()))
        if eval(args.save):
            np.save(f"outputs/{task}/{args.model}/mix_{args.setting}_pass{args.k_en}+{args.k_non_en}.npy", passk_scores_array) 
            