import json
import numpy as np
import argparse


# args
parser=argparse.ArgumentParser()
parser.add_argument("--task", help="xgpqa or xmgsm")
parser.add_argument("--langs", help="languages to choose")
parser.add_argument("--model", help="name of the evaluated model", default="qwen")
parser.add_argument("--lang_setting", help="en or multilingual", default="multilingual")
args=parser.parse_args()


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


def main():
    judgement_file = f"../judge-log/{args.task}/{args.lang_setting}/{model}/judgment_en_prob.jsonl"
    langs = args.langs.split(",")
    results = {langs[0]: np.zeros(448), langs[1]: np.zeros(448), langs[2]: np.zeros(448), langs[3]: np.zeros(448)}
    with open(judgement_file, "r") as fin:
        for line in fin:
            line = json.loads(line)
            choice = line["choice"]
            match choice:
                case "1":
                    results[langs[0]][line["query_id"]] = 1
                case "2":
                    results[langs[1]][line["query_id"]] = 1
                case "3":
                    results[langs[2]][line["query_id"]] = 1
                case "4":
                    results[langs[3]][line["query_id"]] = 1
    np.save(f"{args.model}_{args.task}_chosen_lang.npy", results)
    
if __name__ == "__main__":
    main()