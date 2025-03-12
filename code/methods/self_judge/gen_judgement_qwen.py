import re
import os
import argparse
import json
import random
import time
import concurrent.futures
from tqdm import tqdm

import openai

API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
# SYS_PROMPT = "Please act as an impartial judge and evaluate the quality of the answers provided in two languages to the problem displayed below. You will be given Answer_A and Answer_B, each containing reasoning and a final answer. Your job is to evaluate which answer is better.\n\nBegin your evaluation by generating your own answer to the problem. You must provide your answer before judging any answers.\n\nWhen evaluating the given answers, compare both answers with your answer. You must choose one of the answers according to the correctness of the final answers, neglecting the languages used.\n\nThen if the two final answers are the same, evaluate the reasoning process. Consider which answer presents a clearer, more logical, and well-structured explanation.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Answer_A is better: [[A>B]]\n2. Tie, relatively the same: [[A=B]]\n3. Answer_B is better: [[B>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."
SYS_PROMPT = """Please act as an impartial judge and evaluate the quality of the answers provided in two languages to the problem displayed below. You will be given Answer_A and Answer_B, each containing a reasoning process and a final answer. Your job is to evaluate which answer is better.

Firstly, evaluate the reasoning process and the final answer of each answer, neglecting the language used.

Then make your choice according to the correctness of the final answers, regardless of the reasoning process. Always choose the correct final answer even if the reasoning contains mistakes. If the two final answers are the same, consider which answer presents a clearer, more logical, and well-structured reasoning process.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Answer_A is better: [[A>B]]
2. Tie, relatively the same: [[A=B]]
3. Answer_B is better: [[B>A]]

Example output: "My final verdict is tie: [[A=B]]"."""
PROMPT_TEMPLATE = "<|Problem|>\n{problem}\n\n<|The Start of Answer_A|>\n{response_1}\n<|The End of Answer_A|>\n\n<|The Start of Answer_B|>\n{response_2}\n<|The End of Answer_B|>"
regex_pattern = re.compile(r"\[\[([AB<>=]+)\]\]")

def load_queries(query_files: str, half: bool = True):
    """Load queries from a file."""

    def extract_query(text):
        # return text.split("<|im_start|>user\n")[1].split("<|im_end|>")[0].strip()
        # return text.split("Question: ")[1].split("\nQuestion Translation into")[0].strip()
        return text.split("What is the correct answer to this question:")[1].split("Let's think step by step:")[0].strip()

    def extract_response(text):
        if "<think>" in text:
            if "</think>" in text:
                return text.split("</think>")[1].strip()
            else:
                return text.replace("<think>", "").strip()
        return text

    queries = []
    with open("/cpfs01/shared/XNLP_H800/gaochangjiang/workbench/reasoning/code/methods/judge-log/xgpqa/multilingual/Qwen2.5-72B-Instruct/samples_xgpqa_main_native_cot_zeroshot_en.jsonl", "r") as q_file:
        lines = q_file.readlines()
        if half:
            lines = lines[:len(lines) // 2]
        for line in lines:
            q = json.loads(line)
            queries.append({"query_id": q["doc_id"], "query": extract_query(q["arguments"]["gen_args_0"]["arg_0"]), "answers": []})
    for i, query_file in enumerate(query_files, start=1):
        with open(query_file, "r") as q_file:
            lines = q_file.readlines()
            if half:
                lines = lines[:len(lines) // 2]
            for j, line in enumerate(lines):
                q = json.loads(line)
                queries[j]["answers"].append({"data_id": str(i), "content": extract_response(q["resps"][0][0])})
    return queries

def load_answers(answer_file: str):
    model_answers = {}
    if os.path.exists(answer_file):
        with open(answer_file) as fin:
            for line in fin:
                line = json.loads(line)
                model_answers[line["query_id"]] = line
    return model_answers


def chat_completion_openai(model, messages, temperature, max_tokens, api_dict=None):
    if api_dict:
        client = openai.OpenAI(
            base_url=api_dict["api_base"],
            api_key=api_dict["api_key"],
        )
    else:
        client = openai.OpenAI()
    
    output = "$ERROR$"
    for _ in range(API_MAX_RETRY):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                )
            output = completion.choices[0].message.content
            # print(output)
            break
        except openai.RateLimitError as e:
            print(type(e), e)
            time.sleep(API_RETRY_SLEEP)
        except openai.BadRequestError as e:
            print(messages)
            print(type(e), e)
        except KeyError:
            print(type(e), e)
            break
    
    return output

def get_score(judgment, pattern, pairwise=True):
    matches = pattern.findall(judgment)
    matches = [m for m in matches if m != ""]
    if len(set(matches)) == 0:
        return None, True
    elif len(set(matches)) == 1:
        if pairwise:
            return matches[0].strip("\n"), False
        return int(matches[0])
    else:
        return None, False

def get_answer(model, conv, temperature, max_tokens):
    api_dict = {
        "api_base": "http://localhost:8000/v1",
        "api_key": "EMPTY"
    }
    output = chat_completion_openai(model, conv, temperature, max_tokens, api_dict)
    if "<think>" in output:
        if "</think>" in output:
            output = output.split("</think>")[1].strip()
        else:
            output = output.replace("<think>", "").strip()
    return output


def battle(answer_0, answer_1, problem, model, temperature, max_tokens):
    num_games = 2
    number_of_judgment_attempts = 2
    battle_result = {
        "answer_0_data_id": answer_0["data_id"],
        "answer_1_data_id": answer_1["data_id"],
        "game_results": [],
        "winner": ""
    }
    response_1 = answer_0["content"]
    response_2 = answer_1["content"]
    for game in range(num_games):
        conv = [{"role": "system", "content": SYS_PROMPT}]

        prompt_args = {}
        prompt_args["problem"] = problem
        if game % 2 == 1: # swap position
            response_1, response_2 = response_2, response_1
        prompt_args["response_1"] = response_1
        prompt_args["response_2"] = response_2
        user_prompt = PROMPT_TEMPLATE.format(**prompt_args)
        conv.append({"role": "user", "content": user_prompt})

        judgment = ""
        for _ in range(number_of_judgment_attempts):
            new_judgment = get_answer(
                model,
                conv,
                temperature,
                max_tokens
            )

            judgment += ("\n" + new_judgment)

            score, try_again = get_score(judgment, regex_pattern)

            conv.append({"role": "assistant", "content": new_judgment})

            if not try_again:
                break

            conv.append({"role": "user", "content": "continue your judgment and finish by outputting a final verdict label"})

        result = {
            "user_prompt": conv[1]["content"],
            "judgment": judgment,
            "score": score
        }
        battle_result["game_results"].append(result)

    def compute_score(game_result):
        score_0 = score_1 = 0
        match game_result["score"]:
            case "A>B":
                score_0 += 1
            case "B>A":
                score_1 += 1
        return score_0, score_1

    score_a, score_b = compute_score(battle_result["game_results"][0])
    reverve_scores = compute_score(battle_result["game_results"][1])
    score_a += reverve_scores[1]
    score_b += reverve_scores[0]
    if score_a > score_b:
        battle_result["winner"] = answer_0["data_id"]
    elif score_a < score_b:
        battle_result["winner"] = answer_1["data_id"]
    else:
        battle_result["winner"] = "tie"
    return battle_result

def judgment(**args):
    query = args["query"]
    output_file = args["output_file"]
    model = args["model"]
    temperature = args["temperature"]
    max_tokens = args["max_tokens"]

    problem = query["query"]
    answers = query["answers"]
    num_answers = len(answers)
    output = {
        "query_id": query["query_id"],
        "judge": model,
        "battles": [],
        "scores": {answer["data_id"]: 0 for answer in answers},
        "choice": None
    }

    for i in range(num_answers - 1):
        for j in range(i + 1, num_answers):
            battle_result = battle(answers[i], answers[j], problem, model, temperature, max_tokens)
            if battle_result["winner"] == "tie":
                output["scores"][answers[i]["data_id"]] += 1
                output["scores"][answers[j]["data_id"]] += 1
            else:
                output["scores"][battle_result["winner"]] += 2
            output["battles"].append(battle_result)

    max_score = max(output["scores"].values())
    winner_ids = [data_id for data_id, score in output["scores"].items() if score == max_score]
    output["choice"] = random.choice(winner_ids)


    # answer_set = [query["conversation"][1]["content"]]
    # id_set = [1]
    # if not query["conversation"][2]["content"] in answer_set:
    #     answer_set.append(query["conversation"][2]["content"])
    #     id_set.append(2)
    # if not query["conversation"][3]["content"] in answer_set:
    #     answer_set.append(query["conversation"][3]["content"])
    #     id_set.append(3)

    # match len(answer_set):
    #     case 1:
    #         output["choice"] = query["conversation"][1]["data_id"]
    #     case 2:
    #         output["battles"].append(battle(query["conversation"][id_set[0]], query["conversation"][id_set[1]], source, model, temperature, max_tokens, lang))
    #         output["choice"] = output["battles"][0]["winner"]
    #     case 3:
    #         id_set = random.sample(id_set, k=3)
    #         first_battle = battle(query["conversation"][id_set[0]], query["conversation"][id_set[1]], source, model, temperature, max_tokens, lang)
    #         output["battles"].append(first_battle)
    #         first_winner = first_battle["winner"]
    #         first_winner_id = id_set[0] if first_winner == query["conversation"][id_set[0]]["data_id"] else id_set[1]
    #         second_battle = battle(query["conversation"][first_winner_id], query["conversation"][id_set[2]], source, model, temperature, max_tokens, lang)
    #         output["battles"].append(second_battle)
    #         output["choice"] = second_battle["winner"]
    #     case _:
    #         raise RuntimeError

    with open(output_file, "a") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature",  type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--judge-model", type=str, default="/cpfs01/shared/XNLP_H800/hf_hub/Qwen2.5-72B-Instruct")
    parser.add_argument("--parallel", type=int, default=256)
    args = parser.parse_args()
    print(args)

    # print(f'judge model: {configs["judge_model"]}, baseline: {configs["baseline"]}, baseline model: {configs["baseline_model"]}, reference: {configs["reference"]}, '
    #       + f'reference models: {configs["ref_model"]}, temperature: {configs["temperature"]}, max tokens: {configs["max_tokens"]}, pairwise: {configs["pairwise"]}')
    # model = "DeepSeek-R1-Distill-Llama-70B"
    model = os.path.basename(args.judge_model)
    query_files = [f"../judge-log/xgpqa/multilingual/{model}/samples_xgpqa_main_native_cot_zeroshot_{i}.jsonl" for i in ["en", "es", "ja", "th"]]
    output_file = f"../judge-log/xgpqa/multilingual/{model}/judgment_en_prob.jsonl"
    # query_files = [f"../judge-log/gt-xmgsm/multilingual/{model}/samples_xmgsm_native_cot_google_{i}.jsonl" for i in ["ar", "en", "es", "hu"]]
    # output_file = f"../judge-log/gt-xmgsm/multilingual/{model}/judgment.jsonl"
    # query_files = [f"../judge-log/xgpqa/paraphrase/{model}/samples_xgpqa_main_native_cot_zeroshot_en_p{i}.jsonl" for i in range(1, 5)]
    # output_file = f"../judge-log/xgpqa/paraphrase/{model}/judgment_en_prob.jsonl"
    # query_files = [f"../judge-log/xmgsm/paraphrase/{model}/samples_xmgsm_native_cot_en_p{i}.jsonl" for i in range(1, 5)]
    # output_file = f"../judge-log/xmgsm/paraphrase/{model}/judgment.jsonl"
    # query_files = [f"../judge-log/xmgsm/en/{model}/samples_xmgsm_native_cot_en{i}.jsonl" for i in ["1234", "1357", "2345", "2468"]]
    # output_file = f"../judge-log/xmgsm/en/{model}/judgment.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    queries = load_queries(query_files)
    existing_judgments = load_answers(output_file)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        count = 0
        for query in queries:
            query_id = query["query_id"]

            if query_id in existing_judgments:
                count += 1
                continue

            kwargs = {}
            kwargs["query"] = query
            kwargs["output_file"] = output_file
            kwargs["model"] = args.judge_model
            kwargs["temperature"] = args.temperature
            kwargs["max_tokens"] = args.max_tokens
            future = executor.submit(judgment, **kwargs)
            futures.append(future)

        if count > 0:
            print(f"{count} number of existing judgments")

        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()