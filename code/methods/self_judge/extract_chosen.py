import json

def load_queries(query_files: str, half: bool = True):
    samples = []
    for i, query_file in enumerate(query_files, start=1):
        with open(query_file, "r") as q_file:
            lines = q_file.readlines()
            if half:
                lines = lines[len(lines) // 2:]
            for j, line in enumerate(lines):
                q = json.loads(line)
                del q["doc"]
                if i == 1:
                    samples.append([q])
                else:
                    samples[j].append(q)
    return samples

def load_judgement(judge_file: str):
    model_answers = {}
    with open(judge_file) as fin:
        for line in fin:
            line = json.loads(line)
            model_answers[line["query_id"]] = line
    return model_answers

def main():
    model = "DeepSeek-R1-Distill-Llama-70B"
    sample_files = [f"../judge-log/xgpqa/multilingual/{model}/samples_xgpqa_main_native_cot_zeroshot_{i}.jsonl" for i in ["ar", "es", "ko", "sr"]]
    judgement_file = f"../judge-log/xgpqa/multilingual/{model}/judgment_en_prob.jsonl"
    new_sample_file = f"../judge-log/xgpqa/multilingual/{model}/samples_xgpqa_main_native_cot_zeroshot_self_judge_en_prob.jsonl"

    samples = load_queries(sample_files)
    judgments = load_judgement(judgement_file)
    assert len(samples) == len(judgments)

    chosen_samples = []
    for i in range(len(samples)):
        chosen_id = int(judgments[i]["choice"]) - 1
        chosen_samples.append(samples[i][chosen_id])

    with open(new_sample_file, "w") as fout:
        for sample in chosen_samples:
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    main()