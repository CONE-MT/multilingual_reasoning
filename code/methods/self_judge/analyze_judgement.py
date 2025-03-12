import json

def main():
    judgment_file_path = "../judge-log/xgpqa/multilingual/Qwen2.5-72B-Instruct/judgment_en.jsonl"
    with open(judgment_file_path, "r") as judgment_file:
        judgments = [json.loads(line) for line in judgment_file]
    
    scores = [judgment["scores"][judgment["choice"]] for judgment in judgments]
    print(f"Average score: {sum(scores) / len(scores)}")

if __name__ == '__main__':
    main()