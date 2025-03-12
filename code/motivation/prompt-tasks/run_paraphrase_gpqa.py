import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import jsonlines
import sys


para_lang = sys.argv[1]

data_path = f'/cpfs01/shared/XNLP_H800/huangxu/xeval/data/gpqa/gpqa_main_{para_lang}.jsonl'
save_path = '/cpfs01/shared/XNLP_H800/gaochangjiang/workbench/reasoning/code/motivation/data/gpqa'

model_path = "/cpfs01/shared/XNLP_H800/hf_hub/Qwen2.5-72B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompts = []
informations = []
task_prompts = {
    'en': 'Please paraphrase this text',
    'ja': 'このテキストを言い換えてください',
    'vi': 'Xin hãy diễn giải lại văn bản này',
    'ru': 'Пожалуйста, перефразируйте этот текст',
    'fr': 'Veuillez paraphraser ce texte',
    'es': 'Por favor parafrasee este texto',
    'ar': 'يرجى إعادة صياغة هذا النص',
    'sr': 'Молим вас да парафразирате овај текст'
}
task_prompt = task_prompts[para_lang]

with jsonlines.open(data_path) as reader:
    for obj in reader:
        assert obj['Question']
        assert obj['Correct Answer'] and obj['Incorrect Answer 1'] and obj['Incorrect Answer 2'] and obj['Incorrect Answer 3']
        assert obj['Subdomain'] and obj['High-level domain'] and obj['Record ID'] and obj['Canary String']
        
        prompt = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': 'Please give your direct answer with nothing else.'}, 
             {'role': 'user', 'content': f"{task_prompt}:\n{obj['Question']}"}], 
            tokenize=False,
            max_length=4096, 
            truncation=True, 
            add_generation_prompt=True,
        )
        
        prompts.append(prompt)
        informations.append({
            'Correct Answer': obj['Correct Answer'],
            'Incorrect Answer 1': obj['Incorrect Answer 1'],
            'Incorrect Answer 2': obj['Incorrect Answer 2'],
            'Incorrect Answer 3': obj['Incorrect Answer 3'],
            'Subdomain': obj['Subdomain'],
            'High-level domain': obj['High-level domain'],
            'Record ID': obj['Record ID'],
            'Canary String': obj['Canary String'],
            })


llm = LLM(
    model=model_path, 
    tensor_parallel_size=torch.cuda.device_count(),
    enforce_eager=True,
    gpu_memory_utilization=0.95,
    enable_prefix_caching=True,
    dtype='bfloat16',
    max_model_len=8192,
    trust_remote_code=True,
)


# paras = ['p1', 'p2', 'p3', 'p4']
# seeds = [1357, 2468, 1234, 2345]

paras = ['p17']
seeds = [2048]

# paras = [f'p{k}' for k in range(1, 17)]
# seeds = [1357, 2468, 1234, 2345, 1235, 1236, 1237, 1358, 1359, 1360, 2346, 2347, 2348, 2469, 2470, 2471, 2048]

for para, seed in zip(paras, seeds):
    sampling_params = SamplingParams(
        temperature=0.6, 
        min_tokens=1,
        max_tokens=4096,
        seed=seed
    )
    
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    
    for output, info in zip(outputs, informations):
        paraphrased_question = output.outputs[0].text.strip()
        info['Question'] = paraphrased_question
    
    with jsonlines.open(f'{save_path}/gpqa_main_{para_lang}_{para}.jsonl', 'w') as writer:
        writer.write_all(informations)
    
