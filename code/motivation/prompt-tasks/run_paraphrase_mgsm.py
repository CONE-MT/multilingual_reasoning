import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import jsonlines
import sys


para_lang = sys.argv[1]

data_path = f'/cpfs01/shared/XNLP_H800/huangxu/xeval/data/mgsm/mgsm_human_{para_lang}/test/mgsm_human_{para_lang}_test.jsonl'
save_path = '/cpfs01/shared/XNLP_H800/gaochangjiang/workbench/reasoning/code/motivation/data/mgsm'

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
        assert obj['question'] and obj['answer_number']
        
        prompt = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': 'Please give your direct answer with nothing else.'}, 
             {'role': 'user', 'content': f"{task_prompt}:\n{obj['question']}"}], 
            tokenize=False,
            max_length=4096, 
            truncation=True, 
            add_generation_prompt=True,
        )
        
        prompts.append(prompt)
        informations.append({
            'answer': None,
            'answer_number': obj['answer_number'],
            'equation_solution': None
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
# seeds = [1357, 2468, 1234, 2345, 1235, 1236, 1237, 1358, 1359, 1360, 2346, 2347, 2348, 2469, 2470, 2471]

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
        info['question'] = paraphrased_question
    
    with jsonlines.open(f'{save_path}/mgsm_main_{para_lang}_{para}.jsonl', 'w') as writer:
        writer.write_all(informations)
    
