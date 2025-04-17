TASK=$1
MODEL=$2
TRANSLATE=$3
export OPENAI_API_KEY=$4
BASE_URL=$5

export SYS_PROMPT_STYLE=$TASK
export EXP_TASK="eval"
MAX_TOKENS=4096

PATH_TASKS=/cpfs01/shared/XNLP_H800/gaochangjiang/workbench/reasoning/code/motivation/prompt-tasks

if [[ "$TASK" == "gpqa" ]]; then
    # gpqa
    lm_eval --model openai-chat-completions --tasks xgpqa_main_native_cot_zeroshot_en_para --model_args base_url=${BASE_URL},model=gpt-4o,num_concurrent=16,max_retries=10 --gen_kwargs max_gen_toks=${MAX_TOKENS},do_sample=True,temperature=0.1 --batch_size auto --apply_chat_template=True --include_path $PATH_TASKS --log_samples --output_path log/xgpqa
else
    # mgsm
    lm_eval --model openai-chat-completions --tasks xmgsm_native_cot_en_para --model_args base_url=${BASE_URL},model=gpt-4o,num_concurrent=16,max_retries=10 --gen_kwargs max_gen_toks=${MAX_TOKENS},do_sample=True,temperature=0.1 --batch_size auto --apply_chat_template=True --include_path $PATH_TASKS --log_samples --output_path log/xmgsm --verbosity WARNING
fi
