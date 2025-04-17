TASK=$1
MODEL=$2
TRANSLATE=$3
export OPENAI_API_KEY=$4
BASE_URL=$5

export SYS_PROMPT_STYLE=$TASK
export EXP_TASK="eval"
MAX_TOKENS=4096

if [[ "$TASK" == "gpqa" ]]; then
    PREFIX="xgpqa/${MODEL}/samples_xgpqa_main_native_cot_zeroshot"
else
    PREFIX="xmgsm/${MODEL}/samples_xmgsm_native_cot"
fi

# PATH_TASKS=../BenchMAX/tasks
PATH_TASKS=/cpfs01/shared/XNLP_H800/huangxu/xeval/tasks

mv log/${PREFIX}_en.jsonl log/${PREFIX}_en0.jsonl

# for SEED in 1357 2468 1234 2345 1358 1359 1360 2469 2470 2471 1235 1236 1237 2346 2347 2348 2048
for SEED in 1357 2468 1234 2345 1358 1359 1360 2469
do
    if [[ "$TASK" == "gpqa" ]]; then
        # gpqa
        lm_eval --model openai-chat-completions --tasks xgpqa_main_native_cot_zeroshot_en --model_args base_url=${BASE_URL},model=gpt-4o,num_concurrent=16,max_retries=10,seed=${SEED} --gen_kwargs max_gen_toks=${MAX_TOKENS},do_sample=True,temperature=0.1 --batch_size auto --apply_chat_template=True --include_path $PATH_TASKS --log_samples --output_path log/xgpqa
    else
        # mgsm
        lm_eval --model openai-chat-completions --tasks xmgsm_native_cot_en --model_args base_url=${BASE_URL},model=gpt-4o,num_concurrent=16,max_retries=10,seed=${SEED} --gen_kwargs max_gen_toks=${MAX_TOKENS},do_sample=True,temperature=0.1 --batch_size auto --apply_chat_template=True --include_path $PATH_TASKS --log_samples --output_path log/xmgsm --verbosity WARNING
    fi

    mv log/${PREFIX}_en.jsonl log/${PREFIX}_en${SEED}.jsonl
done

mv log/${PREFIX}_en0.jsonl log/${PREFIX}_en.jsonl
