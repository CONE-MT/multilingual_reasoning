export CUDA_VISIBLE_DEVICES=$1
MODEL=$2
LENGTH=$3
LANG_SETTING=$4
# LANG_SETTING: "multilingual" or "repeat" or "paraphrase"
TASK=$5

TP_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
if [[ "$LANG_SETTING" == "multilingual" ]]; then
    export SYS_PROMPT_STYLE="multi_judge"
else
    if [[ "$TASK" == "xgpqa" ]]; then
        export SYS_PROMPT_STYLE="gpqa"
    else
        export SYS_PROMPT_STYLE="mgsm"
    fi
fi
export EXP_TASK=exp-${TASK}
echo $LENGTH


if [[ "$MODEL" == "qwen" ]]; then
    SIZE=72B
    MODEL_NAME="Qwen2.5-${SIZE}-Instruct"
    if [[ "$LENGTH" == "long" ]]; then
        MAX_TOKENS=4096
    else
        MAX_TOKENS=1024
    fi
elif [[ "$MODEL" == "llama" ]]; then
    SIZE=70B
    MODEL_NAME="Llama-3.1-${SIZE}-Instruct"
    if [[ "$LENGTH" == "long" ]]; then
        MAX_TOKENS=4096
    else
        MAX_TOKENS=1024
    fi
elif [[ "$MODEL" == "r1-llama" ]]; then
    SIZE=70B
    MODEL_NAME="DeepSeek-R1-Distill-Llama-${SIZE}"
    SELF_TRANS_PATH="DeepSeek-R1-Distill-Llama-${SIZE}_trans"
    if [[ "$LENGTH" == "long" ]]; then
        MAX_TOKENS=8192
    else
        MAX_TOKENS=4096
    fi
else
    echo "Unknown model $MODEL"
    exit
fi

echo $MAX_TOKENS

LOG_PATH=judge-log/${TASK}/${LANG_SETTING}
if [[ "$TASK" == "xgpqa" ]]; then
    PREFIX="${EXP}/${MODEL_NAME}/samples_xgpqa_main_native_cot_zeroshot"
else
    PREFIX="${EXP}/${MODEL_NAME}/samples_xmgsm_native_cot"
fi

PATH_TASKS=./judge-tasks
PATH_MODELS=../../hf_hub

if [[ "$LANG_SETTING" == "multilingual" ]]; then
    if [[ "$TASK" == "xgpqa" ]]; then
        lm_eval --model vllm --tasks xgpqa_main_native_cot_zeroshot_${MODEL} --model_args pretrained=${PATH_MODELS}/${MODEL_NAME},tensor_parallel_size=${TP_SIZE},enable_prefix_caching=True,max_model_len=8192,gpu_memory_utilization=0.95,enforce_eager=True --gen_kwargs max_gen_toks=${MAX_TOKENS},do_sample=True,temperature=0.1 --batch_size auto --apply_chat_template=True --include_path ${PATH_TASKS}/xgpqa --log_samples --output_path $LOG_PATH
    elif [[ "$TASK" == "gt-xgpqa" ]]; then
        lm_eval --model vllm --tasks xgpqa_main_google_native_cot_zeroshot_${MODEL} --model_args pretrained=${PATH_MODELS}/${MODEL_NAME},tensor_parallel_size=${TP_SIZE},enable_prefix_caching=True,max_model_len=8192,gpu_memory_utilization=0.95,enforce_eager=True --gen_kwargs max_gen_toks=${MAX_TOKENS},do_sample=True,temperature=0.1 --batch_size auto --apply_chat_template=True --include_path ${PATH_TASKS}/gt-xgpqa --log_samples --output_path $LOG_PATH
    else
        echo "Unsupported task: ${TASK}"
    fi
elif [[ "$LANG_SETTING" == "repeat" ]]; then
    for SEED in 1357 2468 1234 2345
    do
        if [[ "$TASK" == "xgpqa" ]]; then
            lm_eval --model vllm --tasks xgpqa_main_native_cot_zeroshot_en --model_args pretrained=${PATH_MODELS}/${MODEL_NAME},tensor_parallel_size=${TP_SIZE},enable_prefix_caching=True,max_model_len=8192,gpu_memory_utilization=0.95,enforce_eager=True,seed=${SEED} --gen_kwargs max_gen_toks=${MAX_TOKENS},do_sample=True,temperature=0.1 --batch_size auto --apply_chat_template=True --include_path ${PATH_TASKS}/xgpqa --log_samples --output_path $LOG_PATH
        else
            echo "Unsupported task: ${TASK}"
        fi

        mv $LOG_PATH/${PREFIX}_en.jsonl $LOG_PATH/${PREFIX}_en${SEED}.jsonl
    done
elif [[ "$LANG_SETTING" == "paraphrase" ]]; then
    if [[ "$TASK" == "xgpqa" ]]; then
        lm_eval --model vllm --tasks xgpqa_main_native_cot_zeroshot_en_para --model_args pretrained=${PATH_MODELS}/${MODEL_NAME},tensor_parallel_size=${TP_SIZE},enable_prefix_caching=True,max_model_len=8192,gpu_memory_utilization=0.95,enforce_eager=True --gen_kwargs max_gen_toks=${MAX_TOKENS},do_sample=True,temperature=0.1 --batch_size auto --apply_chat_template=True --include_path ${PATH_TASKS}/xgpqa-para --log_samples --output_path $LOG_PATH
    else
        echo "Unsupported task: ${TASK}"
    fi
else
    echo "Unknown language setting: $LANG_SETTING"
    exit
fi
