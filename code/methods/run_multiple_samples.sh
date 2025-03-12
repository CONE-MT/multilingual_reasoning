export CUDA_VISIBLE_DEVICES=$1
MODEL=$2
SIZE=$3
EXP=$4
LENGTH=$5
TASK=$6


TP_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
LANG=en
export SYS_PROMPT_STYLE=${EXP}
export EXP_MODEL_NAME=${MODEL}
export EXP_TASK=exp-${TASK}
echo $LENGTH


if [[ "$MODEL" == "qwen" ]]; then
    MODEL_NAME="Qwen2.5-${SIZE}-Instruct"
    if [[ "$LENGTH" == "long" ]]; then
        MAX_TOKENS=4096
    else
        MAX_TOKENS=1024
    fi
elif [[ "$MODEL" == "llama" ]]; then
    MODEL_NAME="Llama-3.1-${SIZE}-Instruct"
    if [[ "$LENGTH" == "long" ]]; then
        MAX_TOKENS=4096
    else
        MAX_TOKENS=1024
    fi
elif [[ "$MODEL" == "r1-llama" ]]; then
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

LOG_PATH=log/${TASK}

if [[ "$TASK" == "xgpqa" ]]; then
    PREFIX="${EXP}/${MODEL_NAME}/samples_xgpqa_main_native_cot_zeroshot"
else
    PREFIX="${EXP}/${MODEL_NAME}/samples_xmgsm_native_cot"
fi

PATH_TASKS=./prompt_tasks
PATH_MODELS=../../hf_hub

mv ${LOG_PATH}/${PREFIX}_${LANG}_${EXP}.jsonl ${LOG_PATH}/${PREFIX}_${LANG}0_${EXP}.jsonl

for SEED in 1234 2345 1357 2468
do
    if [[ "$TASK" == "xgpqa" ]];then
        lm_eval --model vllm --tasks xgpqa_main_native_cot_zeroshot_en_${EXP} --model_args pretrained=${PATH_MODELS}/${MODEL_NAME},tensor_parallel_size=${TP_SIZE},enable_prefix_caching=True,max_model_len=8192,gpu_memory_utilization=0.95,enforce_eager=True,seed=${SEED} --gen_kwargs max_gen_toks=${MAX_TOKENS},do_sample=True,temperature=0.1 --batch_size auto --apply_chat_template=True --include_path ${PATH_TASKS}/${TASK} --log_samples --output_path log/${TASK}/${EXP}
    else 
        lm_eval --model vllm --tasks xmgsm_native_cot_en_${EXP} --model_args pretrained=${PATH_MODELS}/${MODEL_NAME},tensor_parallel_size=${TP_SIZE},enable_prefix_caching=True,max_model_len=8192,gpu_memory_utilization=0.95,enforce_eager=True,seed=${SEED} --gen_kwargs max_gen_toks=${MAX_TOKENS},do_sample=True,temperature=0.1 --batch_size auto --apply_chat_template=True --include_path ${PATH_TASKS}/${TASK} --log_samples --output_path log/${TASK}/${EXP}
    fi

    mv ${LOG_PATH}/${PREFIX}_${LANG}_${EXP}.jsonl ${LOG_PATH}/${PREFIX}_${LANG}${SEED}_${EXP}.jsonl
done

mv ${LOG_PATH}/${PREFIX}_${LANG}0_${EXP}.jsonl ${LOG_PATH}/${PREFIX}_${LANG}_${EXP}.jsonl
