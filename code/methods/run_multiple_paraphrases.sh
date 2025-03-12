export CUDA_VISIBLE_DEVICES=$1
MODEL=$2
SIZE=$3
TASK=$4
LANG=$5


TP_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
export SYS_PROMPT_STYLE=$TASK
export EXP_TASK="eval"


if [[ "$MODEL" == "qwen" ]]; then
    MODEL_NAME="Qwen2.5-${SIZE}-Instruct"
    SELF_TRANS_PATH="Qwen2.5-${SIZE}-Instruct_trans"
    MAX_TOKENS=4096
elif [[ "$MODEL" == "llama" ]]; then
    MODEL_NAME="Llama-3.1-${SIZE}-Instruct"
    SELF_TRANS_PATH="Llama-3.1-${SIZE}-Instruct_trans"
    MAX_TOKENS=4096
elif [[ "$MODEL" == "r1-llama" ]]; then
    MODEL_NAME="DeepSeek-R1-Distill-Llama-${SIZE}"
    SELF_TRANS_PATH="DeepSeek-R1-Distill-Llama-${SIZE}_trans"
    MAX_TOKENS=8192
else
    echo "Unknown model $MODEL"
    exit
fi


lm_eval --model vllm --tasks xgpqa_main_native_cot_zeroshot_${LANG}_para --model_args pretrained=/cpfs01/shared/XNLP_H800/gaochangjiang/workbench/reasoning/hf_hub/${MODEL_NAME},tensor_parallel_size=${TP_SIZE},enable_prefix_caching=True,max_model_len=8192,gpu_memory_utilization=0.95,enforce_eager=True --gen_kwargs max_gen_toks=${MAX_TOKENS},do_sample=True,temperature=0.1 --batch_size auto --apply_chat_template=True --include_path /cpfs01/shared/XNLP_H800/gaochangjiang/workbench/reasoning/code/methods/judge-tasks/xgpqa-para --log_samples --output_path /cpfs01/shared/XNLP_H800/gaochangjiang/workbench/reasoning/code/methods/judge-log/xgpqa/paraphrase
