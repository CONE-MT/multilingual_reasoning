TASK=$1
MODEL=$2
LANGS=$3

if [[ "$MODEL" == "qwen" ]]; then
    # SIZE=72B
    SIZE=7B
elif [[ "$MODEL" == "llama" ]]; then
    # SIZE=70B
    SIZE=8B
elif [[ "$MODEL" == "r1-llama" ]]; then
    # SIZE=70B
    SIZE=8B
else
    echo "Unknown model $MODEL"
    exit
fi

python score_gpqa_mgsm.py  --task $TASK --mode 1 --model $MODEL --model_size $SIZE --langs $LANGS

python vote_gpqa_mgsm.py  --task $TASK --model $MODEL --model_size $SIZE --langs $LANGS
