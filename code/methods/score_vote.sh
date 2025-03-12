TASK=$1
MODEL=$2
LOG_TYPE=$3
LANG_SETTING=$4
LANGS=$5

python score_gpqa_mgsm.py --task $TASK --model $MODEL --log_type $LOG_TYPE --lang_setting=$LANG_SETTING --exp 5 --langs $LANGS
python vote_gpqa_mgsm.py --task $TASK --model $MODEL --log_type $LOG_TYPE --lang_setting=$LANG_SETTING --exp 5 --langs $LANGS

