task=$1
model=$2

for k in 2 4 6 8 10 12 14 16
do
    # echo "k = $k"
    per_k=$((k / 2))
    python findk_mix_gpqa_mgsm_parallel.py --task $task --model $model --vote False --k_en $per_k --k_non_en $per_k
done
