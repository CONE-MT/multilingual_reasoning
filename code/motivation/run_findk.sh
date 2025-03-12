task=$1
model=$2

echo "multilingual"
for k in {1..17}
do
    # echo "k = $k"
    python findk_gpqa_mgsm_parallel.py --setting multilingual --task $task --model $model --vote True --k $k
done

# echo "repeat"
# for k in {1..17}
# do
#     # echo "k = $k"
#     python findk_gpqa_mgsm_parallel.py --setting repeat --task $task --model $model --vote True --k $k --langs en1234,en1235,en1236,en1237,en1357,en1358,en1359,en1360,en2345,en2346,en2347,en2348,en2468,en2469,en2470,en2471,en2048
# done

# echo "paraphrase"
# for k in {1..17}
# do
#     # echo "k = $k"
#     python findk_gpqa_mgsm_parallel.py --setting paraphrase --task $task --model $model --vote True --k $k --langs en_p1,en_p2,en_p3,en_p4,en_p5,en_p6,en_p7,en_p8,en_p9,en_p10,en_p11,en_p12,en_p13,en_p14,en_p15,en_p16,en_p17
# done
