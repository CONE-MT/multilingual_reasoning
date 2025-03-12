set -e

export VLLM_USE_MODELSCOPE=False
export VLLM_USE_V1=1
unset SETUPTOOLS_USE_DISTUTILS

vllm serve ../../../hf_hub/Qwen2.5-72B-Instruct -tp 8 --enable-prefix-caching --gpu-memory-utilization 0.9
# vllm serve ../../../hf_hub/Llama-3.1-70B-Instruct -tp 8 --enable-prefix-caching --gpu-memory-utilization 0.9
# vllm serve ../../../hf_hub/DeepSeek-R1-Distill-Llama-70B -tp 8 --enable-prefix-caching --gpu-memory-utilization 0.9
