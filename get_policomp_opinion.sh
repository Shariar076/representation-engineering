#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

models=(
    "meta-llama/Llama-2-13b-chat-hf"  #--> done
    # "meta-llama/Llama-2-7b-chat-hf"
    "meta-llama/Meta-Llama-3-8B-Instruct" #--> done
    # "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct" #--> done
    "mistralai/Ministral-8B-Instruct-2410" #--> done
    "mistralai/Mistral-Nemo-Instruct-2407" #--> done
    "mistralai/Mistral-7B-Instruct-v0.3" #--> done
    # "mistralai/Mixtral-8x7B-Instruct-v0.1" #--> done
    "Qwen/Qwen2.5-3B-Instruct" #--> done
    "Qwen/Qwen2.5-32B-Instruct" #--> done
    # "EleutherAI/gpt-neox-20b" # not instruct model
    # "lmsys/vicuna-13b-v1.5" # not instruct model
    "meta-llama/Llama-2-70b-chat-hf" #--> done
    # "meta-llama/Meta-Llama-3-70B-Instruct"
    # "meta-llama/Llama-3.1-70B-Instruct"
)

for i in "${!models[@]}"; do
    echo "Testing ${models[i]}"
    python get_policomp_opinion.py --model  ${models[i]} --device 0
done

echo "Iteration complete"
