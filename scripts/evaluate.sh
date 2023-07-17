dataset=ptb
ntokens=100

CUDA_VISIBLE_DEVICES=0 python evaluate_llama.py \
    --model-name-or-path /home/zx22/prompt/saved_models/llama/pruned_models/decapoda-research/llama-7b-hf_0.5 \
    --ckpt /home/zx22/prompt/saved_models/soft_prompt_llama/sparsegpt/sp50/adamw_lr0.001_steps30000_token100/${dataset}/best.pth \
    --dataset ${dataset} \
    --ntoken ${ntokens} \
    --dtype bfloat16