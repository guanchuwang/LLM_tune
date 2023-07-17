dataset=c4
model=decapoda-research/llama-7b-hf
soft_token_num=100

for opt in adamw; do
for lr in 0.001; do
for steps in 30000; do
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --main_process_port 20655 --multi_gpu soft_prompt_learning.py \
    --model_name_or_path /home/zx22/prompt/saved_models/llama/pruned_models/decapoda-research/llama-7b-hf_0.75 \
    --model ${model} \
    --dataset ${dataset} \
    --eval_every_steps 100 \
    --seqlen 1024 \
    --soft_token_num ${soft_token_num} \
    --prompt_lr ${lr} \
    --max_steps ${steps} \
    --optimizer ${opt} \
    --output_dir sparsegpt/sp75/${opt}_lr${lr}_steps${steps}_token${soft_token_num}/${dataset} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 2>&1 | tee ./logs/log_sparsegpt_sp75_${opt}_lr${lr}_${dataset}_steps${steps}_token${soft_token_num}.txt
done 
done
done

# 2>&1 | tee ./logs/log_sparsegpt_sp50_${opt}_lr${lr}_${dataset}_steps${steps}_token${soft_token_num}.txt