dataset=ptb
model=facebook/opt-350m

for opt in adamw; do
for lr in 0.001; do
for steps in 30000; do
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name_or_path ${model} \
    --model ${model} \
    --dataset ${dataset} \
    --eval_every_steps 100 \
    --seqlen 1024 \
    --lora_dim 16 \
    --soft_token_num 0 \
    --lr ${lr} \
    --max_steps ${steps} \
    --optimizer ${opt} \
    --output_dir gptq/${opt}_lr${lr}_steps${steps}_token${soft_token_num}/${dataset} \
    --per_device_train_batch_size 2 2>&1 | tee ./logs/log_gptq_${opt}_lr${lr}_${dataset}_steps${steps}_token${soft_token_num}.txt
done 
done
done

# decapoda-research/llama-7b-hf \
# | tee ./logs/log_${opt}_lr${lr}_${dataset}_steps${steps}.txt
# --per_device_eval_batch_size 1 \
# taskset -c 64-127
# /home/zx22/prompt/saved_models/llama/pruned_models/decapoda-research/llama-7b-hf_4bit \