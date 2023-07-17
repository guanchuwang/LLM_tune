**Run experiments for gptq**

```
cd gptq
CUDA_VISIBLE_DEVICES=6 python llama.py decapoda-research/llama-7b-hf c4 --wbits 3 --true-sequential --act-order --new-eval --save --save_dir /home/zx22/prompt/saved_models/llama/pruned_models

cd sparsegpt
CUDA_VISIBLE_DEVICES=7 python llama.py decapoda-research/llama-7b-hf c4 --new-eval --save --save_dir /home/zx22/prompt/saved_models/llama/pruned_models --sparsity 0.75
```