import argparse
import os
import ipdb
from transformers import AutoTokenizer, LlamaForCausalLM
from datasets import load_dataset 
import torch
import torch.nn as nn
from prompt import LLamaPromptTuningLM, llama_loader

parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--model', type=str, default= "decapoda-research/llama-7b-hf")
parser.add_argument('--model-name-or-path', type=str, default= "/home/zx22/prompt/saved_models/gpt/pruned_models/c4/pretrained/facebook/opt-1.3b_0.5")
parser.add_argument('--ckpt', type=str, default=None)
# parser.add_argument('--ckpt', type=str, default= "/scratch/zx22/soft_prompt_results/baseline/ptb/best.ckpt")
# parser.add_argument('--ckpt', type=str, default= "/scratch/zx22/soft_prompt_results/adamw_lr0.001_steps30000/c4/best.ckpt")
# parser.add_argument('--ckpt', type=str, default= "/scratch/zx22/soft_prompt_results/unpruned/ptb/best.ckpt")
parser.add_argument('--dataset', type = str, default = "wikitext2")
parser.add_argument('--dtype', type = str, default = "auto")
parser.add_argument('--ntoken', type = int, default = 50)
parser.add_argument('--set_type', type = str, default = "test", choices=["validation","test"])
args = parser.parse_args()


def prepare_input_and_label(model, inputs_ids):
    # shift right
    padded_input_tokens = model._extend_labels(inputs_ids)
    labels = padded_input_tokens[..., 1:].contiguous()
    input_tokens = padded_input_tokens[..., :-1].contiguous()
    labels[input_tokens<0] = -100
    return labels


@torch.no_grad()
def evaluate(prompt_model, valenc, loss_fct, seqlen):
    prompt_model.eval()
    nlls = []
    n_samples = valenc.size(1) // seqlen
    for i in range(n_samples):
        inputs_ids = valenc[:,i*seqlen:(i+1)*seqlen].cuda()
        labels = prepare_input_and_label(prompt_model, inputs_ids)
        try:
            output = prompt_model(inputs_ids)
        except:
            import ipdb; ipdb.set_trace()
        shift_logits = output.logits[:, :-1, :]
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        neg_log_likelihood = loss.float().mean() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))
    return ppl.item()


if args.dtype == 'auto':
    dtype = 'auto'
elif args.dtype == 'bfloat16':
    dtype = torch.bfloat16
elif args.dtype == 'float16':
    dtype = torch.float16
else:
    raise NotImplementedError

if args.ckpt is None:
    model = LLamaPromptTuningLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype, n_tokens=0)
else:
    model = LLamaPromptTuningLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype, n_tokens=args.ntoken)

print(model.dtype)

if args.ckpt is not None:
    state_dicts = torch.load(args.ckpt)
    soft_prompt_state_dict = state_dicts['model']
    model.soft_prompt.load_state_dict(soft_prompt_state_dict)
tokenizer = llama_loader.LLaMATokenizer.from_pretrained(args.model, use_fast=False)
model.seqlen = model.config.max_position_embeddings
model.seqlen = 1024
model.cuda()
model.eval()

if args.dataset == "wikitext2":
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split=args.set_type)
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
elif args.dataset == "ptb":
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split=args.set_type)
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

elif args.dataset == "c4":
    # follow the implementation in datautils.py of SparseGPT.
    testdata = load_dataset('allenai/c4', 'allenai--c4', 
                                        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, 
                                        split='validation')
    testenc = tokenizer(' '.join(testdata[:1100]['text']), return_tensors='pt')



testenc = testenc.input_ids
nsamples = testenc.numel() // model.seqlen
print("nsamples:", nsamples)
print("seqlen:", model.seqlen)
loss_fct = nn.CrossEntropyLoss(reduction='none')

print(f'dataset {args.dataset}\nckpt: {args.ckpt}\nmodel_name_or_path: {args.model_name_or_path}\n')
print(f"{args.set_type} Perplexity:", evaluate(model, testenc, loss_fct, model.seqlen))
assert 1 == 0


with torch.no_grad():
    nlls = []
    for i in range(nsamples):
        batch = testenc[:,i*model.seqlen:(i+1)*model.seqlen].cuda()
        output = model(batch)
        shift_logits = output.logits[:, :-1, :]
        shift_labels = batch[:, 1:]
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
print(f"Perplexity: {ppl.item():3f}")




