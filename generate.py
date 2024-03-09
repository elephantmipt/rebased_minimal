import os.path

import pyrallis
import requests
import wandb

import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from rebased.model import LanguageModel
from rebased.config import ModelConfig
from safetensors.torch import load_model


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

@torch.no_grad()
def main():
    prompt = "Hi, my name is"
    max_tokens = 20
    if not os.path.exists("model.safetensors"):
        from huggingface_hub.file_download import hf_hub_download
        hf_hub_download("elephantmipt/rebased_155m", "model.safetensors")
    if not os.path.exists("config.yaml"):
        from huggingface_hub.file_download import hf_hub_download
        hf_hub_download("elephantmipt/rebased_155m", "config.yaml")
    with open("config.yaml", "r") as f:
        model_config = pyrallis.load(ModelConfig, f)

    model = LanguageModel(model_config)
    load_model(model, "model.safetensors")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.backbone.embeddings.device = device
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b")
    tokenized = tokenizer.encode(prompt, return_tensors="pt")
    torch.set_grad_enabled(False)
    for i in range(max_tokens):
        out = model(tokenized.to(device))
        logits = top_k_top_p_filtering(out.logits[:, -1][0], top_p=0.9, top_k=10)
        probabilities = F.softmax(logits, dim=-1)
        token = torch.multinomial(probabilities, 1)
        tokenized = torch.cat((tokenized.to(device), token.unsqueeze(0)), dim=-1)
        print(tokenizer.decode(tokenized[0].cpu()))
        if token == tokenizer.eos_token_id:
            break
        
    

if __name__ == "__main__":
    main()

