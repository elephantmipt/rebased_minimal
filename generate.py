import pyrallis

import torch
from transformers import AutoTokenizer

from rebased.model import LanguageModel
from rebased.config import ModelConfig
from safetensors.torch import load_model

@torch.no_grad
def main():
    
    with open("config.yaml", "r") as f:
        config = pyrallis.load(ModelConfig, f)
    model = LanguageModel(config)
    
    load_model(model, "model.safetensors")
    model.tie_weights()
    prompt = "Hello, my name is "
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b")
    tokenized = tokenizer.encode(prompt, return_tensors="pt")
    for i in range(10):
        out = model(tokenized)
        token = out[:, -1].argmax(-1)
        tokenized = torch.cat((tokenized, token.unsqueeze(0)), dim=-1)
        print(tokenized)
        print(tokenizer.decode(tokenized[0]))
        
    

if __name__ == "__main__":
    main()

