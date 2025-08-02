import tiktoken
import torch
from config import GPT2_124M_Config
from model import GPT
from utils import get_completion, load_weights_into_gpt
from gpt_download import download_and_load_gpt2

# torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")
device = 'mps' if torch.mps.is_available() else 'cpu'
model = GPT(GPT2_124M_Config).to(device)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
# accounting for weight tying
print(f"Total parameter count: {total_params - model.tok_embed.weight.numel():,}",)

# print("Random weights:", get_completion(model, "How are", device=device))

settings, params = download_and_load_gpt2("124M", "weights")
# print(settings, params.keys())

load_weights_into_gpt(model, params)
model.to(device)

print(get_completion(model, "It is a glorious occasion", device=device))
