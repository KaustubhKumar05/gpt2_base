import tiktoken
import torch
from config import GPT2_124M_Config
from model import GPT
from utils import load_weights_into_gpt
from gpt_download import download_and_load_gpt2

tokenizer = tiktoken.get_encoding("gpt2")
device = 'mps' if torch.mps.is_available() else 'cpu'
print("Initialising model...")
model = GPT(GPT2_124M_Config).to(device)
model.eval()

# total_params = sum(p.numel() for p in model.parameters())
# accounting for weight tying
# print(f"Total parameter count: {total_params - model.tok_embed.weight.numel():,}",)
print("Loading weights...")
settings, params = download_and_load_gpt2("124M", "weights")

load_weights_into_gpt(model, params)
model.to(device)

