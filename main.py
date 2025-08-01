import tiktoken
import torch
from config import GPT2_124M_Config
from model import GPT
from utils import get_completion

torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")

model = GPT(GPT2_124M_Config)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
# weight tying factored in
print(f"Total parameter count: {total_params - model.tok_embed.weight.numel():,}",)

print(get_completion(model, "How are"))
