import tiktoken
import torch

from config import GPT2_124M_Config


def tokenize(text, tokenizer):
    encoded_text = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    # x -> [1, x]
    tokens = torch.tensor(encoded_text).unsqueeze(0)
    return tokens

def tokens_to_text(tokens, tokenizer):
    flat_list = tokens.squeeze(0)
    text = tokenizer.decode(flat_list.tolist())
    return text


def get_completion(model, text, max_new_tokens = 10, ctx_size = GPT2_124M_Config["ctx_length"], tokenizer = tiktoken.get_encoding("gpt2")):
    tokens = tokenize(text, tokenizer)
    for _ in range(max_new_tokens):
        truncated_tokens = tokens[:, -ctx_size:]

        with torch.no_grad():
            logits = model(truncated_tokens)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        next_token = torch.argmax(probas, dim=-1, keepdim=True)

        tokens = torch.cat((tokens, next_token), dim=1)
    return tokens_to_text(tokens, tokenizer)