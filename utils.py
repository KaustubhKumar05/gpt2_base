import tiktoken
import torch
import numpy as np
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


def get_completion(model, text, device, max_new_tokens = 25, ctx_size = GPT2_124M_Config["ctx_length"], tokenizer = tiktoken.get_encoding("gpt2"), temperature = 2):
    tokens = tokenize(text, tokenizer).to(device)
    for _ in range(max_new_tokens):
        truncated_tokens = tokens[:, -ctx_size:]

        with torch.no_grad():
            logits = model(truncated_tokens)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probas, num_samples=1)

        if next_token.item() == tokenizer.eot_token:
            break

        yield tokens_to_text(next_token, tokenizer)
        tokens = torch.cat((tokens, next_token), dim=1)

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch!\nLeft: {left.shape}\nRight: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right, device=left.device))


def load_weights_into_gpt(gpt, params):
    gpt.pos_embed.weight = assign(gpt.pos_embed.weight, params["wpe"])
    gpt.tok_embed.weight = assign(gpt.tok_embed.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].attn.Wq.weight = assign(
            gpt.trf_blocks[b].attn.Wq.weight, q_w.T)
        gpt.trf_blocks[b].attn.Wk.weight = assign(
            gpt.trf_blocks[b].attn.Wk.weight, k_w.T)
        gpt.trf_blocks[b].attn.Wv.weight = assign(
            gpt.trf_blocks[b].attn.Wv.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].attn.Wq.bias = assign(
            gpt.trf_blocks[b].attn.Wq.bias, q_b)
        gpt.trf_blocks[b].attn.Wk.bias = assign(
            gpt.trf_blocks[b].attn.Wk.bias, k_b)
        gpt.trf_blocks[b].attn.Wv.bias = assign(
            gpt.trf_blocks[b].attn.Wv.bias, v_b)

        gpt.trf_blocks[b].attn.out_proj.weight = assign(
            gpt.trf_blocks[b].attn.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].attn.out_proj.bias = assign(
            gpt.trf_blocks[b].attn.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].ln1.scale = assign(
            gpt.trf_blocks[b].ln1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].ln1.shift = assign(
            gpt.trf_blocks[b].ln1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].ln2.scale = assign(
            gpt.trf_blocks[b].ln2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].ln2.shift = assign(
            gpt.trf_blocks[b].ln2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])