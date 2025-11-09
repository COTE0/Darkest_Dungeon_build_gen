import torch
import torch.nn as nn
import csv
import os
from typing import Dict
from tqdm import tqdm
import torch.nn.functional as F
import sys
from model_core import (
    TOKEN_SPECIALS, tokenize_sequence, Vocab,
    Encoder, Decoder, create_decoding_mask,
    MODEL_CONFIG
)

sys.stdout = open("debug_log.txt", "w", encoding="utf-8")


K = MODEL_CONFIG['K_TOP']
TEMP = MODEL_CONFIG['TEMPERATURE']
MAX_LEN = MODEL_CONFIG['MAX_LEN']
EMB_SIZE = MODEL_CONFIG['EMB_SIZE']
HID_SIZE = MODEL_CONFIG['HID_SIZE']
N_LAYERS = MODEL_CONFIG['N_LAYERS']
DROPOUT = MODEL_CONFIG['DROPOUT']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    vocab = Vocab()
    vocab.__dict__.update(ckpt["vocab"])
    vsize = len(vocab)

    encoder = Encoder(vsize, EMB_SIZE, HID_SIZE, N_LAYERS, DROPOUT).to(device)
    decoder = Decoder(vsize, EMB_SIZE, HID_SIZE, N_LAYERS, DROPOUT).to(device)

    encoder.load_state_dict(ckpt["encoder"], strict=False)
    decoder.load_state_dict(ckpt["decoder"], strict=False)

    encoder.eval()
    decoder.eval()
    return encoder, decoder, vocab


@torch.no_grad()
def generate_sequence(input_str: str, encoder, decoder, vocab, device, debug=False):
    encoder.eval()
    decoder.eval()

    input_tokens = tokenize_sequence(input_str)
    if not input_tokens:
        return "<EMPTY>"

    inp_idx = [vocab.token_to_idx(t) for t in input_tokens]
    src = torch.tensor([inp_idx], dtype=torch.long, device=device)
    src_lens = torch.tensor([len(inp_idx)], dtype=torch.long, device=device)
    input_context = [vocab.idx_to_token(i) for i in src[0].tolist()]

    _, enc_hidden = encoder(src, src_lens)
    dec_hidden = enc_hidden

    generated_tokens = [TOKEN_SPECIALS['SOS']]
    inp_tok = torch.tensor([vocab.token_to_idx(TOKEN_SPECIALS['SOS'])], dtype=torch.long, device=device)
    used_global: Dict[str, int] = {}

    pad_idx = vocab.token_to_idx(TOKEN_SPECIALS['PAD'])
    unk_idx = vocab.token_to_idx(TOKEN_SPECIALS['UNK'])

    for t in range(MAX_LEN):
        logits, dec_hidden = decoder.forward_step(inp_tok, dec_hidden, t)
        if debug:
            print(f"\n🔹 STEP {t}")
            print(f"Prev token: {generated_tokens[-1]}")
        if t < 8:  
            eos_idx = vocab.token_to_idx(TOKEN_SPECIALS['EOS'])
            end_idx = vocab.token_to_idx(TOKEN_SPECIALS['END'])
            logits[0, eos_idx] -= 10.0
            logits[0, end_idx] -= 10.0
        mask = create_decoding_mask(
            generated_tokens[1:], vocab, device,
            input_context=input_context,
            used_global=used_global
        ).unsqueeze(0)

        mask[:, pad_idx] = False
        mask[:, unk_idx] = False

        masked_logits = logits.masked_fill(~mask, -float('inf'))
        allowed = [vocab.idx_to_token(i) for i in torch.where(mask[0])[0].tolist()]
        if debug:
            print(f"Allowed tokens: {allowed[:20]}{'...' if len(allowed) > 20 else ''}")
        logits_temp = masked_logits / TEMP

        k_val = min(K, logits_temp.size(-1))
        v, _ = torch.topk(logits_temp, k_val)
        logits_temp[logits_temp < v[:, [-1]]] = -float('inf')

        probs = F.softmax(logits_temp, dim=-1)
        topk_vals, topk_idx = torch.topk(probs, 5)
        topk_tokens = [vocab.idx_to_token(i) for i in topk_idx[0].tolist()]
        topk_probs = [round(p.item(), 4) for p in topk_vals[0]]
        if debug:
            print("Top-4 predictions:")
            for tok, prob in zip(topk_tokens, topk_probs):
                print(f"   {tok:<25} {prob}")
        if torch.isnan(probs).any() or probs.sum().item() == 0:
            next_idx = masked_logits.argmax(dim=-1)
        else:
            next_idx = torch.multinomial(probs, num_samples=1).squeeze(1)

        next_id = next_idx.item()
        next_token = vocab.idx_to_token(next_id)

        if next_token in (TOKEN_SPECIALS['EOS'], TOKEN_SPECIALS['END']):
            if debug:
                print(f"Early stop token: {next_token} (step={t})")
                print("\n"*3)
            break

        if next_id in (pad_idx, unk_idx):
            continue

        if next_token in vocab.dd_data.get('trinkets_all', []):
            used_global[next_token] = used_global.get(next_token, 0) + 1

        generated_tokens.append(next_token)
        inp_tok = next_idx

    tokens = [
        t for t in generated_tokens[1:]
        if t not in TOKEN_SPECIALS.values() or t == TOKEN_SPECIALS["EMPTY_TRINKET"]
    ]

    final_seq = "".join(tokens)

    final_seq = (
        final_seq.replace("<hero>", "")
                 .replace("<skill>", "")
                 .replace("<trinket>", "")
                 .replace("<pos>", "")
                 .replace("<end>", "")
    )

    import re
    final_seq = re.sub(r"(?<=\D)([1234])(?=:)", "", final_seq)

    final_seq = re.sub(r"\|{2,}", "|", final_seq)
    final_seq = re.sub(r"\+{2,}", "+", final_seq)
    final_seq = re.sub(r":{2,}", ":", final_seq)
    final_seq = re.sub(r"\|+", "|", final_seq)
    final_seq = re.sub(r"\++", "+", final_seq)
    final_seq = re.sub(r":+", ":", final_seq)

    final_seq = final_seq.strip()

    return final_seq or "<EMPTY>"



def main():
    model_path = "./data_scripts/seq2seq_checkpoint.pt"
    csv_path = "./data_scripts/teams_dataset.csv"
    out_csv = "./data_scripts/generated_output.csv"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at {model_path}")

    encoder, decoder, vocab = load_model(model_path, DEVICE)
    print(f"✅ Model loaded successfully (device: {DEVICE})")
    print(f"Sampling params → K={K}, T={TEMP}, MAX_LEN={MAX_LEN}")

    results = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for i, row in enumerate(tqdm(rows, desc="Generating")):
        if i >= 200:
            break

        inp = row.get("input_sequence", "").strip()
        if not inp:
            continue
        print(f'Row: {i}')
        gen = generate_sequence(inp, encoder, decoder, vocab, DEVICE, debug=True)
        if not gen.strip():
            gen = "<EMPTY>"

        results.append({
            "input_sequence": inp,
            "target_sequence": row.get("target_sequence", ""),
            "generated_sequence": gen,
            "length": len(gen.split("|"))
        })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input_sequence", "target_sequence", "generated_sequence", "length"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Generation complete. Saved to {out_csv} ({len(results)} rows).")


if __name__ == "__main__":
    main()
