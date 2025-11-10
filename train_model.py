from functools import partial
import os
import random
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.nn.functional as F
from model_core import (
    TOKEN_SPECIALS, TeamDataset, collate_batch,
    Encoder, Decoder, create_decoding_mask, tokenize_sequence,
    MODEL_CONFIG
)

SEED = 69
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(SEED)

NUM_WORKERS = 4
BATCH_SIZE = 64
EMB_SIZE = MODEL_CONFIG['EMB_SIZE']
HID_SIZE = MODEL_CONFIG['HID_SIZE']
N_LAYERS = MODEL_CONFIG['N_LAYERS']
DROPOUT = MODEL_CONFIG['DROPOUT']

LR = 3e-4
MAX_EPOCHS = 30
K = MODEL_CONFIG['K_TOP']
TEMP = MODEL_CONFIG['TEMPERATURE']
MAX_LEN = MODEL_CONFIG['MAX_LEN']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURRENT_EPOCH = 1
torch.backends.cudnn.benchmark = True

def masked_cross_entropy(logits, targets, allowed_mask, pad_idx):
    logits = torch.clamp(logits, -20.0, 20.0)
    masked_logits = logits.masked_fill(~allowed_mask, -20.0)
    log_probs = torch.nn.functional.log_softmax(masked_logits, dim=-1)
    nll = -log_probs[torch.arange(targets.size(0), device=logits.device), targets]
    nll = nll.masked_fill(targets == pad_idx, 0.0)
    valid = (targets != pad_idx)
    return nll[valid].mean() if valid.any() else torch.tensor(0.0, device=logits.device)

def precompute_masks_for_batch(tgt, src, vocab, device):
    batch_masks = []
    pad_idx = vocab.token_to_idx(TOKEN_SPECIALS['PAD'])
    batch_size, T = tgt.size()

    special_mask = torch.zeros((len(vocab),), dtype=torch.bool, device=device)
    special_tokens_allowed = [
        TOKEN_SPECIALS['SOS'], TOKEN_SPECIALS['EOS'],
        TOKEN_SPECIALS['HERO'], TOKEN_SPECIALS['POS'],
        TOKEN_SPECIALS['SKILL'], TOKEN_SPECIALS['TRINKET'],
        TOKEN_SPECIALS['END'], ':', '+', '|'
    ]
    for tkn in special_tokens_allowed:
        if tkn in vocab.token2idx:
            special_mask[vocab.token_to_idx(tkn)] = True


    for i in range(batch_size):
        seq_idxs = tgt[i].tolist()
        seq_tokens = []
        for idx in seq_idxs:
            if idx == pad_idx:
                break
            seq_tokens.append(vocab.idx_to_token(idx))

        src_tokens = []
        for idx in src[i].tolist():
            if idx == pad_idx:
                break
            src_tokens.append(vocab.idx_to_token(idx))

        masks_seq = []
        context = []
        used_global = {}

        for t in range(1, len(seq_tokens)):
            if seq_tokens[t] in TOKEN_SPECIALS.values():
                masks_seq.append(special_mask)
            else:
                mask = create_decoding_mask(
                    context, vocab, device,
                    input_context=src_tokens,
                    used_global=used_global
                )
                mask[vocab.token_to_idx(TOKEN_SPECIALS['PAD'])] = False
                mask[vocab.token_to_idx(TOKEN_SPECIALS['UNK'])] = False
                masks_seq.append(mask)

            current_token = seq_tokens[t]
            if current_token in vocab.dd_data.get('trinkets_all', []):
                used_global[current_token] = used_global.get(current_token, 0) + 1

            context.append(current_token)

        while len(masks_seq) < (T - 1):
            masks_seq.append(special_mask)

        batch_masks.append(masks_seq[:(T - 1)])

    if not batch_masks:
        return torch.empty((0, 0, len(vocab)), dtype=torch.bool, device=device)

    stacked = torch.stack([torch.stack(step) for step in zip(*batch_masks)], dim=0)
    return stacked


@torch.no_grad()
def generate_one_sampled_build(encoder, decoder, input_str, vocab, device):
    encoder.eval(); decoder.eval()    
    input_tokens = tokenize_sequence(input_str)
    inp_idx = [vocab.token_to_idx(t) for t in input_tokens]
    src = torch.tensor([inp_idx], dtype=torch.long, device=device)
    src_lens = torch.tensor([len(inp_idx)], dtype=torch.long, device=device)
    input_context = [vocab.idx_to_token(i) for i in src[0].tolist()]

    _, enc_hidden = encoder(src, src_lens)
    dec_hidden = enc_hidden
    generated_tokens = [TOKEN_SPECIALS['SOS']]
    inp_tok = torch.tensor([vocab.token_to_idx(TOKEN_SPECIALS['SOS'])], dtype=torch.long, device=device)
    used_global = {}

    pad_idx = vocab.token_to_idx(TOKEN_SPECIALS['PAD'])
    unk_idx = vocab.token_to_idx(TOKEN_SPECIALS['UNK'])
    
    K_DEFAULT = K
    TEMP_DEFAULT = TEMP
    
    K_FIRST_HERO = len(vocab.hero_names)
    TEMP_FIRST_HERO = 1.5
    
    log_buffer = "--- DZIENNIK GENEROWANIA PRÓBKOWANEGO ---\n"
    log_buffer += f"Ustawienia domyślne: T={TEMP_DEFAULT}, K={K_DEFAULT}\n"

    for t in range(MAX_LEN):
        log_buffer += f"\n--- KROK {t+1} | OSTATNI TOKEN: '{generated_tokens[-1]}' ---\n"
        
        logits, dec_hidden = decoder.forward_step(inp_tok, dec_hidden, t)

        allowed_mask = create_decoding_mask(
            generated_tokens[1:], vocab, device,
            input_context=input_context,
            used_global=used_global
        ).unsqueeze(0)
        allowed_mask[:, pad_idx] = False
        allowed_mask[:, unk_idx] = False

        current_k = K_DEFAULT
        current_temp = TEMP_DEFAULT
        
        if len(generated_tokens) == 2 and generated_tokens[1] == TOKEN_SPECIALS['HERO']:
            current_k = K_FIRST_HERO
            current_temp = TEMP_FIRST_HERO
            log_buffer += f"ZASADA: WYBÓR PIERWSZEGO BOHATERA. UŻYTE: T={current_temp}, K={current_k}\n"

        masked_logits = logits.masked_fill(~allowed_mask, -float('inf'))
        masked_logits[:, pad_idx] = -float('inf')
        masked_logits[:, unk_idx] = -float('inf')

        logits_temp = masked_logits / current_temp
        
        k_val = min(current_k, logits_temp.size(-1)) 
        v, _ = torch.topk(logits_temp, k_val)
        logits_temp[logits_temp < v[:, [-1]]] = -float('inf')

        probs = F.softmax(logits_temp, dim=-1)
        
        max_log_k = 10 
        log_k = min(max_log_k, k_val)

        top_indices_tensor, top_probs_tensor = torch.topk(probs, log_k)

        top_indices = top_indices_tensor.squeeze(0).cpu().numpy()
        top_probs = top_probs_tensor.squeeze(0).cpu().numpy()
        
        log_buffer += "TOKENY DO WYBORU (Top {} po Softmax):\n".format(len(top_indices))
        for idx, prob in zip(top_indices, top_probs):
            log_buffer += f"  - {vocab.idx_to_token(int(idx)):<20} P={prob:.4f}\n"

        if torch.isnan(probs).any() or probs.sum().item() == 0:
            next_token_idx = masked_logits.argmax(dim=-1)
            log_buffer += "⚠️ Uwaga: Błąd w prawdopodobieństwach, wybrano Greedy.\n"
        else:
            next_token_idx = torch.multinomial(probs, num_samples=1).squeeze(1)

        cand = next_token_idx.item()
        if cand == pad_idx or cand == unk_idx:
            next_token_idx = masked_logits.argmax(dim=-1)
            cand = next_token_idx.item()

        next_token = vocab.idx_to_token(cand)
        log_buffer += f"✅ WYBRANY TOKEN: '{next_token}'\n"

        if next_token in [TOKEN_SPECIALS['EOS'], TOKEN_SPECIALS['END']]:
            break

        if next_token in vocab.dd_data.get('trinkets_all', []):
            used_global[next_token] = used_global.get(next_token, 0) + 1
        
        if "<trinket>" in generated_tokens or any(t in vocab.dd_data.get('trinkets_all', []) for t in generated_tokens):
            if not hasattr(generate_one_sampled_build, "_local_trinkets"):
                generate_one_sampled_build._local_trinkets = set()
            if next_token in vocab.dd_data.get('trinkets_all', []):
                generate_one_sampled_build._local_trinkets.add(next_token)
        if next_token == "|":
            if hasattr(generate_one_sampled_build, "_local_trinkets"):
                del generate_one_sampled_build._local_trinkets


        generated_tokens.append(next_token)
        inp_tok = next_token_idx

    final_seq = "".join(
        [t for t in generated_tokens[1:] if t not in TOKEN_SPECIALS.values()]
    ).strip()
    return final_seq, log_buffer


def train_epoch(encoder, decoder, dataloader, optimizer, vocab, device, scaler, teacher_forcing=1):
    encoder.train(); decoder.train()
    pad_idx = vocab.token_to_idx(TOKEN_SPECIALS['PAD'])
    total_loss = 0.0
    loop = tqdm(dataloader, desc="Training", leave=False)
    
    tf_decay_rate = (CURRENT_EPOCH - 1) / 12 if CURRENT_EPOCH <= 12 else 1.0
    current_tf = max(0.0, teacher_forcing - tf_decay_rate)

    for batch in loop:
        src = batch['input'].to(device, non_blocking=True)
        src_lens = batch['input_lens'].to(device, non_blocking=True)
        tgt = batch['target'].to(device, non_blocking=True)
        if src.size(0) == 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        masks_stacked = precompute_masks_for_batch(tgt, src, vocab, device)

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            _, enc_hidden = encoder(src, src_lens)
            dec_hidden = enc_hidden
            inp_tok = tgt[:, 0]
            loss_sum = 0.0

            for t in range(1, tgt.size(1)):
                logits, dec_hidden = decoder.forward_step(inp_tok, dec_hidden, t)
                if t - 1 >= masks_stacked.size(0):
                    break
                allowed_mask = masks_stacked[t - 1]
                step_loss = masked_cross_entropy(logits, tgt[:, t], allowed_mask, pad_idx)
                loss_sum += step_loss
                
                inp_tok = tgt[:, t] if random.random() < current_tf else logits.argmax(dim=-1).detach()

            loss = loss_sum / max(1, tgt.size(1) - 1)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 0.9)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * src.size(0)
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader.dataset)

@torch.no_grad()
def evaluate(encoder, decoder, dataloader, vocab, device):
    encoder.eval(); decoder.eval()
    pad_idx = vocab.token_to_idx(TOKEN_SPECIALS['PAD'])
    total_loss = 0.0
    loop = tqdm(dataloader, desc="Validation", leave=False)

    for batch in loop:
        src = batch['input'].to(device, non_blocking=True)
        src_lens = batch['input_lens'].to(device, non_blocking=True)
        tgt = batch['target'].to(device, non_blocking=True)
        if src.size(0) == 0:
            continue

        masks_stacked = precompute_masks_for_batch(tgt, src, vocab, device)
        _, enc_hidden = encoder(src, src_lens)
        dec_hidden = enc_hidden
        inp_tok = tgt[:, 0]
        loss_sum = 0.0

        for t in range(1, tgt.size(1)):
            logits, dec_hidden = decoder.forward_step(inp_tok, dec_hidden, t)
            if t - 1 >= masks_stacked.size(0):
                break
            allowed_mask = masks_stacked[t - 1]
            loss_sum += masked_cross_entropy(logits, tgt[:, t], allowed_mask, pad_idx)
            inp_tok = tgt[:, t]

        loss = loss_sum / max(1, tgt.size(1) - 1)
        total_loss += loss.item() * src.size(0)
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader.dataset)

def main():
    csv_path = "./data_scripts/teams_dataset_with_specials.csv"
    out_path = "./data_scripts/seq2seq_checkpoint.pt"

    print(f"Using device: {DEVICE}")
    dataset = TeamDataset(csv_path)
    N = len(dataset)
    indices = list(range(N))
    random.shuffle(indices)
    split = int(N * 0.9)
    train_ds = torch.utils.data.Subset(dataset, indices[:split])
    test_ds = torch.utils.data.Subset(dataset, indices[split:])

    collate_fn = partial(collate_batch, vocab=dataset.vocab, device=DEVICE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

    encoder = Encoder(len(dataset.vocab), EMB_SIZE, HID_SIZE, DROPOUT).to(DEVICE)
    decoder = Decoder(len(dataset.vocab), EMB_SIZE, HID_SIZE, DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=LR, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=1)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    start_epoch = 1
    best_val = float("inf")
    if os.path.exists(out_path):
        print(f"Loading checkpoint from {out_path}")
        ckpt = torch.load(out_path, map_location=DEVICE)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resuming from epoch {start_epoch}")
    global CURRENT_EPOCH 
    CURRENT_EPOCH = start_epoch

    patience, no_improve = 3, 0
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        train_loss = train_epoch(encoder, decoder, train_loader, optimizer, dataset.vocab, DEVICE, scaler)
        CURRENT_EPOCH += 1
        val_loss = evaluate(encoder, decoder, test_loader, dataset.vocab, DEVICE)
        scheduler.step(val_loss)

        random_idx = random.choice(indices[split:])
        random_input = dataset.rows[random_idx]['input']
        random_target = dataset.rows[random_idx]['target']
        
        sampled_build, log_buffer = generate_one_sampled_build(encoder, decoder, random_input, dataset.vocab, DEVICE)
        
        # Zapis logu do pliku
        log_file_path = f"generation_log_epoch_{epoch:02d}.txt"
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write(log_buffer)

        print(f"\nEpoch {epoch:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6g}")
        print(f"--- LOSOWY WYLOSOWANY BUILD (Top-k={K}, T={TEMP}) ---")
        print(f"Szczegółowy log generacji zapisano do: {log_file_path}")
        print(f"INPUT: {random_input}")
        print(f"TARGET: {random_target}")
        print(f"FINAL OUTPUT: {sampled_build}")
        print("-------------------------------------------------")

        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            torch.save({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "vocab": dataset.vocab.__dict__,
                "optimizer": optimizer.state_dict(),
            }, out_path)
            print(f"✅ Saved checkpoint → {out_path}")
        else:
            no_improve += 1
            if no_improve > patience:
                print("Early stopping triggered.")
                break

    print(f"Training finished. Best val loss: {best_val:.4f}")

if __name__ == "__main__":
    main()