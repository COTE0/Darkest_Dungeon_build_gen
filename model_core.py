from typing import List, Dict, Optional, Set
import re
import yaml
import csv
from collections import Counter
import torch
from torch import nn
from torch.utils.data import Dataset
from functools import partial


MODEL_CONFIG = {
    'EMB_SIZE': 128,
    'HID_SIZE': 768,
    'N_LAYERS': 2,
    'DROPOUT': 0.25,
    'K_TOP': 4,
    'TEMPERATURE': 1.1,
    'MAX_LEN': 90,
}

TOKEN_SPECIALS = {
    'PAD': '<pad>',
    'SOS': '<sos>',
    'EOS': '<eos>',
    'UNK': '<unk>',
    'HERO': '<hero>',
    'POS': '<pos>',
    'SKILL': '<skill>',
    'TRINKET': '<trinket>',
    'EMPTY_TRINKET': '<empty_trinket>',
    'END': '<end>'
}

TOKEN_PATTERN = re.compile(r"<hero>|<pos>|<skill>|<trinket>|<end>|<sos>|<eos>|<pad>|<unk>|[A-Za-z0-9_À-ÖØ-öø-ÿ' ]+|[:\+\|]")
def tokenize_sequence(seq: str) -> List[str]:
    seq = seq.strip()
    return TOKEN_PATTERN.findall(seq) if seq else []


class Vocab:
    def __init__(self, min_freq: int = 1, dd_data_path: Optional[str] = "data_scripts/dd_hero_data.yml"):
        self.token2idx = {}
        self.idx2token = []
        self.freqs = Counter()
        self.min_freq = min_freq
        self._freeze = False
        self.dd_data: Dict = {}
        self.hero_names: List[str] = []
        for t in TOKEN_SPECIALS.values():
            self.add_token(t, count=0)
        if dd_data_path:
            self.load_dd_data(dd_data_path)

    def load_dd_data(self, dd_data_path: str):
        try:
            with open(dd_data_path, 'r', encoding='utf-8') as f:
                self.dd_data = yaml.safe_load(f)
                self.hero_names = sorted(list(self.dd_data['heroes'].keys()))
                
                for hero in self.hero_names:
                    self.add_token(hero, count=0)
                    for skill in self.dd_data['heroes'][hero].get('skills', []):
                        self.add_token(skill, count=0)
                    for trinket in self.dd_data['heroes'][hero].get('trinkets', []):
                        self.add_token(trinket, count=0)

                for trinket in self.dd_data.get('trinkets_all', []):
                    self.add_token(trinket, count=0)

                print(f"Loaded DD data from {dd_data_path}. Heroes: {len(self.hero_names)}")
        except FileNotFoundError:
            print(f"Warning: DD data file not found at {dd_data_path}. Dynamic masking will not work.")
            self.dd_data = {'heroes': {}, 'trinkets_all': []}
            self.hero_names = []
        except Exception as e:
            print(f"Error loading DD data: {e}. Dynamic masking will not work.")
            self.dd_data = {'heroes': {}, 'trinkets_all': []}
            self.hero_names = []

    def add_token(self, token: str, count: int = 1):
        if self._freeze:
            return
        if token not in self.token2idx:
            self.token2idx[token] = len(self.idx2token)
            self.idx2token.append(token)
        self.freqs[token] += count

    def build_from_sequences(self, sequences: List[List[str]]):
        for seq in sequences:
            for tok in seq:
                self.add_token(tok)
        
        items = [t for t, c in self.freqs.items() if c >= self.min_freq or t in TOKEN_SPECIALS.values() or t in self.hero_names or t in self.dd_data.get('trinkets_all', [])]
        
        self.token2idx, self.idx2token = {}, []
        for t in TOKEN_SPECIALS.values():
            self.token2idx[t] = len(self.idx2token)
            self.idx2token.append(t)
        
        for t in sorted(set(items) - set(TOKEN_SPECIALS.values())):
            self.token2idx[t] = len(self.idx2token)
            self.idx2token.append(t)
        
        self._freeze = True

    def __len__(self):
        return len(self.idx2token)

    def token_to_idx(self, token: str):
        return self.token2idx.get(token, self.token2idx[TOKEN_SPECIALS['UNK']])

    def idx_to_token(self, idx: int):
        return self.idx2token[idx]


def create_decoding_mask(prev_tokens: List[str], vocab: Vocab, device: torch.device,
                         input_context: List[str], used_global: Dict[str, int]) -> torch.Tensor:
    vsize = len(vocab)
    mask = torch.zeros((vsize,), dtype=torch.bool, device=device)

    if not vocab.dd_data:
        mask[:] = True
        return mask

    last_token = prev_tokens[-1] if prev_tokens else TOKEN_SPECIALS["SOS"]
    hero_names = vocab.hero_names
    all_trinkets = set(vocab.dd_data.get("trinkets_all", []))

    used_heroes = set()
    current_block = []
    last_hero_idx = None

    for i, tok in enumerate(prev_tokens):
        if tok == TOKEN_SPECIALS["HERO"]:
            last_hero_idx = i
        elif tok == "|" and last_hero_idx is not None:
            hero_in_block = next((t for t in prev_tokens[last_hero_idx + 1:i] if t in hero_names), None)
            if hero_in_block:
                used_heroes.add(hero_in_block)
            last_hero_idx = None

    if last_hero_idx is not None:
        current_block = prev_tokens[last_hero_idx + 1:]
        hero_name = next((t for t in current_block if t in hero_names), None)
    else:
        current_block = []
        hero_name = None

    hero_data = vocab.dd_data["heroes"].get(hero_name, {"skills": [], "trinkets": []})
    skill_count = sum(1 for t in current_block if t in hero_data["skills"])
    trinket_count = sum(1 for t in current_block if t in all_trinkets or t in hero_data["trinkets"])
    used_local_skills = set(t for t in current_block if t in hero_data["skills"])
    used_local_trinkets = set(t for t in current_block if t in all_trinkets or t in hero_data["trinkets"])

    def get_available_trinkets(allow_empty=True):
        hero_trinkets = set(hero_data.get("trinkets", []))
        possible = hero_trinkets.union(all_trinkets)
        available_in_input = [t for t in input_context if t in possible]

        available = []
        for trink in available_in_input:
            max_allowed = input_context.count(trink)
            if trink not in used_local_trinkets and used_global.get(trink, 0) < max_allowed:
                available.append(trink)

        if not available and allow_empty:
            return [TOKEN_SPECIALS["EMPTY_TRINKET"]]

        return list(set(available))

    allowed_tokens = []

    if last_token in [TOKEN_SPECIALS["SOS"], "|"]:
        if len(used_heroes) >= 4:
            allowed_tokens = [TOKEN_SPECIALS["END"], TOKEN_SPECIALS["EOS"]]
        else:
            allowed_tokens = [TOKEN_SPECIALS["HERO"]]

    elif last_token == TOKEN_SPECIALS["HERO"]:
        available = [h for h in input_context if h in hero_names and h not in used_heroes]
        allowed_tokens = available

    elif last_token in hero_names:
        allowed_tokens = [TOKEN_SPECIALS["POS"]]

    elif last_token == TOKEN_SPECIALS["POS"]:
        allowed_tokens = ['1', '2', '3', '4']

    elif last_token in ['1', '2', '3', '4']:
        allowed_tokens = [":"]

    elif last_token == ":":
        allowed_tokens = [TOKEN_SPECIALS["SKILL"]]

    elif last_token == TOKEN_SPECIALS["SKILL"]:
        allowed_tokens = list(set(hero_data.get("skills", [])) - used_local_skills)

    elif last_token in hero_data.get("skills", []):
        if skill_count < 4:
            allowed_tokens = ["+"]
        elif skill_count == 4:
            allowed_tokens = ["+"]

    elif last_token == "+":
        if skill_count < 4:
            allowed_tokens = list(set(hero_data.get("skills", [])) - used_local_skills)
        elif skill_count == 4 and trinket_count == 0:
            allowed_tokens = [TOKEN_SPECIALS["TRINKET"]]
        elif skill_count == 4 and trinket_count == 1:
            allowed_tokens = get_available_trinkets()

    elif last_token == TOKEN_SPECIALS["TRINKET"]:
        allowed_tokens = get_available_trinkets(allow_empty=True)

    elif last_token == TOKEN_SPECIALS["EMPTY_TRINKET"]:
        if len(used_heroes) < 3:
            allowed_tokens = ["|"]
        else:
            allowed_tokens = [TOKEN_SPECIALS["END"], TOKEN_SPECIALS["EOS"]]

    elif last_token in all_trinkets or last_token in hero_data.get("trinkets", []):
        if trinket_count == 1:
            allowed_tokens = ["+"]
        elif trinket_count == 2:
            if len(used_heroes) < 3:
                allowed_tokens = ["|"]
            else:
                allowed_tokens = [TOKEN_SPECIALS["END"], TOKEN_SPECIALS["EOS"]]

    for token in allowed_tokens:
        if token in vocab.token2idx:
            mask[vocab.token_to_idx(token)] = True

    mask[vocab.token_to_idx(TOKEN_SPECIALS["UNK"])] = False
    mask[vocab.token_to_idx(TOKEN_SPECIALS["PAD"])] = False
    return mask

class TeamDataset(Dataset):
    def __init__(self, csv_path: str, vocab: Vocab = None):
        self.rows = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                inp = r.get('input_sequence', '').strip()
                tgt = r.get('target_sequence', '').strip()
                if inp and tgt:
                    self.rows.append({'input': inp, 'target': tgt})

        self.inputs_tok = [tokenize_sequence(r['input']) for r in self.rows]
        self.targets_tok = [tokenize_sequence(r['target']) for r in self.rows]

        if vocab is None:
            self.vocab = Vocab(min_freq=1) 
            self.vocab.build_from_sequences(self.inputs_tok + self.targets_tok)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return {'input_tokens': self.inputs_tok[idx], 'target_tokens': self.targets_tok[idx]}


def collate_batch(batch: List[Dict], vocab: Vocab, device=None):
    PAD = vocab.token_to_idx(TOKEN_SPECIALS['PAD'])
    batch_inp = [b['input_tokens'] for b in batch]
    batch_tgt = [b['target_tokens'] for b in batch]

    inp_idx = [[vocab.token_to_idx(t) for t in seq] for seq in batch_inp]
    tgt_idx = [[vocab.token_to_idx(TOKEN_SPECIALS['SOS'])] +
               [vocab.token_to_idx(t) for t in seq] +
               [vocab.token_to_idx(TOKEN_SPECIALS['EOS'])] for seq in batch_tgt]

    max_inp = max(len(x) for x in inp_idx) if inp_idx else 0
    max_tgt = max(len(x) for x in tgt_idx) if tgt_idx else 0
    
    if max_inp == 0 or max_tgt == 0:
        return {
            'input': torch.empty((0, 0), dtype=torch.long),
            'input_lens': torch.empty((0,), dtype=torch.long),
            'target': torch.empty((0, 0), dtype=torch.long),
            'target_lens': torch.empty((0,), dtype=torch.long),
        }

    inp_tensor = torch.full((len(batch), max_inp), PAD, dtype=torch.long)
    tgt_tensor = torch.full((len(batch), max_tgt), PAD, dtype=torch.long)

    for i, seq in enumerate(inp_idx):
        inp_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    for i, seq in enumerate(tgt_idx):
        tgt_tensor[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        
    return {
        'input': inp_tensor, 
        'input_lens': torch.tensor([len(x) for x in inp_idx], dtype=torch.long),
        'target': tgt_tensor,
        'target_lens': torch.tensor([len(x) for x in tgt_idx], dtype=torch.long),
    }

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, dropout, n_layers = MODEL_CONFIG.get('N_LAYERS')):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, hid_size, num_layers=n_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.hid_proj = nn.Linear(hid_size * 2, hid_size)

    def forward(self, src, src_lens):
        emb = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        h = h.view(self.lstm.num_layers, 2, h.size(1), h.size(2))
        c = c.view(self.lstm.num_layers, 2, c.size(1), c.size(2))
        last_h = torch.cat([h[-1, 0], h[-1, 1]], dim=-1)
        last_c = torch.cat([c[-1, 0], c[-1, 1]], dim=-1)
        projected_h = torch.tanh(self.hid_proj(last_h))
        projected_c = torch.tanh(self.hid_proj(last_c))
        dec_h = projected_h.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        dec_c = projected_c.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        return out, (dec_h, dec_c)

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, dropout, n_layers = MODEL_CONFIG.get('N_LAYERS'), max_len=MODEL_CONFIG['MAX_LEN']):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, emb_size)
        self.lstm = nn.LSTM(emb_size, hid_size, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.out = nn.Linear(hid_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(self, input_tok, hidden, step_idx):
        tok_emb = self.embedding(input_tok.unsqueeze(1))
        pos_emb = self.pos_embedding(torch.tensor([step_idx], device=tok_emb.device)).unsqueeze(0)
        emb = self.dropout(tok_emb + pos_emb)
        out, hidden = self.lstm(emb, hidden)
        logits = self.out(out.squeeze(1))
        return logits, hidden