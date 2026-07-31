#!/usr/bin/env python3

"""

rnn_ont_torch_v3.py — quality-score GRU conditioned on base identity.



Rationale: scaling hidden size (64->256) and adding local attention both

plateaued around ~3.88-3.91 bpc on held-out data. That's evidence the

bottleneck isn't model capacity or context window — it's that the model

only ever sees the quality-score stream in isolation. ONT quality scores

are known to correlate strongly with the underlying base call and local

sequence context (e.g. homopolymer runs tend to have systematically lower

confidence). This version gives the model that signal directly.



At each step, predicting quality_{t+1} is conditioned on:

  - quality_t              (as before)

  - base_t                 (the base aligned with quality_t)

  - base_{t+1}              (the base aligned with the quality being predicted

                             — legitimate to use, since in a real compression

                             pipeline the base sequence is decoded/known

                             BEFORE quality scores are compressed)



Input format: paired_data.txt, one read per line, tab-separated:

    <bases>\t<qualities>

Use extract_paired.py to build this from a FASTQ or BAM.



Held-out split logic (read-level, fixed seed) is unchanged from v2, so

runs are directly comparable to the earlier hidden-size / attention

experiments as long as the read counts/order are similar (they may not

line up 1:1 since paired extraction re-parses from FASTQ/BAM rather than

the plain quality-only file, so treat this as a fresh baseline rather

than a strict apples-to-apples re-run).

"""



import argparse

import math

import random

import time



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader



BASE_VOCAB = ['A', 'C', 'G', 'T', 'N']

BASE_STOI = {b: i for i, b in enumerate(BASE_VOCAB)}

BASE_UNK = BASE_STOI['N']





# --------------------------------------------------------------------------

# Data

# --------------------------------------------------------------------------



class PairedReadDataset(Dataset):

    def __init__(self, pairs, qual_stoi):

        self.samples = []

        for bases, quals in pairs:

            qual_ids = [qual_stoi[ch] for ch in quals if ch in qual_stoi]

            if len(qual_ids) != len(quals) or len(qual_ids) < 2:

                continue  # drop reads with any out-of-vocab quality char

            base_ids = [BASE_STOI.get(b, BASE_UNK) for b in bases]

            self.samples.append((

                torch.tensor(base_ids, dtype=torch.long),

                torch.tensor(qual_ids, dtype=torch.long),

            ))



    def __len__(self):

        return len(self.samples)



    def __getitem__(self, idx):

        return self.samples[idx]





def collate_pad(batch):

    bases, quals = zip(*batch)

    lengths = torch.tensor([len(x) for x in quals], dtype=torch.long)

    base_padded = nn.utils.rnn.pad_sequence(bases, batch_first=True, padding_value=BASE_UNK)

    qual_padded = nn.utils.rnn.pad_sequence(quals, batch_first=True, padding_value=0)

    return base_padded, qual_padded, lengths





def load_paired(path):

    pairs = []

    with open(path) as f:

        for line in f:

            line = line.rstrip('\n\r')

            if not line or '\t' not in line:

                continue

            bases, quals = line.split('\t', 1)

            if len(bases) == len(quals) and len(bases) >= 2:

                pairs.append((bases, quals))



    all_qual_chars = ''.join(q for _, q in pairs)

    qual_vocab = sorted(set(all_qual_chars))

    qual_stoi = {c: i for i, c in enumerate(qual_vocab)}

    return pairs, qual_vocab, qual_stoi





def split_pairs(pairs, val_frac, seed):

    idx = list(range(len(pairs)))

    rng = random.Random(seed)

    rng.shuffle(idx)

    n_val = max(1, int(len(idx) * val_frac))

    val_idx = set(idx[:n_val])

    train_pairs = [p for i, p in enumerate(pairs) if i not in val_idx]

    val_pairs = [p for i, p in enumerate(pairs) if i in val_idx]

    return train_pairs, val_pairs





def chunk_pairs(pairs, chunk_len):

    if not chunk_len:

        return pairs

    chunked = []

    for bases, quals in pairs:

        n = len(quals)

        if n <= chunk_len:

            chunked.append((bases, quals))

        else:

            for i in range(0, n - 1, chunk_len):

                b_piece = bases[i:i + chunk_len + 1]

                q_piece = quals[i:i + chunk_len + 1]

                if len(q_piece) >= 2:

                    chunked.append((b_piece, q_piece))

    return chunked





# --------------------------------------------------------------------------

# Model

# --------------------------------------------------------------------------



class BaseConditionedGRUCell(nn.Module):

    """

    Same gate structure as before, but the input at each step is the

    concatenation of [quality_t embedding, base_t embedding,

    base_{t+1} embedding], projected through a Linear layer instead of a

    raw one-hot lookup (since it's now a multi-part input, not one symbol).

    """



    def __init__(self, qual_vocab_size, hidden_size, qual_emb_dim=32, base_emb_dim=8):

        super().__init__()

        self.hidden_size = hidden_size



        self.qual_emb = nn.Embedding(qual_vocab_size, qual_emb_dim)

        self.base_emb = nn.Embedding(len(BASE_VOCAB), base_emb_dim)

        input_dim = qual_emb_dim + 2 * base_emb_dim  # qual_t, base_t, base_{t+1}



        self.U_r = nn.Linear(input_dim, hidden_size, bias=False)

        self.U_u = nn.Linear(input_dim, hidden_size, bias=False)

        self.U_c = nn.Linear(input_dim, hidden_size, bias=False)



        self.W_r = nn.Linear(hidden_size, hidden_size, bias=True)

        self.W_u = nn.Linear(hidden_size, hidden_size, bias=True)

        self.W_c = nn.Linear(hidden_size, hidden_size, bias=True)



        scale = math.sqrt(2.0 / (hidden_size + hidden_size))

        for lin in (self.W_r, self.W_u, self.W_c):

            nn.init.normal_(lin.weight, std=scale)

            nn.init.zeros_(lin.bias)

        in_scale = math.sqrt(2.0 / (hidden_size + input_dim))

        for lin in (self.U_r, self.U_u, self.U_c):

            nn.init.normal_(lin.weight, std=in_scale)



    def encode_input(self, qual_t, base_t, base_t1):

        qe = self.qual_emb(qual_t)

        be_t = self.base_emb(base_t)

        be_t1 = self.base_emb(base_t1)

        return torch.cat([qe, be_t, be_t1], dim=-1)



    def forward(self, x_t, h_prev):

        r = torch.sigmoid(self.W_r(h_prev) + self.U_r(x_t))

        u = torch.sigmoid(self.W_u(h_prev) + self.U_u(x_t))

        c = torch.tanh(self.W_c(r * h_prev) + self.U_c(x_t))

        return (1 - u) * h_prev + u * c





class QualGRUBaseConditioned(nn.Module):

    def __init__(self, qual_vocab_size, hidden_size, qual_emb_dim=32, base_emb_dim=8):

        super().__init__()

        self.hidden_size = hidden_size

        self.cell = BaseConditionedGRUCell(qual_vocab_size, hidden_size, qual_emb_dim, base_emb_dim)

        self.out_proj = nn.Linear(hidden_size, qual_vocab_size)



    def forward(self, base_padded, qual_padded, lengths):

        """

        base_padded, qual_padded: (B, T)

        Predicts quality_{t+1} from quality_t, base_t, base_{t+1}, and h_prev.

        Returns logits: (B, T-1, V)

        """

        B, T = qual_padded.shape

        device = qual_padded.device

        h = torch.zeros(B, self.hidden_size, device=device)



        hidden_states = []

        for t in range(T - 1):

            x_t = self.cell.encode_input(qual_padded[:, t], base_padded[:, t], base_padded[:, t + 1])

            h = self.cell(x_t, h)

            hidden_states.append(h)

        h_seq = torch.stack(hidden_states, dim=1)

        return self.out_proj(h_seq)





# --------------------------------------------------------------------------

# Train / Eval

# --------------------------------------------------------------------------



def masked_cross_entropy(logits, targets, lengths):

    B, Tm1, V = logits.shape

    device = logits.device

    pos = torch.arange(Tm1, device=device).unsqueeze(0)

    mask = pos < (lengths.unsqueeze(1) - 1)

    loss_flat = F.cross_entropy(

        logits.reshape(-1, V), targets.reshape(-1), reduction='none'

    ).view(B, Tm1)

    loss = (loss_flat * mask).sum() / mask.sum().clamp(min=1)

    return loss, mask.sum()





@torch.no_grad()

def evaluate(model, loader, device):

    model.eval()

    total_loss, total_tokens = 0.0, 0

    for base_padded, qual_padded, lengths in loader:

        base_padded = base_padded.to(device, non_blocking=True)

        qual_padded = qual_padded.to(device, non_blocking=True)

        lengths = lengths.to(device, non_blocking=True)

        targets = qual_padded[:, 1:]

        logits = model(base_padded, qual_padded, lengths)

        loss, n_tok = masked_cross_entropy(logits, targets, lengths)

        total_loss += loss.item() * n_tok.item()

        total_tokens += n_tok.item()

    model.train()

    avg_loss = total_loss / max(total_tokens, 1)

    return avg_loss, avg_loss / math.log(2)





def train(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}", flush=True)



    pairs, qual_vocab, qual_stoi = load_paired(args.data)

    print(f"Built quality vocab of size {len(qual_vocab)}: {''.join(qual_vocab)}")

    print(f"Loaded {len(pairs)} paired reads", flush=True)



    train_pairs, val_pairs = split_pairs(pairs, args.val_frac, args.split_seed)

    print(f"Split: {len(train_pairs)} train, {len(val_pairs)} val "

          f"(val_frac={args.val_frac}, seed={args.split_seed})", flush=True)



    train_chunks = chunk_pairs(train_pairs, args.seq_len)

    val_chunks = chunk_pairs(val_pairs, args.seq_len)

    print(f"Train chunks: {len(train_chunks)}  Val chunks: {len(val_chunks)}", flush=True)



    train_loader = DataLoader(

        PairedReadDataset(train_chunks, qual_stoi), batch_size=args.batch_size, shuffle=True,

        collate_fn=collate_pad, num_workers=2, pin_memory=(device.type == 'cuda'),

    )

    val_loader = DataLoader(

        PairedReadDataset(val_chunks, qual_stoi), batch_size=args.batch_size, shuffle=False,

        collate_fn=collate_pad, num_workers=2, pin_memory=(device.type == 'cuda'),

    )



    model = QualGRUBaseConditioned(len(qual_vocab), args.hidden_size).to(device)



    if args.load:

        state = torch.load(args.load, map_location=device)

        model.load_state_dict(state['model'])

        print(f"Loaded weights from {args.load}")



    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())

    print(f"Model has {n_params:,} parameters", flush=True)



    best_val_bpc = float('inf')



    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):

        model.train()

        total_loss, total_tokens = 0.0, 0

        t0 = time.time()

        for step, (base_padded, qual_padded, lengths) in enumerate(train_loader):

            base_padded = base_padded.to(device, non_blocking=True)

            qual_padded = qual_padded.to(device, non_blocking=True)

            lengths = lengths.to(device, non_blocking=True)

            targets = qual_padded[:, 1:]



            logits = model(base_padded, qual_padded, lengths)

            loss, n_tok = masked_cross_entropy(logits, targets, lengths)



            opt.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            opt.step()



            total_loss += loss.item() * n_tok.item()

            total_tokens += n_tok.item()



            if step % 200 == 0:

                bpc = loss.item() / math.log(2)

                print(f"epoch {epoch}  step {step}/{len(train_loader)}  "

                      f"loss {loss.item():.4f}  bpc {bpc:.4f}", flush=True)



        train_avg_loss = total_loss / max(total_tokens, 1)

        train_avg_bpc = train_avg_loss / math.log(2)

        val_avg_loss, val_avg_bpc = evaluate(model, val_loader, device)

        dt = time.time() - t0



        marker = ""

        if val_avg_bpc < best_val_bpc:

            best_val_bpc = val_avg_bpc

            marker = "  <-- best so far"

        print(f"epoch {epoch} done  train_bpc {train_avg_bpc:.4f}  "

              f"val_bpc {val_avg_bpc:.4f}  ({dt:.1f}s){marker}", flush=True)



        if args.save:

            ckpt_path = f"{args.save}_epoch{epoch}.pt"

            torch.save({

                'model': model.state_dict(),

                'qual_vocab': qual_vocab,

                'hidden_size': args.hidden_size,

                'epoch': epoch,

                'val_bpc': val_avg_bpc,

            }, ckpt_path)

            print(f"checkpoint saved to {ckpt_path}", flush=True)





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Base-conditioned GRU for ONT quality-score compression.")

    parser.add_argument("data", help="Paired data file (tab-separated bases\\tqualities, one read per line)")

    parser.add_argument("--hidden-size", type=int, default=256)

    parser.add_argument("--seq-len", type=int, default=200)

    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--epochs", type=int, default=15)

    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--val-frac", type=float, default=0.1)

    parser.add_argument("--split-seed", type=int, default=42)

    parser.add_argument("--save", type=str, default=None)

    parser.add_argument("--load", type=str, default=None)

    parser.add_argument("--start-epoch", type=int, default=0)

    args = parser.parse_args()



    if args.seq_len == 0:

        args.seq_len = None



    train(args)
