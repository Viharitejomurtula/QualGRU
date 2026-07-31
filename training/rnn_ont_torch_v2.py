#!/usr/bin/env python3

"""

rnn_ont_torch_v2.py — same model as rnn_ont_torch.py, plus a proper

read-level held-out validation split.



Why read-level, not chunk-level: chunks from the same original read share

strong local structure. If we split at the chunk level, some chunks from a

read could land in train and others (immediately adjacent) in val, which

leaks information and inflates val performance. Splitting whole reads first,

then chunking each split independently, avoids that leakage.



New flags:

  --val-frac   fraction of reads held out for validation (default 0.1)

  --split-seed seed for the train/val split (default 42, keep fixed across

               runs so different hyperparameter runs are compared on the

               SAME held-out set)



Each epoch now prints both train bpc (over batches seen during training,

same as before) and val bpc (full pass over the held-out set, no grad,

model.eval()). val bpc is the number that actually matters for deciding

whether more capacity/attention helps.

"""



import argparse

import math

import random

import time



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader





# --------------------------------------------------------------------------

# Data

# --------------------------------------------------------------------------



class ReadDataset(Dataset):

    def __init__(self, reads, stoi):

        self.samples = []

        for read in reads:

            ids = [stoi[ch] for ch in read if ch in stoi]

            if len(ids) >= 2:

                self.samples.append(torch.tensor(ids, dtype=torch.long))



    def __len__(self):

        return len(self.samples)



    def __getitem__(self, idx):

        return self.samples[idx]





def collate_pad(batch):

    lengths = torch.tensor([len(x) for x in batch], dtype=torch.long)

    padded = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)

    return padded, lengths





def load_raw_reads(path):

    with open(path) as f:

        lines = f.readlines()

    raw_reads = [line.rstrip('\n\r') for line in lines if line.strip()]



    all_chars = ''.join(raw_reads)

    vocab = sorted(set(all_chars))

    stoi = {c: i for i, c in enumerate(vocab)}

    itos = {i: c for i, c in enumerate(vocab)}



    reads = [[ch for ch in r if ch in stoi] for r in raw_reads]

    reads = [r for r in reads if len(r) >= 2]

    return reads, vocab, stoi, itos





def split_reads(reads, val_frac, seed):

    idx = list(range(len(reads)))

    rng = random.Random(seed)

    rng.shuffle(idx)

    n_val = max(1, int(len(idx) * val_frac))

    val_idx = set(idx[:n_val])

    train_reads = [r for i, r in enumerate(reads) if i not in val_idx]

    val_reads = [r for i, r in enumerate(reads) if i in val_idx]

    return train_reads, val_reads





def chunk_reads(reads, chunk_len):

    if not chunk_len:

        return reads

    chunked = []

    for r in reads:

        if len(r) <= chunk_len:

            chunked.append(r)

        else:

            for i in range(0, len(r) - 1, chunk_len):

                piece = r[i:i + chunk_len + 1]

                if len(piece) >= 2:

                    chunked.append(piece)

    return chunked





# --------------------------------------------------------------------------

# Model (identical to rnn_ont_torch.py)

# --------------------------------------------------------------------------



class ManualGRUCell(nn.Module):

    def __init__(self, vocab_size, hidden_size):

        super().__init__()

        self.hidden_size = hidden_size

        self.U_r = nn.Embedding(vocab_size, hidden_size)

        self.U_u = nn.Embedding(vocab_size, hidden_size)

        self.U_c = nn.Embedding(vocab_size, hidden_size)



        self.W_r = nn.Linear(hidden_size, hidden_size, bias=True)

        self.W_u = nn.Linear(hidden_size, hidden_size, bias=True)

        self.W_c = nn.Linear(hidden_size, hidden_size, bias=True)



        scale = math.sqrt(2.0 / (hidden_size + hidden_size))

        for lin in (self.W_r, self.W_u, self.W_c):

            nn.init.normal_(lin.weight, std=scale)

            nn.init.zeros_(lin.bias)

        emb_scale = math.sqrt(2.0 / (hidden_size + vocab_size))

        for emb in (self.U_r, self.U_u, self.U_c):

            nn.init.normal_(emb.weight, std=emb_scale)



    def forward(self, x_t, h_prev):

        ux_r, ux_u, ux_c = self.U_r(x_t), self.U_u(x_t), self.U_c(x_t)

        r = torch.sigmoid(self.W_r(h_prev) + ux_r)

        u = torch.sigmoid(self.W_u(h_prev) + ux_u)

        c = torch.tanh(self.W_c(r * h_prev) + ux_c)

        h_new = (1 - u) * h_prev + u * c

        return h_new





class LocalAttention(nn.Module):

    def __init__(self, hidden_size, window, n_heads=4):

        super().__init__()

        assert hidden_size % n_heads == 0

        self.window = window

        self.n_heads = n_heads

        self.head_dim = hidden_size // n_heads

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

        self.out = nn.Linear(hidden_size, hidden_size, bias=False)



    def forward(self, h_seq):

        B, T, H = h_seq.shape

        qkv = self.qkv(h_seq).view(B, T, 3, self.n_heads, self.head_dim)

        q, k, v = qkv.unbind(dim=2)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))



        idx = torch.arange(T, device=h_seq.device)

        rel = idx[:, None] - idx[None, :]

        mask = (rel >= 0) & (rel <= self.window)



        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        attn = attn.masked_fill(~mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)



        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, T, H)

        return self.out(out)





class QualGRU(nn.Module):

    def __init__(self, vocab_size, hidden_size, attn_window=0, attn_heads=4):

        super().__init__()

        self.hidden_size = hidden_size

        self.cell = ManualGRUCell(vocab_size, hidden_size)

        self.attn = LocalAttention(hidden_size, attn_window, attn_heads) if attn_window > 0 else None

        self.out_proj = nn.Linear(hidden_size, vocab_size)



    def forward(self, x_padded, lengths):

        B, T = x_padded.shape

        device = x_padded.device

        h = torch.zeros(B, self.hidden_size, device=device)



        hidden_states = []

        for t in range(T - 1):

            h = self.cell(x_padded[:, t], h)

            hidden_states.append(h)

        h_seq = torch.stack(hidden_states, dim=1)



        if self.attn is not None:

            h_seq = h_seq + self.attn(h_seq)



        logits = self.out_proj(h_seq)

        return logits





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

    for x_padded, lengths in loader:

        x_padded = x_padded.to(device, non_blocking=True)

        lengths = lengths.to(device, non_blocking=True)

        targets = x_padded[:, 1:]

        logits = model(x_padded, lengths)

        loss, n_tok = masked_cross_entropy(logits, targets, lengths)

        total_loss += loss.item() * n_tok.item()

        total_tokens += n_tok.item()

    model.train()

    avg_loss = total_loss / max(total_tokens, 1)

    return avg_loss, avg_loss / math.log(2)





def train(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}", flush=True)



    raw_reads, vocab, stoi, itos = load_raw_reads(args.data)

    vocab_size = len(vocab)

    print(f"Built vocab of size {vocab_size}: {''.join(vocab)}")

    print(f"Loaded {len(raw_reads)} raw reads", flush=True)



    train_reads, val_reads = split_reads(raw_reads, args.val_frac, args.split_seed)

    print(f"Split: {len(train_reads)} train reads, {len(val_reads)} val reads "

          f"(val_frac={args.val_frac}, seed={args.split_seed})", flush=True)



    train_chunks = chunk_reads(train_reads, args.seq_len)

    val_chunks = chunk_reads(val_reads, args.seq_len)

    print(f"Train chunks: {len(train_chunks)}  Val chunks: {len(val_chunks)}", flush=True)



    train_loader = DataLoader(

        ReadDataset(train_chunks, stoi), batch_size=args.batch_size, shuffle=True,

        collate_fn=collate_pad, num_workers=2, pin_memory=(device.type == 'cuda'),

    )

    val_loader = DataLoader(

        ReadDataset(val_chunks, stoi), batch_size=args.batch_size, shuffle=False,

        collate_fn=collate_pad, num_workers=2, pin_memory=(device.type == 'cuda'),

    )



    model = QualGRU(

        vocab_size, args.hidden_size,

        attn_window=args.attn_window, attn_heads=args.attn_heads,

    ).to(device)



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

        for step, (x_padded, lengths) in enumerate(train_loader):

            x_padded = x_padded.to(device, non_blocking=True)

            lengths = lengths.to(device, non_blocking=True)

            targets = x_padded[:, 1:]



            logits = model(x_padded, lengths)

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

                'vocab': vocab,

                'hidden_size': args.hidden_size,

                'attn_window': args.attn_window,

                'attn_heads': args.attn_heads,

                'epoch': epoch,

                'val_bpc': val_avg_bpc,

            }, ckpt_path)

            print(f"checkpoint saved to {ckpt_path}", flush=True)





if __name__ == "__main__":

    parser = argparse.ArgumentParser(

        description="GPU-accelerated GRU training with held-out validation."

    )

    parser.add_argument("data", help="Path to file with one read per line")

    parser.add_argument("--hidden-size", type=int, default=256)

    parser.add_argument("--seq-len", type=int, default=200)

    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--attn-window", type=int, default=0)

    parser.add_argument("--attn-heads", type=int, default=4)

    parser.add_argument("--val-frac", type=float, default=0.1)

    parser.add_argument("--split-seed", type=int, default=42)

    parser.add_argument("--save", type=str, default=None)

    parser.add_argument("--load", type=str, default=None)

    parser.add_argument("--start-epoch", type=int, default=0)

    args = parser.parse_args()



    if args.seq_len == 0:

        args.seq_len = None



    train(args)
