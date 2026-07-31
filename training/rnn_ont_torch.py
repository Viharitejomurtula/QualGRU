#!/usr/bin/env python3

"""

PyTorch port of rnn_ont.py — GPU-accelerated GRU for ONT quality-score compression.



Key differences from the NumPy version:

  - Trains on batches of reads in parallel (padded + masked) instead of one

    read at a time. This is what actually lets the A100 help you — a single

    read at a time is too small to saturate a GPU.

  - Autograd replaces the manual BPTT block. The custom GRU cell below

    reproduces your exact equations (r * h_prev BEFORE the W_c matmul, not

    PyTorch's built-in nn.GRUCell convention, which differs), so bpc numbers

    stay comparable to your existing NumPy benchmarks.

  - Optional local (windowed, causal) self-attention layer on top of the GRU

    hidden states, per your Phase 1 roadmap note (hidden 256-512 + local

    attention). Off by default; turn on with --attn-window > 0.

  - Read-boundary resets are preserved: hidden state resets to zero at the

    start of every read, same as build_read_chunks()/train() in the original.



Checkpoints are NOT binary-compatible with the .npz files from rnn_ont.py

(different tensor layout/library), but the loss curve and eval harness

(bits-per-char) are apples-to-apples comparable.

"""



import argparse

import math

import time



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader





# --------------------------------------------------------------------------

# Data

# --------------------------------------------------------------------------



class ReadDataset(Dataset):

    """Each item is one full read, encoded as a LongTensor of token ids."""



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

    """Pad a batch of variable-length reads. Returns (padded, lengths)."""

    lengths = torch.tensor([len(x) for x in batch], dtype=torch.long)

    padded = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)

    return padded, lengths





def load_reads(path, chunk_len=None):

    with open(path) as f:

        lines = f.readlines()

    raw_reads = [line.rstrip('\n\r') for line in lines if line.strip()]



    all_chars = ''.join(raw_reads)

    vocab = sorted(set(all_chars))

    stoi = {c: i for i, c in enumerate(vocab)}

    itos = {i: c for i, c in enumerate(vocab)}



    reads = [[ch for ch in r if ch in stoi] for r in raw_reads]

    reads = [r for r in reads if len(r) >= 2]



    # Optionally split long reads into fixed-length chunks, same spirit as

    # build_read_chunks() in the original — keeps sequence lengths bounded

    # so padding waste stays low and BPTT-through-time stays cheap.

    if chunk_len:

        chunked = []

        for r in reads:

            if len(r) <= chunk_len:

                chunked.append(r)

            else:

                for i in range(0, len(r) - 1, chunk_len):

                    piece = r[i:i + chunk_len + 1]

                    if len(piece) >= 2:

                        chunked.append(piece)

        reads = chunked



    return reads, vocab, stoi, itos





# --------------------------------------------------------------------------

# Model

# --------------------------------------------------------------------------



class ManualGRUCell(nn.Module):

    """

    Reproduces rnn_ont.py's exact gate equations:

      r = sigmoid(W_r h + U_r x + b_r)

      u = sigmoid(W_u h + U_u x + b_u)

      c = tanh(W_c (r * h) + U_c x + b_c)      <-- r applied BEFORE W_c matmul

      h' = (1 - u) * h + u * c

    """



    def __init__(self, vocab_size, hidden_size):

        super().__init__()

        self.hidden_size = hidden_size

        # nn.Embedding rows == columns of U_* in the original (one-hot @ U == row lookup)

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

        # x_t: (B,) long token ids ; h_prev: (B, H)

        ux_r, ux_u, ux_c = self.U_r(x_t), self.U_u(x_t), self.U_c(x_t)

        r = torch.sigmoid(self.W_r(h_prev) + ux_r)

        u = torch.sigmoid(self.W_u(h_prev) + ux_u)

        c = torch.tanh(self.W_c(r * h_prev) + ux_c)

        h_new = (1 - u) * h_prev + u * c

        return h_new





class LocalAttention(nn.Module):

    """

    Causal windowed self-attention over the GRU's hidden-state sequence.

    Each position attends only to itself and the previous `window` steps —

    cheap (O(n * window)) and matches the "local attention" note in the

    Phase 1 roadmap, as opposed to full O(n^2) attention.

    """



    def __init__(self, hidden_size, window, n_heads=4):

        super().__init__()

        assert hidden_size % n_heads == 0, "hidden_size must be divisible by n_heads"

        self.window = window

        self.n_heads = n_heads

        self.head_dim = hidden_size // n_heads

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

        self.out = nn.Linear(hidden_size, hidden_size, bias=False)



    def forward(self, h_seq):

        # h_seq: (B, T, H)

        B, T, H = h_seq.shape

        qkv = self.qkv(h_seq).view(B, T, 3, self.n_heads, self.head_dim)

        q, k, v = qkv.unbind(dim=2)  # each (B, T, n_heads, head_dim)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, n_heads, T, head_dim)



        # Causal + local-window mask: position i can see [i-window, i]

        idx = torch.arange(T, device=h_seq.device)

        rel = idx[None, :] - idx[:, None]  # (T, T): key_pos - query_pos... careful with sign

        rel = idx[:, None] - idx[None, :]  # query - key

        mask = (rel >= 0) & (rel <= self.window)  # (T, T) bool, True = allowed



        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,heads,T,T)

        attn = attn.masked_fill(~mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)



        out = torch.matmul(attn, v)  # (B, heads, T, head_dim)

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

        """

        x_padded: (B, T) long token ids (padded with 0)

        lengths:  (B,) actual read lengths

        Returns logits for predicting x[:, 1:] from x[:, :-1]: (B, T-1, V)

        """

        B, T = x_padded.shape

        device = x_padded.device

        h = torch.zeros(B, self.hidden_size, device=device)



        hidden_states = []

        for t in range(T - 1):

            h = self.cell(x_padded[:, t], h)

            hidden_states.append(h)

        h_seq = torch.stack(hidden_states, dim=1)  # (B, T-1, H)



        if self.attn is not None:

            h_seq = h_seq + self.attn(h_seq)  # residual local attention



        logits = self.out_proj(h_seq)  # (B, T-1, V)

        return logits





# --------------------------------------------------------------------------

# Training

# --------------------------------------------------------------------------



def masked_cross_entropy(logits, targets, lengths):

    """logits: (B,T-1,V), targets: (B,T-1) shifted-by-one token ids, lengths: (B,)"""

    B, Tm1, V = logits.shape

    device = logits.device

    pos = torch.arange(Tm1, device=device).unsqueeze(0)  # (1, T-1)

    # a prediction at position t is valid if t < length-1 (need a target at t+1)

    mask = pos < (lengths.unsqueeze(1) - 1)  # (B, T-1)



    loss_flat = F.cross_entropy(

        logits.reshape(-1, V), targets.reshape(-1), reduction='none'

    ).view(B, Tm1)

    loss = (loss_flat * mask).sum() / mask.sum().clamp(min=1)

    return loss, mask.sum()





def train(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}", flush=True)



    reads, vocab, stoi, itos = load_reads(args.data, chunk_len=args.seq_len)

    vocab_size = len(vocab)

    print(f"Built vocab of size {vocab_size}: {''.join(vocab)}")

    print(f"Loaded {len(reads)} reads/chunks", flush=True)



    dataset = ReadDataset(reads, stoi)

    loader = DataLoader(

        dataset, batch_size=args.batch_size, shuffle=True,

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



    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):

        model.train()

        total_loss, total_tokens = 0.0, 0

        t0 = time.time()

        for step, (x_padded, lengths) in enumerate(loader):

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

                print(f"epoch {epoch}  step {step}/{len(loader)}  "

                      f"loss {loss.item():.4f}  bpc {bpc:.4f}", flush=True)



        avg_loss = total_loss / max(total_tokens, 1)

        avg_bpc = avg_loss / math.log(2)

        dt = time.time() - t0

        print(f"epoch {epoch} done  avg loss {avg_loss:.4f}  "

              f"avg bpc {avg_bpc:.4f}  ({dt:.1f}s)", flush=True)



        if args.save:

            ckpt_path = f"{args.save}_epoch{epoch}.pt"

            torch.save({

                'model': model.state_dict(),

                'vocab': vocab,

                'hidden_size': args.hidden_size,

                'attn_window': args.attn_window,

                'attn_heads': args.attn_heads,

                'epoch': epoch,

            }, ckpt_path)

            print(f"checkpoint saved to {ckpt_path}", flush=True)





if __name__ == "__main__":

    parser = argparse.ArgumentParser(

        description="GPU-accelerated GRU training for ONT quality-score compression."

    )

    parser.add_argument("data", help="Path to file with one read (quality string) per line")

    parser.add_argument("--hidden-size", type=int, default=256)

    parser.add_argument("--seq-len", type=int, default=200,

                         help="Chunk length for splitting long reads (default: 200; "

                              "use 0/None to disable chunking)")

    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--attn-window", type=int, default=0,

                         help="Local causal attention window size; 0 disables attention")

    parser.add_argument("--attn-heads", type=int, default=4)

    parser.add_argument("--save", type=str, default=None, help="Checkpoint path prefix")

    parser.add_argument("--load", type=str, default=None, help="Checkpoint .pt to resume from")

    parser.add_argument("--start-epoch", type=int, default=0)

    args = parser.parse_args()



    if args.seq_len == 0:

        args.seq_len = None



    train(args)
