#!/usr/bin/env python3

"""

rnn_ont_torch_v3_lossy.py — same base-conditioned GRU as rnn_ont_torch_v3.py,

with fixed-width quality-score binning applied as a preprocessing step.



This is standard lossy quality-score compression: quality values are

quantized into N bins BEFORE the model ever sees them. The model trains on,

and predicts, bin representative values only — never the original raw

quality value. On decompression, you get back the bin representative (e.g.

6, 15, 22, 27, 32, 37, ...), not the original exact score. This matches how

ENANO/CoLoRd/Illumina's built-in lossy modes work: the loss happens once, at

quantization time, and is fully reversible from that point on (the pipeline

after quantization is lossless w.r.t. the BINNED values).



Binning scheme: fixed-width, 8 bins by default, computed over the observed

Q-value range in the training data. Bin edges are equal-width; each bin's

representative is the (rounded) bin midpoint. This mirrors the standard

approach used as a baseline across ENANO/CoLoRd/SPRING lossy comparisons

(those tools bin, then run their existing lossless codec on the binned

stream — same idea here, just with the GRU as the lossless-stage predictor).



Everything else (model architecture, training loop, base-conditioning,

held-out split) is UNCHANGED from rnn_ont_torch_v3.py. Only load_paired()

gained a binning step; the rest just adapts automatically to the smaller

resulting vocab size.



Usage:

    python3 rnn_ont_torch_v3_lossy.py paired_chr20_10k.txt --n-bins 8 \

        --hidden-size 256 --epochs 15 --lr 3e-4 --save weights_v3_lossy8

"""



import argparse

import bisect

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

# Binning

# --------------------------------------------------------------------------



def compute_fixed_bins(all_qual_chars, n_bins=8):

    """

    Fixed-width binning over the observed Phred+33 Q-value range.

    Returns (bin_edges, bin_reps):

      bin_edges: n_bins-1 interior boundaries (Q-value space), used with

                 bisect to assign a raw Q value to a bin index.

      bin_reps:  n_bins representative Q values (one per bin), each the

                 rounded midpoint of that bin's [low, high) range.

    """

    q_values = [ord(c) - 33 for c in all_qual_chars]

    qmin, qmax = min(q_values), max(q_values)

    width = (qmax - qmin) / n_bins



    bin_edges = [qmin + width * i for i in range(1, n_bins)]  # interior edges

    bin_reps = []

    for i in range(n_bins):

        lo = qmin + width * i

        hi = qmin + width * (i + 1)

        bin_reps.append(round((lo + hi) / 2))



    return bin_edges, bin_reps





def bin_char(ch, bin_edges, bin_reps):

    """Map one raw Phred+33 quality char to its bin's representative char."""

    q = ord(ch) - 33

    idx = bisect.bisect_right(bin_edges, q)  # which bin q falls into

    rep_q = bin_reps[idx]

    return chr(rep_q + 33)





def bin_quality_string(quals, bin_edges, bin_reps):

    return ''.join(bin_char(c, bin_edges, bin_reps) for c in quals)





# --------------------------------------------------------------------------

# Data

# --------------------------------------------------------------------------



class PairedReadDataset(Dataset):

    def __init__(self, pairs, qual_stoi):

        self.samples = []

        for bases, quals in pairs:

            qual_ids = [qual_stoi[ch] for ch in quals if ch in qual_stoi]

            if len(qual_ids) != len(quals) or len(qual_ids) < 2:

                continue

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





def load_paired(path, n_bins=8):

    raw_pairs = []

    with open(path) as f:

        for line in f:

            line = line.rstrip('\n\r')

            if not line or '\t' not in line:

                continue

            bases, quals = line.split('\t', 1)

            if len(bases) == len(quals) and len(bases) >= 2:

                raw_pairs.append((bases, quals))



    all_qual_chars = ''.join(q for _, q in raw_pairs)

    bin_edges, bin_reps = compute_fixed_bins(all_qual_chars, n_bins=n_bins)

    print(f"Computed {n_bins} fixed-width bins.")

    print(f"  Bin edges (Q-value space): {bin_edges}")

    print(f"  Bin representatives (Q-value space): {bin_reps}")



    # Apply binning to every read's quality string BEFORE building vocab.

    pairs = [(bases, bin_quality_string(quals, bin_edges, bin_reps)) for bases, quals in raw_pairs]



    all_binned_chars = ''.join(q for _, q in pairs)

    qual_vocab = sorted(set(all_binned_chars))

    qual_stoi = {c: i for i, c in enumerate(qual_vocab)}



    return pairs, qual_vocab, qual_stoi, bin_edges, bin_reps





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

# Model (identical to rnn_ont_torch_v3.py — unchanged)

# --------------------------------------------------------------------------



class BaseConditionedGRUCell(nn.Module):

    def __init__(self, qual_vocab_size, hidden_size, qual_emb_dim=32, base_emb_dim=8):

        super().__init__()

        self.hidden_size = hidden_size

        self.qual_emb = nn.Embedding(qual_vocab_size, qual_emb_dim)

        self.base_emb = nn.Embedding(len(BASE_VOCAB), base_emb_dim)

        input_dim = qual_emb_dim + 2 * base_emb_dim



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



    pairs, qual_vocab, qual_stoi, bin_edges, bin_reps = load_paired(args.data, n_bins=args.n_bins)

    print(f"Built quality vocab of size {len(qual_vocab)} (post-binning): {''.join(qual_vocab)}")

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

                'n_bins': args.n_bins,

                'bin_edges': bin_edges,

                'bin_reps': bin_reps,

            }, ckpt_path)

            print(f"checkpoint saved to {ckpt_path}", flush=True)





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Lossy (fixed-width binned) base-conditioned GRU for ONT quality-score compression.")

    parser.add_argument("data", help="Paired data file (tab-separated bases\\tqualities, one read per line)")

    parser.add_argument("--n-bins", type=int, default=8, help="Number of fixed-width quality bins (default: 8)")

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
