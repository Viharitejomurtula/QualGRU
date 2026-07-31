#!/usr/bin/env python3

"""

benchmark_throughput.py — measure CPU inference throughput for a saved

QualGRU (v3, base-conditioned) checkpoint, in characters/sec and MB/s.



Runs two modes:

  1. Unbatched (batch_size=1) — the naive per-read loop

  2. Batched — many reads processed simultaneously, to show the effect of

     amortizing Python/dispatch overhead across a batch



Usage:

    python3 benchmark_throughput.py weights_v3_base_epoch14.pt paired_chr20_10k.txt

"""



import argparse

import time



import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader



BASE_VOCAB = ['A', 'C', 'G', 'T', 'N']

BASE_STOI = {b: i for i, b in enumerate(BASE_VOCAB)}

BASE_UNK = BASE_STOI['N']





class PairedReadDataset(Dataset):

    def __init__(self, pairs, qual_stoi, max_reads=None):

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

            if max_reads and len(self.samples) >= max_reads:

                break



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





def load_paired(path, max_reads=2000):

    pairs = []

    with open(path) as f:

        for line in f:

            line = line.rstrip('\n\r')

            if not line or '\t' not in line:

                continue

            bases, quals = line.split('\t', 1)

            if len(bases) == len(quals) and len(bases) >= 2:

                pairs.append((bases, quals))

            if len(pairs) >= max_reads:

                break

    return pairs





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

        for t in range(T - 1):

            x_t = self.cell.encode_input(qual_padded[:, t], base_padded[:, t], base_padded[:, t + 1])

            h = self.cell(x_t, h)

        return None  # throughput bench doesn't need logits, just timing





@torch.no_grad()

def run_benchmark(model, pairs, qual_stoi, batch_size, device, label):

    dataset = PairedReadDataset(pairs, qual_stoi)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)



    total_chars = 0

    t0 = time.time()

    for base_padded, qual_padded, lengths in loader:

        base_padded = base_padded.to(device)

        qual_padded = qual_padded.to(device)

        lengths = lengths.to(device)

        model(base_padded, qual_padded, lengths)

        total_chars += lengths.sum().item()

    dt = time.time() - t0



    chars_per_sec = total_chars / dt

    mb_per_sec = chars_per_sec / (1024 * 1024)  # 1 char = 1 byte for quality scores

    print(f"[{label}] batch_size={batch_size}  "

          f"{total_chars:,} chars in {dt:.2f}s  "

          f"= {chars_per_sec:,.0f} chars/sec  = {mb_per_sec:.3f} MB/s")

    return chars_per_sec





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("checkpoint")

    parser.add_argument("data")

    parser.add_argument("--max-reads", type=int, default=2000)

    args = parser.parse_args()



    torch.set_num_threads(torch.get_num_threads())  # uses all available by default

    print(f"torch using {torch.get_num_threads()} CPU threads", flush=True)



    device = torch.device('cpu')

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    qual_vocab = ckpt['qual_vocab']

    qual_stoi = {c: i for i, c in enumerate(qual_vocab)}

    hidden_size = ckpt['hidden_size']



    model = QualGRUBaseConditioned(len(qual_vocab), hidden_size).to(device)

    model.load_state_dict(ckpt['model'])

    model.eval()



    pairs = load_paired(args.data, max_reads=args.max_reads)

    print(f"Loaded {len(pairs)} reads for benchmarking, hidden_size={hidden_size}", flush=True)



    for bs in [32, 128]:

        run_benchmark(model, pairs, qual_stoi, bs, device, f"h={hidden_size}")





if __name__ == "__main__":

    main()
