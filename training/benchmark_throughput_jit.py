#!/usr/bin/env python3

"""

benchmark_throughput_jit.py — same model as benchmark_throughput.py, but the

GRU cell + timestep loop are compiled with torch.jit.script. This removes

Python interpreter overhead per timestep, which is the actual bottleneck at

~18,600 sequential steps/read (not FLOPs, not batch size, not thread count).



Usage:

    python3 benchmark_throughput_jit.py weights_v3_base_epoch14.pt paired_chr20_10k.txt --max-reads 10

"""



import argparse

import time

from typing import List, Tuple



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





class ScriptableGRU(nn.Module):

    """

    Same math as BaseConditionedGRUCell + the timestep loop in

    QualGRUBaseConditioned.forward, but written as a single TorchScript-

    compatible module so torch.jit.script can compile the WHOLE loop into

    one fused graph instead of re-entering Python for every timestep.

    """



    def __init__(self, qual_vocab_size: int, hidden_size: int,

                 qual_emb_dim: int = 32, base_emb_dim: int = 8):

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



    def forward(self, base_padded: torch.Tensor, qual_padded: torch.Tensor) -> torch.Tensor:

        B, T = qual_padded.shape

        h = torch.zeros(B, self.hidden_size, dtype=qual_padded.dtype if False else torch.float32,

                         device=qual_padded.device)

        # TorchScript requires a statically-typed loop; T-1 is a Tensor-derived

        # int, which is fine since qual_padded.shape is known at trace/script time

        for t in range(T - 1):

            qe = self.qual_emb(qual_padded[:, t])

            be_t = self.base_emb(base_padded[:, t])

            be_t1 = self.base_emb(base_padded[:, t + 1])

            x_t = torch.cat([qe, be_t, be_t1], dim=-1)



            r = torch.sigmoid(self.W_r(h) + self.U_r(x_t))

            u = torch.sigmoid(self.W_u(h) + self.U_u(x_t))

            c = torch.tanh(self.W_c(r * h) + self.U_c(x_t))

            h = (1 - u) * h + u * c

        return h





@torch.no_grad()

def run_benchmark(model, pairs, qual_stoi, batch_size, device, label):

    dataset = PairedReadDataset(pairs, qual_stoi)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)



    total_chars = 0

    t0 = time.time()

    for i, (base_padded, qual_padded, lengths) in enumerate(loader):

        base_padded = base_padded.to(device)

        qual_padded = qual_padded.to(device)

        model(base_padded, qual_padded)

        total_chars += lengths.sum().item()

        if i % 5 == 0:

            elapsed = time.time() - t0

            rate = total_chars / elapsed if elapsed > 0 else 0

            print(f"  [{label} bs={batch_size}] batch {i}  "

                  f"{total_chars:,} chars so far  "

                  f"({rate:,.0f} chars/sec running avg)", flush=True)

    dt = time.time() - t0



    chars_per_sec = total_chars / dt

    mb_per_sec = chars_per_sec / (1024 * 1024)

    print(f"[{label}] batch_size={batch_size}  "

          f"{total_chars:,} chars in {dt:.2f}s  "

          f"= {chars_per_sec:,.0f} chars/sec  = {mb_per_sec:.3f} MB/s")

    return chars_per_sec





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("checkpoint")

    parser.add_argument("data")

    parser.add_argument("--max-reads", type=int, default=10)

    parser.add_argument("--threads", type=int, default=1)

    args = parser.parse_args()



    torch.set_num_threads(args.threads)

    print(f"torch using {torch.get_num_threads()} CPU threads", flush=True)



    device = torch.device('cpu')

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    qual_vocab = ckpt['qual_vocab']

    qual_stoi = {c: i for i, c in enumerate(qual_vocab)}

    hidden_size = ckpt['hidden_size']



    # Build the scriptable model and load in the matching weights by name.

    # Parameter names match exactly (qual_emb, base_emb, U_r/U_u/U_c, W_r/W_u/W_c)

    # because ScriptableGRU mirrors BaseConditionedGRUCell's submodule names.

    model = ScriptableGRU(len(qual_vocab), hidden_size)

    cell_state = {k.replace('cell.', ''): v for k, v in ckpt['model'].items() if k.startswith('cell.')}

    missing, unexpected = model.load_state_dict(cell_state, strict=False)

    print(f"Loaded cell weights. missing={missing} unexpected={unexpected}", flush=True)

    model.eval()



    print("Compiling with torch.jit.script...", flush=True)

    t_compile = time.time()

    scripted_model = torch.jit.script(model)

    print(f"Compiled in {time.time() - t_compile:.2f}s", flush=True)



    pairs = load_paired(args.data, max_reads=args.max_reads)

    print(f"Loaded {len(pairs)} reads for benchmarking, hidden_size={hidden_size}", flush=True)



    for bs in [32, 128]:

        run_benchmark(scripted_model, pairs, qual_stoi, bs, device, f"JIT h={hidden_size}")





if __name__ == "__main__":

    main()
