#!/usr/bin/env python3

"""

benchmark_throughput_mp.py — shard reads across N worker processes, each

running the JIT-compiled GRU single-threaded. Throughput here should scale

close to linearly with process count (up to core count), since each read's

sequential chain is independent of every other read's.



Usage:

    python3 benchmark_throughput_mp.py weights_v3_base_epoch14.pt paired_chr20_10k.txt \

        --max-reads 200 --num-workers 32

"""



import argparse

import time

from multiprocessing import Pool



import torch

import torch.nn as nn



BASE_VOCAB = ['A', 'C', 'G', 'T', 'N']

BASE_STOI = {b: i for i, b in enumerate(BASE_VOCAB)}

BASE_UNK = BASE_STOI['N']



_worker_model = None

_worker_qual_stoi = None





class ScriptableGRU(nn.Module):

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

        h = torch.zeros(B, self.hidden_size, dtype=torch.float32, device=qual_padded.device)

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





def load_paired(path, max_reads):

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





def _init_worker(checkpoint_path):

    """Runs once per worker process: pin to 1 thread, load+script the model."""

    global _worker_model, _worker_qual_stoi

    torch.set_num_threads(1)



    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    qual_vocab = ckpt['qual_vocab']

    _worker_qual_stoi = {c: i for i, c in enumerate(qual_vocab)}

    hidden_size = ckpt['hidden_size']



    model = ScriptableGRU(len(qual_vocab), hidden_size)

    cell_state = {k.replace('cell.', ''): v for k, v in ckpt['model'].items() if k.startswith('cell.')}

    model.load_state_dict(cell_state, strict=False)

    model.eval()

    _worker_model = torch.jit.script(model)





@torch.no_grad()

def _process_read(pair):

    """Runs ONE read through the model in this worker; returns char count."""

    bases, quals = pair

    qual_ids = [_worker_qual_stoi[ch] for ch in quals if ch in _worker_qual_stoi]

    if len(qual_ids) != len(quals) or len(qual_ids) < 2:

        return 0

    base_ids = [BASE_STOI.get(b, BASE_UNK) for b in bases]



    qual_t = torch.tensor([qual_ids], dtype=torch.long)   # (1, T)

    base_t = torch.tensor([base_ids], dtype=torch.long)   # (1, T)

    _worker_model(base_t, qual_t)

    return len(quals)





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("checkpoint")

    parser.add_argument("data")

    parser.add_argument("--max-reads", type=int, default=200)

    parser.add_argument("--num-workers", type=int, default=32)

    args = parser.parse_args()



    pairs = load_paired(args.data, args.max_reads)

    total_chars_expected = sum(len(q) for _, q in pairs)

    print(f"Loaded {len(pairs)} reads, {total_chars_expected:,} total chars", flush=True)

    print(f"Spawning {args.num_workers} worker processes...", flush=True)



    t0 = time.time()

    with Pool(processes=args.num_workers, initializer=_init_worker,

              initargs=(args.checkpoint,)) as pool:

        results = pool.map(_process_read, pairs)

    dt = time.time() - t0



    total_chars = sum(results)

    chars_per_sec = total_chars / dt

    mb_per_sec = chars_per_sec / (1024 * 1024)



    print(f"\n=== RESULT ===")

    print(f"Workers: {args.num_workers}")

    print(f"Total chars processed: {total_chars:,}")

    print(f"Wall time: {dt:.2f}s")

    print(f"Throughput: {chars_per_sec:,.0f} chars/sec = {mb_per_sec:.4f} MB/s")

    print(f"(target: 10 MB/s = 10,485,760 chars/sec)")

    print(f"Gap to target: {10_485_760 / chars_per_sec:.1f}x")





if __name__ == "__main__":

    main()
