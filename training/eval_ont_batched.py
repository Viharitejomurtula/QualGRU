#!/usr/bin/env python3



import argparse

import time

import numpy as np

from scipy.special import expit





def forward_batch(seqs, weights, stoi, vocab_size):

    """

    Batched GRU forward pass. Processes B sequences in parallel.

    seqs: list of B sequences (each a list of chars), all same length L+1

    Returns: total_loss (float), total_chars (int)

    """

    B = len(seqs)

    L = len(seqs[0]) - 1  # number of prediction steps



    W_r, U_r, b_r = weights['W_r'], weights['U_r'], weights['b_r']

    W_u, U_u, b_u = weights['W_u'], weights['U_u'], weights['b_u']

    W_c, U_c, b_c = weights['W_c'], weights['U_c'], weights['b_c']

    W_y, b_y      = weights['W_y'], weights['b_y']



    hidden_size = W_r.shape[0]



    # Build one-hot input matrix: shape [L+1, B, vocab_size]

    X = np.zeros((L + 1, B, vocab_size), dtype=np.float32)

    for b, seq in enumerate(seqs):

        for t, ch in enumerate(seq):

            if ch in stoi:

                X[t, b, stoi[ch]] = 1.0



    # Precompute all input projections at once (big matmuls, not per-step)

    # X reshaped to [(L+1)*B, vocab_size] for a single matmul, then reshape back

    X_flat = X.reshape(-1, vocab_size)  # [(L+1)*B, vocab_size]



    Ux_r = (X_flat @ U_r.T).reshape(L + 1, B, hidden_size)  # [L+1, B, H]

    Ux_u = (X_flat @ U_u.T).reshape(L + 1, B, hidden_size)

    Ux_c = (X_flat @ U_c.T).reshape(L + 1, B, hidden_size)



    # Output projection precomputed per step after hidden state computed

    h = np.zeros((B, hidden_size), dtype=np.float32)  # [B, H]



    total_loss = 0.0

    total_chars = 0



    for t in range(L):

        # Recurrent projections: [B, H]

        Wh_r = h @ W_r.T

        Wh_u = h @ W_u.T



        r = expit(Wh_r + Ux_r[t] + b_r.T)   # [B, H]

        u = expit(Wh_u + Ux_u[t] + b_u.T)   # [B, H]



        Wh_c = (r * h) @ W_c.T

        c = np.tanh(Wh_c + Ux_c[t] + b_c.T) # [B, H]



        h = (1 - u) * h + u * c              # [B, H]



        # Output logits: [B, vocab_size]

        y = h @ W_y.T + b_y.T

        y -= y.max(axis=1, keepdims=True)    # numerical stability

        exp_y = np.exp(y)

        p = exp_y / exp_y.sum(axis=1, keepdims=True)  # [B, vocab_size]



        # Gather loss for each sequence's target at t+1

        for b, seq in enumerate(seqs):

            if t + 1 < len(seq) and seq[t + 1] in stoi:

                target = stoi[seq[t + 1]]

                total_loss += -np.log(p[b, target] + 1e-10)

                total_chars += 1



    return total_loss, total_chars





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Batched GRU eval on quality score data.")

    parser.add_argument("data", help="Path to data file")

    parser.add_argument("--load", required=True, help="Weights file (without .npz)")

    parser.add_argument("--seq-len", type=int, default=100)

    parser.add_argument("--holdout-frac", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=128,

                        help="Number of sequences to process in parallel (default 128)")

    args = parser.parse_args()



    # Load weights

    npz = np.load(args.load + ".npz")

    weights = {k: npz[k] for k in npz.files if k != 'vocab'}



    if 'vocab' in npz.files:

        VOCAB = [str(c) for c in npz['vocab']]

        print(f"Loaded vocab from checkpoint: size {len(VOCAB)}")

    else:

        vocab_size_from_weights = weights['W_y'].shape[0]

        print(f"No vocab in checkpoint; W_y suggests {vocab_size_from_weights} symbols")

        VOCAB = None



    # Load data

    with open(args.data) as f:

        raw = f.read()



    if VOCAB is None:

        VOCAB = ['F', ':', ',', '#']

        print(f"Using fallback vocab: {VOCAB}")



    stoi = {c: i for i, c in enumerate(VOCAB)}

    vocab_size = len(VOCAB)



    data = [ch for ch in raw if ch in stoi]

    n_total = len(data)

    n_eval = int(n_total * args.holdout_frac)

    n_train = n_total - n_eval

    eval_data = data[n_train:]

    print(f"Total chars: {n_total:,} | Eval chars: {n_eval:,} ({args.holdout_frac*100:.0f}%)")



    # Build chunks

    chunks = [eval_data[i:i + args.seq_len + 1]

              for i in range(0, len(eval_data) - args.seq_len, args.seq_len)]

    print(f"Eval chunks: {len(chunks)} | seq_len: {args.seq_len} | batch_size: {args.batch_size}")



    total_loss = 0.0

    total_chars = 0

    t0 = time.perf_counter()



    for i in range(0, len(chunks), args.batch_size):

        batch = chunks[i:i + args.batch_size]

        # Pad all seqs to same length within batch

        max_len = max(len(s) for s in batch)

        padded = [s + [s[-1]] * (max_len - len(s)) for s in batch]



        loss, chars = forward_batch(padded, weights, stoi, vocab_size)

        total_loss += loss

        total_chars += chars



        if (i // args.batch_size) % 20 == 0:

            elapsed = time.perf_counter() - t0

            chars_done = total_chars

            speed = chars_done / elapsed / 1e6 if elapsed > 0 else 0

            print(f"  batch {i//args.batch_size}/{len(chunks)//args.batch_size}"

                  f"  running bpc: {(total_loss/max(total_chars,1))/np.log(2):.4f}"

                  f"  speed: {speed:.3f} MB/s", flush=True)



    elapsed = time.perf_counter() - t0

    avg_loss = total_loss / total_chars

    bpc = avg_loss / np.log(2)

    speed_mbs = total_chars / elapsed / 1e6



    print(f"\nResults over {total_chars:,} eval chars:")

    print(f"  avg cross-entropy loss : {avg_loss:.4f} nats")

    print(f"  bits per character     : {bpc:.4f}")

    print(f"  random baseline bpc    : {np.log2(vocab_size):.4f}  (vocab {vocab_size})")

    print(f"  elapsed                : {elapsed:.1f}s")

    print(f"  throughput             : {speed_mbs:.3f} MB/s")
