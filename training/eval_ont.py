#!/usr/bin/env python3

import argparse

import numpy as np

from scipy.special import expit





def forward(chars, weights, stoi, vocab_size, h_prev=None):

    n = len(chars)

    hidden_size = weights['W_r'].shape[0]

    W_r, U_r, b_r = weights['W_r'], weights['U_r'], weights['b_r']

    W_u, U_u, b_u = weights['W_u'], weights['U_u'], weights['b_u']

    W_c, U_c, b_c = weights['W_c'], weights['U_c'], weights['b_c']

    W_y, b_y      = weights['W_y'], weights['b_y']

    h = np.zeros((n, hidden_size))

    if h_prev is not None:

        h[-1] = h_prev.reshape(-1)

    total_loss = 0.0

    for t in range(n - 1):

        x_t = np.zeros((vocab_size, 1))

        x_t[stoi[chars[t]]] = 1

        h_prev_col = h[t - 1].reshape(-1, 1)

        z_r = np.dot(W_r, h_prev_col) + np.dot(U_r, x_t) + b_r

        r = expit(z_r)

        z_u = np.dot(W_u, h_prev_col) + np.dot(U_u, x_t) + b_u

        u = expit(z_u)

        z_c = np.dot(W_c, r * h_prev_col) + np.dot(U_c, x_t) + b_c

        c = np.tanh(z_c)

        h[t] = ((1 - u) * h_prev_col + u * c).reshape(-1)

        y = np.dot(W_y, h[t]) + b_y

        shifted = y - np.max(y)

        p = np.exp(shifted) / np.sum(np.exp(shifted))

        target = stoi[chars[t + 1]]

        total_loss += -np.log(p[target])

    return h[n - 2].reshape(-1, 1), total_loss





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate GRU on quality score data.")

    parser.add_argument("data", help="Path to data file")

    parser.add_argument("--load", required=True, help="Weights file (without .npz)")

    parser.add_argument("--seq-len", type=int, default=25)

    parser.add_argument("--holdout-frac", type=float, default=0.1,

                        help="Fraction of data to hold out for eval (default 0.1 = last 10%)")

    parser.add_argument("--reset-hidden", action="store_true",

                        help="Reset hidden state between chunks (more honest eval)")

    args = parser.parse_args()



    # Load checkpoint

    npz = np.load(args.load + ".npz")

    weights = {k: npz[k] for k in npz.files if k != 'vocab'}



    # Reconstruct vocab from checkpoint if available, else infer from W_y shape

    if 'vocab' in npz.files:

        VOCAB = [str(c) for c in npz['vocab']]

        print(f"Loaded vocab from checkpoint: size {len(VOCAB)}")

    else:

        vocab_size_from_weights = weights['W_y'].shape[0]

        print(f"No vocab in checkpoint; inferring from data (W_y suggests {vocab_size_from_weights} symbols)")

        VOCAB = None  # will build from data below



    # Load data

    with open(args.data) as f:

        raw = f.read()



    if VOCAB is None:

        VOCAB = ['F', ':', ',', '#']

        print(f"Using known training vocab: size {len(VOCAB)}: {''.join(VOCAB)}")

        assert len(VOCAB) == weights['W_y'].shape[0], (

            f"Vocab from data ({len(VOCAB)}) doesn't match checkpoint W_y rows ({weights['W_y'].shape[0]})"

        )



    stoi = {c: i for i, c in enumerate(VOCAB)}

    vocab_size = len(VOCAB)



    # Filter to in-vocab chars and split train/eval

    data = [ch for ch in raw if ch in stoi]

    n_total = len(data)

    n_eval = int(n_total * args.holdout_frac)

    n_train = n_total - n_eval

    eval_data = data[n_train:]

    print(f"Loaded {n_total:,} chars total; using last {n_eval:,} ({args.holdout_frac*100:.0f}%) for eval")

    print(f"Loaded weights from {args.load}.npz")



    # Chunk the eval portion

    chunks = [eval_data[i:i + args.seq_len + 1]

              for i in range(0, len(eval_data) - args.seq_len, args.seq_len)]

    print(f"Eval chunks: {len(chunks)}, seq_len: {args.seq_len}")

    print(f"Hidden state reset between chunks: {args.reset_hidden}")



    total_loss = 0.0

    total_chars = 0

    h_prev = None

    for i, chunk in enumerate(chunks):

        if args.reset_hidden:

            h_prev = None

        h_prev, loss = forward(chunk, weights, stoi, vocab_size, h_prev)

        total_loss += loss

        total_chars += len(chunk) - 1

        if i % 5000 == 0:

            print(f"  step {i}/{len(chunks)}  running avg loss {total_loss / max(total_chars,1):.4f}",

                  flush=True)



    avg_loss = total_loss / total_chars

    bpc = avg_loss / np.log(2)

    H_marginal = np.log2(vocab_size)

    print(f"\nResults over {total_chars:,} eval chars:")

    print(f"  avg cross-entropy loss : {avg_loss:.4f} nats")

    print(f"  bits per character     : {bpc:.4f}")

    print(f"  random baseline bpc    : {H_marginal:.4f}  (vocab size {vocab_size})")
