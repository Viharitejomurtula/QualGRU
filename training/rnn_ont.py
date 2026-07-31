#!/usr/bin/env python3

import argparse

import numpy as np

from scipy.special import expit



VOCAB = None

stoi = None

itos = None

vocab_size = None



def reset_gate(W_r, h_prev, U_r, x, b_r):

    z_r = np.dot(W_r, h_prev) + np.dot(U_r, x) + b_r

    return expit(z_r)





def update_gate(W_u, h_prev, U_u, x, b_u):

    z_u = np.dot(W_u, h_prev) + np.dot(U_u, x) + b_u

    return expit(z_u)





def new_memory(W_c, r, h_prev, U_c, x, b_c):

    had = r * h_prev

    z_c = np.dot(W_c, had) + np.dot(U_c, x) + b_c

    return np.tanh(z_c)





def final_memory(u, h_prev, c):

    return ((1 - u) * h_prev) + (u * c)





def init_weights(hidden_size):

    rnn_scale = np.sqrt(2.0 / (hidden_size + hidden_size))

    inp_scale = np.sqrt(2.0 / (hidden_size + vocab_size))

    out_scale = np.sqrt(2.0 / (vocab_size + hidden_size))

    return {

        'W_r': np.random.randn(hidden_size, hidden_size) * rnn_scale,

        'U_r': np.random.randn(hidden_size, vocab_size) * inp_scale,

        'b_r': np.zeros((hidden_size, 1)),

        'W_u': np.random.randn(hidden_size, hidden_size) * rnn_scale,

        'U_u': np.random.randn(hidden_size, vocab_size) * inp_scale,

        'b_u': np.zeros((hidden_size, 1)),

        'W_c': np.random.randn(hidden_size, hidden_size) * rnn_scale,

        'U_c': np.random.randn(hidden_size, vocab_size) * inp_scale,

        'b_c': np.zeros((hidden_size, 1)),

        'W_y': np.random.randn(vocab_size, hidden_size) * out_scale,

        'b_y': np.zeros(vocab_size),

    }





def init_adam(weights):

    m = {k: np.zeros_like(v) for k, v in weights.items()}

    v = {k: np.zeros_like(v) for k, v in weights.items()}

    return m, v





def clip_grads(grads, thresh=5.0):

    total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))

    if total_norm > thresh:

        for key in grads:

            grads[key] *= thresh / total_norm

    return grads





def adam_step(weights, grads, m, v, t, lr=1e-3):

    t += 1

    for key in weights:

        m[key] = 0.9 * m[key] + 0.1 * grads[key]

        v[key] = 0.999 * v[key] + 0.001 * grads[key]**2

        m_hat = m[key] / (1 - 0.9**t)

        v_hat = v[key] / (1 - 0.999**t)

        weights[key] -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    return weights, m, v, t





def sample_gru(chars, weights, h_prev=None):

    """

    h_prev: hidden state to carry in, or None to start from zero.

    NOTE: h_prev should be None at the start of every NEW READ.

    Only pass a real h_prev when continuing mid-read across seq_len chunks.

    """

    n = len(chars)

    hidden_size = weights['W_r'].shape[0]



    W_r, U_r, b_r = weights['W_r'], weights['U_r'], weights['b_r']

    W_u, U_u, b_u = weights['W_u'], weights['U_u'], weights['b_u']

    W_c, U_c, b_c = weights['W_c'], weights['U_c'], weights['b_c']

    W_y, b_y      = weights['W_y'], weights['b_y']



    h = np.zeros((n, hidden_size))

    # h[-1] slot holds the carried-in state (zeros if None / new read)

    if h_prev is not None:

        h[-1] = h_prev.reshape(-1)

    # else h[-1] stays zero — this is the read-boundary reset



    y = np.zeros((n, vocab_size))

    p = np.zeros((n, vocab_size))



    L = 0

    l = np.zeros(n)



    dW_r, dW_u, dW_c = np.zeros_like(W_r), np.zeros_like(W_u), np.zeros_like(W_c)

    dW_y = np.zeros_like(W_y)

    dU_r, dU_u, dU_c = np.zeros_like(U_r), np.zeros_like(U_u), np.zeros_like(U_c)

    db_r, db_u, db_c = np.zeros_like(b_r), np.zeros_like(b_u), np.zeros_like(b_c)

    db_y = np.zeros_like(b_y)



    target = np.zeros(n - 1, dtype=int)



    r_store = np.zeros((n, hidden_size, 1))

    u_store = np.zeros((n, hidden_size, 1))

    c_store = np.zeros((n, hidden_size, 1))

    x_store = np.zeros((n, vocab_size, 1))



    dh_prev = np.zeros((hidden_size, 1))



    # forward pass

    for t in range(n - 1):

        x_t = np.zeros((vocab_size, 1))

        x_t[stoi[chars[t]]] = 1

        h_prev_col = h[t - 1].reshape(-1, 1)

        r = reset_gate(W_r, h_prev_col, U_r, x_t, b_r)

        u = update_gate(W_u, h_prev_col, U_u, x_t, b_u)

        c = new_memory(W_c, r, h_prev_col, U_c, x_t, b_c)



        x_store[t] = x_t

        r_store[t] = r

        u_store[t] = u

        c_store[t] = c



        h[t] = final_memory(u, h_prev_col, c).reshape(-1)

        y[t] = np.dot(W_y, h[t]) + b_y

        shifted = y[t] - np.max(y[t])

        p[t] = np.exp(shifted) / np.sum(np.exp(shifted))

        target[t] = stoi[chars[t + 1]]

        l[t] = -1 * np.log(p[t][target[t]])

        L += l[t]



    # bptt

    for t in reversed(range(n - 1)):

        dy = np.copy(p[t])

        dy[target[t]] -= 1

        db_y += dy

        dW_y += np.outer(dy, h[t])



        dh = np.dot(W_y.T, dy).reshape(-1, 1) + dh_prev

        dc = dh * u_store[t]

        dz_c = dc * (1 - c_store[t] ** 2)

        h_til = r_store[t] * h[t - 1].reshape(-1, 1)

        dW_c += np.dot(dz_c, h_til.T)



        dU_c += np.dot(dz_c, x_store[t].T)

        db_c += dz_c



        du = dh * (c_store[t] - h[t - 1].reshape(-1, 1))

        dz_u = du * u_store[t] * (1 - u_store[t])

        dW_u += np.dot(dz_u, h[t - 1].reshape(1, -1))

        dU_u += np.dot(dz_u, x_store[t].T)

        db_u += dz_u



        dr = (np.dot(W_c.T, dz_c)) * h[t - 1].reshape(-1, 1)

        dz_r = dr * r_store[t] * (1 - r_store[t])

        dW_r += np.dot(dz_r, h[t - 1].reshape(1, -1))

        dU_r += np.dot(dz_r, x_store[t].T)

        db_r += dz_r



        dh_prev = np.dot(W_r.T, dz_r)

        dh_prev += np.dot(W_u.T, dz_u)

        dh_prev += dh * (1 - u_store[t])



    grads = {

        'W_r': dW_r, 'U_r': dU_r, 'b_r': db_r,

        'W_u': dW_u, 'U_u': dU_u, 'b_u': db_u,

        'W_c': dW_c, 'U_c': dU_c, 'b_c': db_c,

        'W_y': dW_y, 'b_y': db_y,

    }



    return h[n - 2].reshape(-1, 1), L, grads





def build_read_chunks(reads, seq_len):

    """

    Build (chunk, is_new_read) pairs from a list of reads (each a list of chars).

    Each read is independently sliced into seq_len+1 chunks. The hidden state

    must reset to None at the start of every read (is_new_read=True for the

    first chunk of each read), and may carry over between chunks WITHIN a read.

    """

    chunks = []

    for read in reads:

        if len(read) <= seq_len:

            # short read: just use it as one chunk if long enough to predict anything

            if len(read) >= 2:

                chunks.append((read, True))

            continue

        first = True

        for i in range(0, len(read) - seq_len, seq_len):

            chunk = read[i:i + seq_len + 1]

            chunks.append((chunk, first))

            first = False

    return chunks





def train(reads, hidden_size=64, seq_len=25, epochs=10, lr=1e-3,

          init_weights_dict=None, start_epoch=0, save_path=None):

    weights = init_weights_dict if init_weights_dict is not None else init_weights(hidden_size)

    m, v = init_adam(weights)

    t = 0



    chunks = build_read_chunks(reads, seq_len)

    print(f"Built {len(chunks)} chunks from {len(reads)} reads "

          f"(read-boundary resets enabled)", flush=True)



    for epoch in range(start_epoch, start_epoch + epochs):

        total_loss = 0.0

        h_prev = None

        for step, (chunk, is_new_read) in enumerate(chunks):

            if is_new_read:

                h_prev = None  # <-- THE FIX: reset hidden state at read boundaries

            h_prev, loss, grads = sample_gru(chunk, weights, h_prev)

            grads = clip_grads(grads)

            weights, m, v, t = adam_step(weights, grads, m, v, t, lr=lr)

            total_loss += loss

            if step % 1000 == 0:

                print(f"epoch {epoch}  step {step}/{len(chunks)}  loss {loss:.4f}", flush=True)

        print(f"epoch {epoch} done  avg loss {total_loss / len(chunks):.4f}", flush=True)

        if save_path:

            ckpt = f"{save_path}_epoch{epoch}"

            np.savez(ckpt, vocab=np.array(VOCAB), **weights)

            print(f"checkpoint saved to {ckpt}.npz", flush=True)



    return weights





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a GRU on quality score data (with read-boundary resets).")

    parser.add_argument("data", help="Path to file containing quality score strings, one read per line")

    parser.add_argument("--hidden-size", type=int, default=64, help="GRU hidden state size (default: 64)")

    parser.add_argument("--seq-len", type=int, default=25, help="Sequence chunk length (default: 25)")

    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")

    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate (default: 1e-3)")

    parser.add_argument("--save", type=str, default=None, help="Path to save trained weights (without .npz extension)")

    parser.add_argument("--load", type=str, default=None, help="Path to load weights from (without .npz extension)")

    parser.add_argument("--start-epoch", type=int, default=0, help="Epoch number to start from (for display, default: 0)")

    args = parser.parse_args()



    with open(args.data) as f:

        lines = f.readlines()



    # Each line is one read. Strip newline, keep as a separate sequence.

    raw_reads = [line.rstrip('\n\r') for line in lines if line.strip()]



    # Build vocab from ALL chars across all reads

    all_chars = ''.join(raw_reads)

    distinct = sorted(set(all_chars))

    VOCAB = distinct

    stoi = {c: i for i, c in enumerate(VOCAB)}

    itos = {i: c for i, c in enumerate(VOCAB)}

    vocab_size = len(VOCAB)

    print(f"Built vocab of size {vocab_size}: {''.join(VOCAB)}")



    globals()['VOCAB'] = VOCAB

    globals()['stoi'] = stoi

    globals()['itos'] = itos

    globals()['vocab_size'] = vocab_size



    # Convert each read into a list of in-vocab chars (drop anything stray)

    reads = [[ch for ch in read if ch in stoi] for read in raw_reads]

    reads = [r for r in reads if len(r) >= 2]  # need at least 2 chars to predict anything

    total_chars = sum(len(r) for r in reads)

    print(f"Loaded {len(reads)} reads, {total_chars} total chars")



    init_weights_dict = None

    if args.load:

        npz = np.load(args.load + ".npz")

        init_weights_dict = {k: npz[k] for k in npz.files if k != 'vocab'}

        print(f"Loaded weights from {args.load}.npz")



    weights = train(reads, hidden_size=args.hidden_size, seq_len=args.seq_len,

                    epochs=args.epochs, lr=args.lr,

                    init_weights_dict=init_weights_dict, start_epoch=args.start_epoch,

                    save_path=args.save)
