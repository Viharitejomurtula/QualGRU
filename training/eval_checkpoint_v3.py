#!/usr/bin/env python3

"""

eval_checkpoint_v3.py — load a saved base-conditioned QualGRU (v3) checkpoint

and report bpc on any paired data file. No training, no gradients.



Usage:

    python3 eval_checkpoint_v3.py weights_v3_base_epoch14.pt paired_chr20_unseen_500.txt

"""



import argparse

import math



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader



BASE_VOCAB = ['A', 'C', 'G', 'T', 'N']

BASE_STOI = {b: i for i, b in enumerate(BASE_VOCAB)}

BASE_UNK = BASE_STOI['N']





class PairedReadDataset(Dataset):

    def __init__(self, pairs, qual_stoi):

        self.samples = []

        for bases, quals in pairs:

            qual_ids = [qual_stoi[ch] for ch in quals if ch in qual_stoi]

            if len(qual_ids) != len(quals) or len(qual_ids) < 2:

                continue  # drop reads containing a quality char unseen at train time

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





def load_paired(path, chunk_len=None):

    pairs = []

    with open(path) as f:

        for line in f:

            line = line.rstrip('\n\r')

            if not line or '\t' not in line:

                continue

            bases, quals = line.split('\t', 1)

            if len(bases) == len(quals) and len(bases) >= 2:

                pairs.append((bases, quals))



    if chunk_len:

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

        pairs = chunked



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

        hidden_states = []

        for t in range(T - 1):

            x_t = self.cell.encode_input(qual_padded[:, t], base_padded[:, t], base_padded[:, t + 1])

            h = self.cell(x_t, h)

            hidden_states.append(h)

        h_seq = torch.stack(hidden_states, dim=1)

        return self.out_proj(h_seq)





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

def main():

    parser = argparse.ArgumentParser(description="Evaluate a base-conditioned QualGRU checkpoint's bpc on paired data.")

    parser.add_argument("checkpoint", help="Path to .pt checkpoint (from rnn_ont_torch_v3.py)")

    parser.add_argument("data", help="Paired data file (bases\\tqualities, one read per line)")

    parser.add_argument("--seq-len", type=int, default=200)

    parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}", flush=True)



    ckpt = torch.load(args.checkpoint, map_location=device)

    qual_vocab = ckpt['qual_vocab']

    qual_stoi = {c: i for i, c in enumerate(qual_vocab)}

    hidden_size = ckpt['hidden_size']



    print(f"Checkpoint: hidden_size={hidden_size}, qual_vocab_size={len(qual_vocab)}, "

          f"epoch={ckpt.get('epoch', '?')}", flush=True)

    if 'val_bpc' in ckpt:

        print(f"  (checkpoint's own recorded val_bpc at save time: {ckpt['val_bpc']:.4f})")



    model = QualGRUBaseConditioned(len(qual_vocab), hidden_size).to(device)

    model.load_state_dict(ckpt['model'])

    model.eval()



    pairs = load_paired(args.data, args.seq_len)

    print(f"Loaded {len(pairs)} chunks from {args.data}", flush=True)



    loader = DataLoader(

        PairedReadDataset(pairs, qual_stoi), batch_size=args.batch_size, shuffle=False,

        collate_fn=collate_pad,

    )



    total_loss, total_tokens = 0.0, 0

    for base_padded, qual_padded, lengths in loader:

        base_padded = base_padded.to(device)

        qual_padded = qual_padded.to(device)

        lengths = lengths.to(device)

        targets = qual_padded[:, 1:]

        logits = model(base_padded, qual_padded, lengths)

        loss, n_tok = masked_cross_entropy(logits, targets, lengths)

        total_loss += loss.item() * n_tok.item()

        total_tokens += n_tok.item()



    avg_loss = total_loss / max(total_tokens, 1)

    avg_bpc = avg_loss / math.log(2)

    print(f"\n=== RESULT ===")

    print(f"Checkpoint: {args.checkpoint}")

    print(f"Data:       {args.data}")

    print(f"Tokens evaluated: {total_tokens:,}")

    print(f"bpc: {avg_bpc:.4f}")





if __name__ == "__main__":

    main()
