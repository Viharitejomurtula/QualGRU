

import time

from pathlib import Path



import numpy as np

import torch



# ------------------------------------------------------------

# Constants: same rANS scale as QualGRU

# ------------------------------------------------------------



PROB_BITS = 16

M = 1 << PROB_BITS

LOWER = 1 << 23



BATCH = 65536

SEQ_LEN = 1024



ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "compressor" / "models" / "h64"



FASTQ = Path(

    "/private/home/vtejomur/compression/compressor_ont/CoLoRd/chr20_slice.fastq"

)



DEVICE = "cuda"



torch.backends.cuda.matmul.allow_tf32 = False





# ------------------------------------------------------------

# Load real h64 weights

# ------------------------------------------------------------



with open(MODEL / "vocab.txt") as f:

    vocab = f.read().rstrip("\r\n")



V = len(vocab)

H = 64

QE = 32

BE = 8

X = 48



qmap = {c: i for i, c in enumerate(vocab)}

bmap = {

    "A": 0,

    "C": 1,

    "G": 2,

    "T": 3,

    "N": 4,

}





def matrix(name, rows, cols):

    a = np.fromfile(MODEL / name, dtype=np.float32)

    assert a.size == rows * cols, (name, a.size)

    return torch.from_numpy(

        a.reshape(rows, cols).copy()

    ).to(DEVICE)





def vector(name, n):

    a = np.fromfile(MODEL / name, dtype=np.float32)

    assert a.size == n, (name, a.size)

    return torch.from_numpy(a.copy()).to(DEVICE)





qual_emb = matrix("cell_qual_emb_weight.bin", V, QE)

base_emb = matrix("cell_base_emb_weight.bin", 5, BE)



U_r = matrix("cell_U_r_weight.bin", H, X)

U_u = matrix("cell_U_u_weight.bin", H, X)

U_c = matrix("cell_U_c_weight.bin", H, X)



W_r = matrix("cell_W_r_weight.bin", H, H)

W_u = matrix("cell_W_u_weight.bin", H, H)

W_c = matrix("cell_W_c_weight.bin", H, H)



b_r = vector("cell_W_r_bias.bin", H)

b_u = vector("cell_W_u_bias.bin", H)

b_c = vector("cell_W_c_bias.bin", H)



out_W = matrix("out_proj_weight.bin", V, H)

out_b = vector("out_proj_bias.bin", V)



W_ru = torch.cat((W_r, W_u), dim=0)

U_ru = torch.cat((U_r, U_u), dim=0)

b_ru = torch.cat((b_r, b_u), dim=0)





# ------------------------------------------------------------

# Read real ONT reads

# Fixed-length prefixes for milestone 1.

# ------------------------------------------------------------



q_rows = []

b_rows = []



with open(FASTQ) as f:

    while len(q_rows) < BATCH:

        name = f.readline()

        if not name:

            break



        seq = f.readline().strip()

        f.readline()

        qual = f.readline().strip()



        if len(seq) < SEQ_LEN or len(qual) < SEQ_LEN:

            continue



        try:

            qi = [qmap[c] for c in qual[:SEQ_LEN]]

        except KeyError:

            continue



        bi = [

            bmap.get(c.upper(), 4)

            for c in seq[:SEQ_LEN]

        ]



        q_rows.append(qi)

        b_rows.append(bi)



if len(q_rows) != BATCH:

    raise RuntimeError(

        f"needed {BATCH} reads >= {SEQ_LEN}, got {len(q_rows)}"

    )



q_true = torch.tensor(

    q_rows,

    dtype=torch.long,

    device=DEVICE,

)



bases = torch.tensor(

    b_rows,

    dtype=torch.long,

    device=DEVICE,

)



print("GPU:", torch.cuda.get_device_name(0))

print("BATCH:", BATCH)

print("SEQ_LEN:", SEQ_LEN)

print("VOCAB:", V)





# ------------------------------------------------------------

# Real h64 model timestep

# ------------------------------------------------------------



@torch.no_grad()

def model_step(h, q_idx, b0, b1):



    x = torch.cat(

        (

            qual_emb[q_idx],

            base_emb[b0],

            base_emb[b1],

        ),

        dim=1,

    )



    ru = h @ W_ru.T + x @ U_ru.T + b_ru



    r = torch.sigmoid(ru[:, :H])

    u = torch.sigmoid(ru[:, H:])



    c = torch.tanh(

        (r * h) @ W_c.T

        + x @ U_c.T

        + b_c

    )



    h = (1.0 - u) * h + u * c



    logits = h @ out_W.T + out_b

    probs = torch.softmax(logits, dim=1)



    return h, probs





# ------------------------------------------------------------

# GPU-friendly deterministic quantiser.

#

# Encoder and decoder use THIS SAME function.

# It produces positive integer frequencies summing to M.

#

# New GPU regime; not claiming CPU archive compatibility.

# ------------------------------------------------------------



@torch.no_grad()

def quantise_gpu(probs):



    scaled = probs * float(M)



    freq = torch.floor(

        scaled

    ).to(torch.int64)



    # rANS requires every symbol to have positive frequency.

    freq = torch.clamp(

        freq,

        min=1,

    )



    total = freq.sum(

        dim=1

    )



    # Residual needed to make each row sum exactly to M.

    delta = M - total



    # Put the residual into the largest-frequency symbol.

    # Avoids the expensive per-row argsort.

    biggest = torch.argmax(

        freq,

        dim=1,

        keepdim=True,

    )



    freq.scatter_add_(

        1,

        biggest,

        delta.unsqueeze(1),

    )



    prefix = torch.cumsum(

        freq,

        dim=1,

    )



    zeros = torch.zeros(

        freq.shape[0],

        1,

        dtype=freq.dtype,

        device=freq.device,

    )



    cum = torch.cat(

        (zeros, prefix),

        dim=1,

    )



    return freq, cum
# ------------------------------------------------------------

# Forward pass for encoder:

# build GPU frequency tables from true previous symbols.

# Store only freq; cumulative values can be reconstructed.

# ------------------------------------------------------------



print()

print("===== GPU ENCODE MODEL/TABLE PASS =====")



torch.cuda.synchronize()

enc_start = time.perf_counter()



h = torch.zeros(

    BATCH,

    H,

    dtype=torch.float32,

    device=DEVICE,

)



freq_tables = torch.empty(

    BATCH,

    SEQ_LEN - 1,

    V,

    dtype=torch.int32,

    device=DEVICE,

)



for t in range(SEQ_LEN - 1):



    h, probs = model_step(

        h,

        q_true[:, t],

        bases[:, t],

        bases[:, t + 1],

    )



    freq, _ = quantise_gpu(probs)



    freq_tables[:, t, :] = freq.to(

        torch.int32

    )



torch.cuda.synchronize()



table_seconds = time.perf_counter() - enc_start



print(

    "table pass:",

    f"{BATCH * (SEQ_LEN - 1) / table_seconds / 1e6:.3f}",

    "M symbols/s",

)





# ------------------------------------------------------------

# Batched rANS encode backwards.

#

# Every row gets its own independent rANS state and byte area.

# ------------------------------------------------------------



print()

print("===== GPU RANS ENCODE =====")



MAX_BYTES = SEQ_LEN * 8 + 64



streams = torch.zeros(

    BATCH,

    MAX_BYTES,

    dtype=torch.uint8,

    device=DEVICE,

)



ptr = torch.full(

    (BATCH,),

    MAX_BYTES,

    dtype=torch.int64,

    device=DEVICE,

)



xstate = torch.full(

    (BATCH,),

    LOWER,

    dtype=torch.int64,

    device=DEVICE,

)



rows = torch.arange(

    BATCH,

    device=DEVICE,

)



torch.cuda.synchronize()

rans_enc_start = time.perf_counter()



for t in range(SEQ_LEN - 2, -1, -1):



    freq = freq_tables[:, t, :].to(

        torch.int64

    )



    target = q_true[:, t + 1]



    fs = freq.gather(

        1,

        target.unsqueeze(1),

    ).squeeze(1)



    # cumulative frequency for target symbol

    prefix = torch.cumsum(freq, dim=1)



    cs = torch.where(

        target == 0,

        torch.zeros_like(fs),

        prefix.gather(

            1,

            (target - 1).clamp(min=0).unsqueeze(1),

        ).squeeze(1),

    )



    bound = 32768 * fs



    # At most a few bytes are needed for a 32-bit rANS state.

    for _ in range(4):



        emit = xstate >= bound



        new_ptr = ptr - emit.to(torch.int64)



        safe_ptr = torch.clamp(

            new_ptr,

            min=0,

            max=MAX_BYTES - 1,

        )



        old = streams[

            rows,

            safe_ptr,

        ]



        byte = (

            xstate & 0xFF

        ).to(torch.uint8)



        streams[

            rows,

            safe_ptr,

        ] = torch.where(

            emit,

            byte,

            old,

        )



        ptr = new_ptr



        xstate = torch.where(

            emit,

            xstate >> 8,

            xstate,

        )



    xstate = (

        ((xstate // fs) << PROB_BITS)

        + (xstate % fs)

        + cs

    )



torch.cuda.synchronize()



rans_enc_seconds = (

    time.perf_counter()

    - rans_enc_start

)



encoded_bytes = (

    MAX_BYTES - ptr

).sum().item()



rans_bpc = (

    8.0 * encoded_bytes

    / (BATCH * (SEQ_LEN - 1))

)



print(

    "rANS encode:",

    f"{BATCH * (SEQ_LEN - 1) / rans_enc_seconds / 1e6:.3f}",

    "M symbols/s",

)



print("encoded bytes:", encoded_bytes)

print("rANS quality bpc:", f"{rans_bpc:.4f}")





# ------------------------------------------------------------

# FULL GPU DECODE

#

# This is the important path:

#

# model

# -> probabilities

# -> quantise

# -> rANS symbol

# -> next model input

# ------------------------------------------------------------



print()

print("===== FULL GPU DECODE =====")



decoded = torch.empty(

    BATCH,

    SEQ_LEN,

    dtype=torch.long,

    device=DEVICE,

)



decoded[:, 0] = q_true[:, 0]



h = torch.zeros(

    BATCH,

    H,

    dtype=torch.float32,

    device=DEVICE,

)



dx = xstate.clone()

dptr = ptr.clone()



torch.cuda.synchronize()



start_event = torch.cuda.Event(

    enable_timing=True

)

end_event = torch.cuda.Event(

    enable_timing=True

)



start_event.record()



for t in range(SEQ_LEN - 1):



    h, probs = model_step(

        h,

        decoded[:, t],

        bases[:, t],

        bases[:, t + 1],

    )



    freq, cum = quantise_gpu(probs)



    slot = dx & (M - 1)



    # Find symbol whose cumulative range contains slot.

    symbol = (

        cum[:, 1:]

        <= slot.unsqueeze(1)

    ).sum(dim=1)



    fs = freq.gather(

        1,

        symbol.unsqueeze(1),

    ).squeeze(1)



    cs = cum.gather(

        1,

        symbol.unsqueeze(1),

    ).squeeze(1)



    dx = (

        fs * (dx >> PROB_BITS)

        + slot

        - cs

    )



    # rANS renormalisation.

    for _ in range(4):



        need = dx < LOWER



        safe_ptr = torch.clamp(

            dptr,

            min=0,

            max=MAX_BYTES - 1,

        )



        byte = streams[

            rows,

            safe_ptr,

        ].to(torch.int64)



        dx = torch.where(

            need,

            (dx << 8) | byte,

            dx,

        )



        dptr = (

            dptr

            + need.to(torch.int64)

        )



    decoded[:, t + 1] = symbol



end_event.record()

torch.cuda.synchronize()



decode_ms = start_event.elapsed_time(

    end_event

)



decode_seconds = decode_ms / 1000.0



decoded_symbols = (

    BATCH * (SEQ_LEN - 1)

)



decode_rate = (

    decoded_symbols

    / decode_seconds

)



# ------------------------------------------------------------

# Validation

# ------------------------------------------------------------



mismatch = (

    decoded != q_true

)



n_bad = mismatch.sum().item()



print(

    "decode rate:",

    f"{decode_rate / 1e6:.3f}",

    "M symbols/s",

)



print(

    "TARGET_50M:",

    "PASS"

    if decode_rate >= 50_000_000

    else "FAIL",

)



print("mismatched symbols:", n_bad)



if n_bad:



    loc = torch.nonzero(

        mismatch,

        as_tuple=False,

    )[0].tolist()



    r, t = loc



    print(

        "FIRST_MISMATCH:",

        "read", r,

        "t", t,

        "expected", q_true[r, t].item(),

        "got", decoded[r, t].item(),

    )



    raise SystemExit(1)



# Verify decoder consumed exactly the emitted bytes.

remaining = (

    dptr != MAX_BYTES

).sum().item()



print(

    "streams not fully consumed:",

    remaining,

)



if remaining:

    raise SystemExit(2)



print()

print("GPU_RANS_ROUNDTRIP=PASS")

