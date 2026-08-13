

import time

from pathlib import Path



import numpy as np

import torch





ROOT = Path(__file__).resolve().parents[1]



# Find the actual h64 model directory rather than assuming layout.

candidates = []

for vocab in ROOT.rglob("vocab.txt"):

    if vocab.parent.name == "h64":

        candidates.append(vocab.parent)



if not candidates:

    raise RuntimeError("Could not find h64/vocab.txt under repo")



MODEL = candidates[0]



print("MODEL_DIR:", MODEL)





def load_matrix(name, rows, cols):

    x = np.fromfile(MODEL / name, dtype=np.float32)



    if x.size != rows * cols:

        raise RuntimeError(

            f"{name}: expected {rows*cols} floats, got {x.size}"

        )



    return torch.from_numpy(x.reshape(rows, cols).copy()).cuda()





def load_vector(name, size):

    x = np.fromfile(MODEL / name, dtype=np.float32)



    if x.size != size:

        raise RuntimeError(

            f"{name}: expected {size} floats, got {x.size}"

        )



    return torch.from_numpy(x.copy()).cuda()





with open(MODEL / "vocab.txt", "r") as f:

    vocab = f.read().rstrip("\r\n")



V = len(vocab)

H = 64

QE = 32

BE = 8

X = QE + 2 * BE



print("VOCAB_SIZE:", V)

print("HIDDEN:", H)

print("INPUT_DIM:", X)



qual_emb = load_matrix(

    "cell_qual_emb_weight.bin", V, QE

)



base_emb = load_matrix(

    "cell_base_emb_weight.bin", 5, BE

)



U_r = load_matrix("cell_U_r_weight.bin", H, X)

U_u = load_matrix("cell_U_u_weight.bin", H, X)

U_c = load_matrix("cell_U_c_weight.bin", H, X)



W_r = load_matrix("cell_W_r_weight.bin", H, H)

W_u = load_matrix("cell_W_u_weight.bin", H, H)

W_c = load_matrix("cell_W_c_weight.bin", H, H)



b_r = load_vector("cell_W_r_bias.bin", H)

b_u = load_vector("cell_W_u_bias.bin", H)

b_c = load_vector("cell_W_c_bias.bin", H)



out_W = load_matrix("out_proj_weight.bin", V, H)

out_b = load_vector("out_proj_bias.bin", V)



# Same fusion used by the C++ batched implementation.

W_ru = torch.cat((W_r, W_u), dim=0)

U_ru = torch.cat((U_r, U_u), dim=0)

b_ru = torch.cat((b_r, b_u), dim=0)





@torch.no_grad()

def model_step(h, q_idx, b0_idx, b1_idx):

    qe = qual_emb[q_idx]

    be0 = base_emb[b0_idx]

    be1 = base_emb[b1_idx]



    x = torch.cat((qe, be0, be1), dim=1)



    ru = (

        h @ W_ru.T

        + x @ U_ru.T

        + b_ru

    )



    r = torch.sigmoid(ru[:, :H])

    u = torch.sigmoid(ru[:, H:])



    c_pre = (

        (r * h) @ W_c.T

        + x @ U_c.T

        + b_c

    )



    c = torch.tanh(c_pre)



    h_new = (

        (1.0 - u) * h

        + u * c

    )



    logits = (

        h_new @ out_W.T

        + out_b

    )



    probs = torch.softmax(logits, dim=1)



    return h_new, probs





print("TORCH:", torch.__version__)

print("TORCH_CUDA:", torch.version.cuda)

print("CUDA_AVAILABLE:", torch.cuda.is_available())



if not torch.cuda.is_available():

    raise RuntimeError("CUDA unavailable inside GPU allocation")



print("GPU:", torch.cuda.get_device_name(0))

print("CAPABILITY:", torch.cuda.get_device_capability(0))



torch.manual_seed(1234)



# First benchmark strict FP32.

torch.backends.cuda.matmul.allow_tf32 = False



print()

print("=== QUALGRU H64 REAL-WEIGHT GPU BENCHMARK ===")

print("TF32: OFF")

print()



batches = [

    4096,

    8192,

    16384,

    32768,

    65536,

]



STEPS = 500

WARMUP = 50



best_rate = 0.0

best_batch = None



for B in batches:



    h = torch.zeros(

        B, H,

        device="cuda",

        dtype=torch.float32,

    )



    q_idx = torch.randint(

        0, V, (B,),

        device="cuda",

    )



    b0_idx = torch.randint(

        0, 5, (B,),

        device="cuda",

    )



    b1_idx = torch.randint(

        0, 5, (B,),

        device="cuda",

    )



    for _ in range(WARMUP):

        h, probs = model_step(

            h, q_idx, b0_idx, b1_idx

        )



    torch.cuda.synchronize()



    start = torch.cuda.Event(

        enable_timing=True

    )

    end = torch.cuda.Event(

        enable_timing=True

    )



    start.record()



    for _ in range(STEPS):

        h, probs = model_step(

            h, q_idx, b0_idx, b1_idx

        )



    end.record()

    torch.cuda.synchronize()



    ms = start.elapsed_time(end)

    seconds = ms / 1000.0



    symbols = B * STEPS

    rate = symbols / seconds



    if rate > best_rate:

        best_rate = rate

        best_batch = B



    # Materialize one result so benchmark output is unquestionably used.

    checksum = float(

        probs[0].sum().item()

    )



    print(

        f"B={B:4d}  "

        f"{ms/STEPS:9.4f} ms/step  "

        f"{rate/1e6:9.3f} M symbols/s  "

        f"checksum={checksum:.6f}"

    )





print()

print(

    f"BEST_BATCH={best_batch} "

    f"BEST_RATE={best_rate/1e6:.3f} M symbols/s"

)



print(

    "TARGET_50M=",

    "PASS" if best_rate >= 50_000_000 else "FAIL"

)

