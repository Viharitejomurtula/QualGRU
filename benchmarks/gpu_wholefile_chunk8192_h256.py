

import time

from pathlib import Path



import numpy as np

import torch



PROB_BITS = 16

M = 1 << PROB_BITS

LOWER = 1 << 23



ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "compressor" / "models" / "h256"



FASTQ = Path(

    "/private/home/vtejomur/compression/compressor_ont/CoLoRd/chr20_slice.fastq"

)



DEVICE = "cuda"
CHUNK = 8192



# Keeps the giant [T,B,50] frequency tensor safely below A100 memory.

CELL_BUDGET = 2_000_000_000

MAX_BATCH = 131072



torch.backends.cuda.matmul.allow_tf32 = True

torch.set_float32_matmul_precision("high")





# ============================================================

# MODEL

# ============================================================



with open(MODEL / "vocab.txt") as f:

    vocab = f.read().rstrip("\r\n")



V = len(vocab)

H = 256

QE = 32

BE = 8

X = 48



qmap = {c: i for i, c in enumerate(vocab)}



q_lut = np.full(256, 255, dtype=np.uint8)

for c, i in qmap.items():

    q_lut[ord(c)] = i



base_lut = np.full(256, 4, dtype=np.uint8)

for c, i in {

    "A": 0, "C": 1, "G": 2, "T": 3, "N": 4,

    "a": 0, "c": 1, "g": 2, "t": 3, "n": 4,

}.items():

    base_lut[ord(c)] = i





def matrix(name, rows, cols):

    a = np.fromfile(MODEL / name, dtype=np.float32)

    assert a.size == rows * cols

    return torch.from_numpy(

        a.reshape(rows, cols).copy()

    ).to(DEVICE)





def vector(name, n):

    a = np.fromfile(MODEL / name, dtype=np.float32)

    assert a.size == n

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





@torch.no_grad()

def quantise_freq_only(probs):



    scaled = probs * float(M)



    freq = torch.floor(

        scaled

    ).to(torch.int64)



    freq = torch.clamp(

        freq,

        min=1,

    )



    total = freq.sum(dim=1)

    delta = M - total



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



    return freq





@torch.no_grad()

def quantise_gpu(probs):



    freq = quantise_freq_only(probs)



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





# ============================================================

# READ WHOLE FASTQ

# ============================================================



print("reading whole FASTQ...", flush=True)



seqs = []

quals = []

lengths = []



source_reads = 0



with open(FASTQ) as f:



    while True:



        name = f.readline()



        if not name:

            break



        seq = f.readline().strip()

        f.readline()

        qual = f.readline().strip()



        if len(seq) != len(qual):

            raise RuntimeError(

                f"sequence/quality length mismatch: "

                f"{len(seq)} vs {len(qual)}"

            )



        source_reads += 1



        for chunk_start in range(

            0,

            len(qual),

            CHUNK,

        ):



            chunk_end = min(

                chunk_start + CHUNK,

                len(qual),

            )



            seq_chunk = seq[

                chunk_start:chunk_end

            ]



            qual_chunk = qual[

                chunk_start:chunk_end

            ]



            seqs.append(seq_chunk)

            quals.append(qual_chunk)

            lengths.append(len(qual_chunk))





lengths = np.asarray(

    lengths,

    dtype=np.int64,

)



order = np.argsort(

    -lengths,

    kind="stable",

)



lengths = lengths[order]



seqs = [

    seqs[i]

    for i in order

]



quals = [

    quals[i]

    for i in order

]



N = len(lengths)

TOTAL_SYMBOLS = int(lengths.sum())

TOTAL_TRANSITIONS = int(

    np.maximum(lengths - 1, 0).sum()

)



print("GPU:", torch.cuda.get_device_name(0))

print("SOURCE_READS:", source_reads)
print("CHUNKS:", N)

print("QUALITY_SYMBOLS:", TOTAL_SYMBOLS)

print("ENCODED_TRANSITIONS:", TOTAL_TRANSITIONS)

print("MAX_CHUNK_LENGTH:", int(lengths[0]))

print("CELL_BUDGET:", CELL_BUDGET)

print(flush=True)





# ============================================================

# GLOBAL GPU WARMUP

# ============================================================



print("warming A100...", flush=True)



WB = 65536



wh = torch.zeros(

    WB,

    H,

    dtype=torch.float32,

    device=DEVICE,

)



wq = torch.zeros(

    WB,

    dtype=torch.long,

    device=DEVICE,

)



wb = torch.zeros(

    WB,

    dtype=torch.long,

    device=DEVICE,

)



for _ in range(32):



    wh, wp = model_step(

        wh,

        wq,

        wb,

        wb,

    )



    wf = quantise_freq_only(wp)



torch.cuda.synchronize()



del wh, wq, wb, wp, wf



torch.cuda.empty_cache()



print("warmup complete", flush=True)

print(flush=True)





# ============================================================

# BATCH PARTITION

# ============================================================



def get_batch_end(i):



    T = int(lengths[i])



    by_budget = max(

        1,

        CELL_BUDGET // max(T, 1),

    )



    B = min(

        MAX_BATCH,

        int(by_budget),

        N - i,

    )



    # Isolate only the extreme long-read tail.

    # For the normal <=65k region, allow large batches and let

    # active_counts naturally shrink the population.

    if T > 65536:



        cutoff = max(

            2,

            int(T * 0.75),

        )



        block = lengths[i:i + B]



        similar = np.searchsorted(

            -block,

            -cutoff,

            side="right",

        )



        if similar > 0:

            B = min(

                B,

                int(similar),

            )



    return i + max(B, 1)





total_table_seconds = 0.0

total_rans_seconds = 0.0

total_decode_seconds = 0.0



total_encoded_bytes = 0

total_mismatches = 0

total_unconsumed = 0

processed_transitions = 0



whole_wall_start = time.perf_counter()



batch_no = 0

i = 0





# ============================================================

# PROCESS ALL READS

# ============================================================



while i < N:



    j = get_batch_end(i)



    batch_no += 1



    lens = lengths[i:j].copy()



    B = len(lens)

    T = int(lens[0])



    batch_transitions = int(

        np.maximum(lens - 1, 0).sum()

    )



    if T < 2:



        i = j

        continue



    print(

        f"===== BATCH {batch_no} ===== "

        f"reads={B} "

        f"maxlen={T} "

        f"minlen={int(lens[-1])} "

        f"transitions={batch_transitions:,}",

        flush=True,

    )





    # --------------------------------------------------------

    # Build dense uint8 batch.

    # --------------------------------------------------------



    q_np = np.zeros(

        (B, T),

        dtype=np.uint8,

    )



    b_np = np.full(

        (B, T),

        4,

        dtype=np.uint8,

    )



    for r in range(B):



        q_ascii = np.frombuffer(

            quals[i + r].encode("ascii"),

            dtype=np.uint8,

        )



        q_idx = q_lut[q_ascii]



        if np.any(q_idx == 255):

            raise RuntimeError(

                f"quality symbol outside vocab "

                f"in sorted read {i+r}"

            )



        b_ascii = np.frombuffer(

            seqs[i + r].encode("ascii"),

            dtype=np.uint8,

        )



        L = int(lens[r])



        q_np[r, :L] = q_idx

        b_np[r, :L] = base_lut[b_ascii]





    q = torch.from_numpy(

        q_np

    ).to(

        DEVICE,

        non_blocking=False,

    )



    bases = torch.from_numpy(

        b_np

    ).to(

        DEVICE,

        non_blocking=False,

    )



    del q_np, b_np





    # active[t] = number of reads with length > t+1.

    active_counts = np.searchsorted(

        -lens,

        -np.arange(

            1,

            T,

            dtype=np.int64,

        ),

        side="left",

    )





    # ========================================================

    # FORWARD MODEL + FREQUENCY TABLE GENERATION

    # ========================================================



    h = torch.zeros(

        B,

        H,

        dtype=torch.float32,

        device=DEVICE,

    )



    # Only the two values rANS needs for the known target symbol.

    #

    # Old:

    #   50 int32 values / transition = 200 bytes

    #

    # New:

    #   fs int32 + cs int32 = 8 bytes

    #

    # 25x smaller entropy-state storage.

    fs_tables = torch.empty(

        T - 1,

        B,

        dtype=torch.int32,

        device=DEVICE,

    )



    cs_tables = torch.empty(

        T - 1,

        B,

        dtype=torch.int32,

        device=DEVICE,

    )



    sym_ids = torch.arange(

        V,

        dtype=torch.int64,

        device=DEVICE,

    ).view(1, V)



    torch.cuda.synchronize()

    t0 = time.perf_counter()



    for t in range(T - 1):



        a = int(active_counts[t])



        if a == 0:

            break



        hn, probs = model_step(

            h[:a],

            q[:a, t].long(),

            bases[:a, t].long(),

            bases[:a, t + 1].long(),

        )



        h[:a] = hn



        freq = quantise_freq_only(

            probs

        )



        target = q[

            :a,

            t + 1,

        ].long()



        fs = freq.gather(

            1,

            target.unsqueeze(1),

        ).squeeze(1)



        # Exact cumulative frequency before target, without

        # materializing/storing the whole CDF.

        cs = (

            freq

            * (

                sym_ids

                < target.unsqueeze(1)

            )

        ).sum(dim=1)



        fs_tables[

            t,

            :a,

        ] = fs.to(torch.int32)



        cs_tables[

            t,

            :a,

        ] = cs.to(torch.int32)



    torch.cuda.synchronize()



    table_seconds = (

        time.perf_counter() - t0

    )



    total_table_seconds += table_seconds



    table_rate = (

        batch_transitions

        / table_seconds

    )



    print(

        "compact fs/cs pass:",

        f"{table_rate / 1e6:.3f}",

        "M symbols/s",

        flush=True,

    )



    del h

    del sym_ids





    # ========================================================

    # rANS ENCODE BACKWARDS

    # ========================================================



    MAX_BYTES = T * 4 + 64



    streams = torch.zeros(

        B,

        MAX_BYTES,

        dtype=torch.uint8,

        device=DEVICE,

    )



    ptr = torch.full(

        (B,),

        MAX_BYTES,

        dtype=torch.int64,

        device=DEVICE,

    )



    xstate = torch.full(

        (B,),

        LOWER,

        dtype=torch.int64,

        device=DEVICE,

    )



    rows = torch.arange(

        B,

        device=DEVICE,

    )



    torch.cuda.synchronize()

    t0 = time.perf_counter()



    for t in range(T - 2, -1, -1):



        a = int(active_counts[t])



        if a == 0:

            continue



        fs = fs_tables[

            t,

            :a,

        ].to(torch.int64)



        cs = cs_tables[

            t,

            :a,

        ].to(torch.int64)



        x = xstate[:a]

        p = ptr[:a]



        bound = (

            ((LOWER >> PROB_BITS) << 8)

            * fs

        )



        for _ in range(4):



            emit = x >= bound



            new_p = (

                p

                - emit.to(torch.int64)

            )



            write_pos = torch.clamp(

                new_p,

                min=0,

                max=MAX_BYTES - 1,

            )



            old = streams[

                rows[:a],

                write_pos,

            ]



            streams[

                rows[:a],

                write_pos,

            ] = torch.where(

                emit,

                (x & 0xFF).to(torch.uint8),

                old,

            )



            p = new_p



            x = torch.where(

                emit,

                x >> 8,

                x,

            )



        x = (

            ((x // fs) << PROB_BITS)

            + (x % fs)

            + cs

        )



        xstate[:a] = x

        ptr[:a] = p





    torch.cuda.synchronize()



    rans_seconds = (

        time.perf_counter() - t0

    )



    total_rans_seconds += rans_seconds



    encoded_bytes = int(

        (

            MAX_BYTES - ptr

        ).sum().item()

    )



    total_encoded_bytes += encoded_bytes



    rans_rate = (

        batch_transitions

        / rans_seconds

    )



    full_comp_seconds = (

        table_seconds

        + rans_seconds

    )



    full_comp_rate = (

        batch_transitions

        / full_comp_seconds

    )



    print(

        "rANS encode:",

        f"{rans_rate / 1e6:.3f}",

        "M symbols/s",

        flush=True,

    )



    print(

        "full compression:",

        f"{full_comp_rate / 1e6:.3f}",

        "M symbols/s",

        flush=True,

    )





    # Frequency tables are no longer needed after encode.

    del fs_tables, cs_tables



    torch.cuda.empty_cache()





    # ========================================================

    # FULL GPU DECODE

    # ========================================================



    h = torch.zeros(

        B,

        H,

        dtype=torch.float32,

        device=DEVICE,

    )



    q_current = q[

        :,

        0,

    ].long().clone()



    dx = xstate.clone()

    dptr = ptr.clone()



    mismatch_gpu = torch.zeros(

        (),

        dtype=torch.int64,

        device=DEVICE,

    )



    torch.cuda.synchronize()

    t0 = time.perf_counter()



    for t in range(T - 1):



        a = int(active_counts[t])



        if a == 0:

            break



        hn, probs = model_step(

            h[:a],

            q_current[:a],

            bases[:a, t].long(),

            bases[:a, t + 1].long(),

        )



        h[:a] = hn



        freq, cum = quantise_gpu(

            probs

        )



        slot = (

            dx[:a]

            & (M - 1)

        )



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



        x = (

            fs * (

                dx[:a] >> PROB_BITS

            )

            + slot

            - cs

        )



        p = dptr[:a]



        for _ in range(4):



            need = x < LOWER



            safe_ptr = torch.clamp(

                p,

                min=0,

                max=MAX_BYTES - 1,

            )



            byte = streams[

                rows[:a],

                safe_ptr,

            ].to(torch.int64)



            x = torch.where(

                need,

                (x << 8) | byte,

                x,

            )



            p = (

                p

                + need.to(torch.int64)

            )



        dx[:a] = x

        dptr[:a] = p



        expected = q[

            :a,

            t + 1,

        ].long()



        mismatch_gpu = (

            mismatch_gpu

            + (

                symbol != expected

            ).sum()

        )



        q_current[:a] = symbol





    torch.cuda.synchronize()



    decode_seconds = (

        time.perf_counter() - t0

    )



    total_decode_seconds += decode_seconds



    mismatches = int(

        mismatch_gpu.item()

    )



    unconsumed = int(

        (

            dptr != MAX_BYTES

        ).sum().item()

    )



    total_mismatches += mismatches

    total_unconsumed += unconsumed



    decode_rate = (

        batch_transitions

        / decode_seconds

    )



    print(

        "full decode:",

        f"{decode_rate / 1e6:.3f}",

        "M symbols/s",

        flush=True,

    )



    print(

        "mismatches:",

        mismatches,

        "unconsumed:",

        unconsumed,

        flush=True,

    )



    print(flush=True)





    processed_transitions += (

        batch_transitions

    )



    del (

        q,

        bases,

        h,

        q_current,

        dx,

        dptr,

        streams,

        ptr,

        xstate,

        rows,

        mismatch_gpu,

    )



    torch.cuda.empty_cache()



    i = j





# ============================================================

# FINAL WHOLE-FILE REPORT

# ============================================================



whole_wall_seconds = (

    time.perf_counter()

    - whole_wall_start

)



full_compression_seconds = (

    total_table_seconds

    + total_rans_seconds

)



compression_rate = (

    processed_transitions

    / full_compression_seconds

)



decode_rate = (

    processed_transitions

    / total_decode_seconds

)



payload_bpc = (

    total_encoded_bytes

    * 8.0

    / TOTAL_SYMBOLS

)



state_bytes = N * 4

seed_bytes = N

length_bytes = N * 4



codec_bytes = (

    total_encoded_bytes

    + state_bytes

    + seed_bytes

)



archive_bytes = (

    codec_bytes

    + length_bytes

)



codec_bpc = (

    codec_bytes

    * 8.0

    / TOTAL_SYMBOLS

)



archive_bpc = (

    archive_bytes

    * 8.0

    / TOTAL_SYMBOLS

)



print()

print("========================================")

print("WHOLE-FILE QUALGRU GPU RESULT")

print("========================================")



print("reads:", N)

print("quality symbols:", TOTAL_SYMBOLS)

print("encoded transitions:", processed_transitions)

print("batches:", batch_no)



print()

print("===== FULL GPU COMPRESSION =====")

print(

    "seconds:",

    f"{full_compression_seconds:.3f}",

)

print(

    "throughput:",

    f"{compression_rate / 1e6:.3f}",

    "M symbols/s",

)

print(

    "TARGET_50M:",

    "PASS"

    if compression_rate >= 50e6

    else "FAIL",

)



print()

print("===== FULL GPU DECOMPRESSION =====")

print(

    "seconds:",

    f"{total_decode_seconds:.3f}",

)

print(

    "throughput:",

    f"{decode_rate / 1e6:.3f}",

    "M symbols/s",

)

print(

    "TARGET_50M:",

    "PASS"

    if decode_rate >= 50e6

    else "FAIL",

)



print()

print("payload bytes:", total_encoded_bytes)

print("final-state bytes:", state_bytes)

print("chunk-seed bytes:", seed_bytes)

print("stream-length bytes:", length_bytes)



print(

    "payload bpc:",

    f"{payload_bpc:.4f}",

)



print(

    "codec-required quality bpc:",

    f"{codec_bpc:.4f}",

)



print(

    "self-contained archive bpc:",

    f"{archive_bpc:.4f}",

)



print(

    "mismatched symbols:",

    total_mismatches,

)



print(

    "streams not fully consumed:",

    total_unconsumed,

)



print(

    "GPU_RANS_ROUNDTRIP:",

    "PASS"

    if (

        total_mismatches == 0

        and total_unconsumed == 0

    )

    else "FAIL",

)



print()

print(

    "whole processing wall:",

    f"{whole_wall_seconds:.3f}",

    "s",

)

