

import time

import torch

import triton

import triton.language as tl



M = 65536

V = 50

BLOCK = 64





@triton.jit

def quant_kernel(

    probs_ptr,

    freq_ptr,

    n_rows: tl.constexpr,

    V: tl.constexpr,

    BLOCK: tl.constexpr,

):

    row = tl.program_id(0)

    offs = tl.arange(0, BLOCK)

    mask = offs < V



    p = tl.load(

        probs_ptr + row * V + offs,

        mask=mask,

        other=0.0,

    )



    scaled = p * 65536.0



    # Triton 2.0 has no tl.floor.

    # scaled is nonnegative, so float->int truncation == floor.

    freq = scaled.to(tl.int32)

    floored = freq.to(tl.float32)

    freq = tl.where(mask, freq, 0)

    freq = tl.where(mask & (freq < 1), 1, freq)



    total = tl.sum(freq, axis=0)



    deficit = 65536 - total

    deficit = tl.maximum(deficit, 0)



    frac = scaled - floored

    frac = tl.where(mask, frac, -2.0)



    # Same largest-remainder idea as current GPU quantiser.

    # Pick at most V highest fractional parts.

    for k in range(50):

        idx = tl.argmax(frac, axis=0)



        take = k < deficit

        selected = offs == idx



        freq += (selected & take).to(tl.int32)



        frac = tl.where(

            selected,

            -2.0,

            frac,

        )



    # Handle possible overfull row from min-frequency clamping.

    total2 = tl.sum(freq, axis=0)



    excess = total2 - 65536

    excess = tl.maximum(excess, 0)



    biggest = tl.argmax(freq, axis=0)



    freq -= (

        (offs == biggest).to(tl.int32)

        * excess

    )



    tl.store(

        freq_ptr + row * V + offs,

        freq,

        mask=mask,

    )





@torch.no_grad()

def torch_quant(probs):

    scaled = probs * float(M)

    floored = torch.floor(scaled)



    freq = floored.to(torch.int64)

    freq = torch.clamp(freq, min=1)



    total = freq.sum(dim=1)

    deficit = torch.clamp(M - total, min=0)



    frac = scaled - floored



    order = torch.argsort(

        frac,

        dim=1,

        descending=True,

    )



    ranks = torch.arange(

        V,

        device=probs.device,

    ).unsqueeze(0)



    add_rank = (

        ranks < deficit.unsqueeze(1)

    ).to(torch.int64)



    additions = torch.zeros_like(freq)

    additions.scatter_(1, order, add_rank)

    freq += additions



    excess = torch.clamp(

        freq.sum(dim=1) - M,

        min=0,

    )



    biggest = torch.argmax(

        freq,

        dim=1,

        keepdim=True,

    )



    freq.scatter_add_(

        1,

        biggest,

        -excess.unsqueeze(1),

    )



    return freq.to(torch.int32)





print("GPU:", torch.cuda.get_device_name(0))

print("TRITON:", triton.__version__)



for B in [4096, 16384, 32768, 65536]:



    torch.manual_seed(1234)



    logits = torch.randn(

        B, V,

        device="cuda",

        dtype=torch.float32,

    )



    probs = torch.softmax(logits, dim=1)



    ref = torch_quant(probs)



    out = torch.empty(

        B, V,

        device="cuda",

        dtype=torch.int32,

    )



    grid = (B,)



    quant_kernel[grid](

        probs,

        out,

        B,

        V=V,

        BLOCK=BLOCK,

    )



    torch.cuda.synchronize()



    bad = (out != ref).sum().item()



    print()

    print("BATCH", B)

    print("mismatched frequency entries:", bad)



    # Warmup

    for _ in range(20):

        quant_kernel[grid](

            probs,

            out,

            B,

            V=V,

            BLOCK=BLOCK,

        )



    torch.cuda.synchronize()



    start = torch.cuda.Event(enable_timing=True)

    end = torch.cuda.Event(enable_timing=True)



    N = 200



    start.record()



    for _ in range(N):

        quant_kernel[grid](

            probs,

            out,

            B,

            V=V,

            BLOCK=BLOCK,

        )



    end.record()

    torch.cuda.synchronize()



    ms = start.elapsed_time(end) / N



    rate = B / (ms / 1000.0)



    print(

        f"Triton quantiser: "

        f"{ms:.4f} ms/step  "

        f"{rate/1e6:.3f} M tables/s"

    )



    # Benchmark current PyTorch implementation too.

    for _ in range(5):

        tmp = torch_quant(probs)



    torch.cuda.synchronize()



    start.record()



    for _ in range(20):

        tmp = torch_quant(probs)



    end.record()

    torch.cuda.synchronize()



    torch_ms = start.elapsed_time(end) / 20

    torch_rate = B / (torch_ms / 1000.0)



    print(

        f"PyTorch quantiser: "

        f"{torch_ms:.4f} ms/step  "

        f"{torch_rate/1e6:.3f} M tables/s"

    )



    print(

        f"speedup: {torch_ms/ms:.2f}x"

    )

