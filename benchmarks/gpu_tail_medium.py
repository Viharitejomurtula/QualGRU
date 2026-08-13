
import time

import torch



torch.backends.cuda.matmul.allow_tf32 = True

torch.backends.cudnn.allow_tf32 = True



DEVICE = "cuda"

H = 64

X = 48

V = 50



torch.manual_seed(123)



# Same dimensions as QualGRU h64.

Wru = torch.randn(2 * H, H, device=DEVICE) * 0.05

Uru = torch.randn(2 * H, X, device=DEVICE) * 0.05

bru = torch.randn(2 * H, device=DEVICE) * 0.01



Wc = torch.randn(H, H, device=DEVICE) * 0.05

Uc = torch.randn(H, X, device=DEVICE) * 0.05

bc = torch.randn(H, device=DEVICE) * 0.01



Wo = torch.randn(V, H, device=DEVICE) * 0.05

bo = torch.randn(V, device=DEVICE) * 0.01





def step(h, x):

    g = h @ Wru.T + x @ Uru.T + bru

    r, u = torch.sigmoid(g).chunk(2, dim=1)



    c = torch.tanh(

        (r * h) @ Wc.T +

        x @ Uc.T +

        bc

    )



    h = (1.0 - u) * h + u * c



    logits = h @ Wo.T + bo

    probs = torch.softmax(logits, dim=1)



    return h, probs





def make_runner(K):

    def run(h, xs):

        probs = None



        for t in range(K):

            h, probs = step(h, xs[t])



        return h, probs



    return run





def bench(fn, h, xs, K, loops=30):

    # Warmup

    for _ in range(10):

        fn(h, xs)



    torch.cuda.synchronize()



    start = torch.cuda.Event(enable_timing=True)

    end = torch.cuda.Event(enable_timing=True)



    start.record()



    for _ in range(loops):

        fn(h, xs)



    end.record()



    torch.cuda.synchronize()



    ms = start.elapsed_time(end) / loops



    symbols = h.shape[0] * K

    rate = symbols / (ms / 1000.0)



    return ms, rate





print("GPU:", torch.cuda.get_device_name(0))

print("Torch:", torch.__version__)

print("TF32:", torch.backends.cuda.matmul.allow_tf32)



BATCHES = [256, 1024, 4096]

KS = [1, 8, 32, 128]



for B in BATCHES:

    print()

    print("=" * 60)

    print("BATCH:", B)



    h = torch.zeros(B, H, device=DEVICE)



    for K in KS:

        xs = torch.randn(K, B, X, device=DEVICE)



        eager = make_runner(K)



        compiled = torch.compile(

            make_runner(K),

            mode="reduce-overhead",

            fullgraph=True,

        )



        # Trigger compilation.

        compiled(h, xs)

        torch.cuda.synchronize()



        eager_ms, eager_rate = bench(

            eager, h, xs, K

        )



        compiled_ms, compiled_rate = bench(

            compiled, h, xs, K

        )



        print()

        print(f"K={K}")

        print(

            f"eager:    {eager_ms:.4f} ms/call  "

            f"{eager_rate/1e6:.3f} M sym/s"

        )

        print(

            f"compiled: {compiled_ms:.4f} ms/call  "

            f"{compiled_rate/1e6:.3f} M sym/s"

        )

        print(

            f"speedup:  {compiled_rate/eager_rate:.2f}x"

        )



        if compiled_rate >= 5e6:

            verdict = "PASS_5M"

        elif compiled_rate >= 2.2e6:

            verdict = "PASS_MIN"

        else:

            verdict = "FAIL"



        print("TAIL_TARGET:", verdict)

