import torch
import torch.nn as nn
import torch.optim as optim
from math import factorial
from decimal import Decimal, getcontext
import time

getcontext().prec = 5000
def ramanujan_digits(n_terms=400, n_digits=4000):
    t = Decimal(0)
    for k in range(n_terms):
        t += Decimal(factorial(4*k)) * Decimal(1103 + 26390*k) / \
             ((Decimal(factorial(k))**4) * Decimal(396)**(4*k))
    v = (Decimal(2).sqrt() * 2 / Decimal(9801)) * t
    return ''.join(c for c in format(v,'f') if c.isdigit())[1:n_digits+1]

DIGIT_ARRAY = torch.tensor([int(c) for c in ramanujan_digits()], dtype=torch.long)

class CondRamanujanLinear(nn.Module):
    def __init__(self, in_f, out_f, w_off, b_off, darr):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        n = in_f * out_f
        w = torch.arange(w_off, w_off + n) % len(darr)
        self.register_buffer('w_idx', darr[w])
        b = torch.arange(b_off, b_off + out_f) % len(darr)
        self.register_buffer('b_idx', darr[b])
    def forward(self, x, d_emb, b_emb, scale):
        W = scale * d_emb[self.w_idx].view(self.out_f, self.in_f)
        return x @ W.t() + b_emb[self.b_idx]

class CondOracle(nn.Module):
    def __init__(self, hidden, darr):
        super().__init__()
        self.l1 = CondRamanujanLinear(3, hidden, 0, 1000, darr)
        self.l2 = CondRamanujanLinear(hidden, 1, 500, 1400, darr)
    def forward(self, x, meta):
        h = torch.nn.functional.leaky_relu(self.l1(x, meta[0:10], meta[10:20], meta[20:21]), 0.1)
        return torch.sigmoid(self.l2(h, meta[21:31], meta[31:41], meta[41:42]))

class TaskGenerator(nn.Module):
    def __init__(self, n_tasks, h=8, out=42):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_tasks, h), nn.ReLU(), nn.Linear(h, out))
    def forward(self, oh):
        return self.net(oh)

def make_task(fn_id):
    X = torch.tensor([[float(a), float(b), float(c)]
                      for a in range(2) for b in range(2) for c in range(2)])
    bits = [(fn_id >> i) & 1 for i in range(8)]
    y = torch.tensor([[float(bits[i])] for i in range(8)])
    return X, y

def run_one(N, gen_h, seed, hidden=256, epochs=1000, lr=0.01):
    torch.manual_seed(seed)
    model = CondOracle(hidden, DIGIT_ARRAY)
    gen = TaskGenerator(N, h=gen_h, out=42)
    opt = optim.Adam(gen.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    tasks = [make_task(i) for i in range(N)]
    gen_params = sum(p.numel() for p in gen.parameters())

    best_loss = float('inf'); best_state = None
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        total = 0.0
        for i in range(N):
            oh = torch.zeros(N); oh[i] = 1.0
            meta = gen(oh.unsqueeze(0)).squeeze(0)
            total = total + loss_fn(model(tasks[i][0], meta), tasks[i][1])
        total = total / N
        total.backward()
        opt.step()
        if total.item() < best_loss:
            best_loss = total.item()
            best_state = {k: v.clone() for k, v in gen.state_dict().items()}

    gen.load_state_dict(best_state)
    with torch.no_grad():
        accs = []
        for i, (X, y) in enumerate(tasks):
            oh = torch.zeros(N); oh[i] = 1.0
            meta = gen(oh.unsqueeze(0)).squeeze(0)
            accs.append(((model(X, meta) > 0.5).float() == y).float().mean().item() * 100)

    return {
        'N': N, 'gen_h': gen_h, 'seed': seed,
        'gen_params': gen_params,
        'sep_params': N * 42,
        'ratio': (N * 42) / gen_params,
        'solved': sum(1 for a in accs if a >= 99.0),
        'min_acc': min(accs),
        'avg_acc': sum(accs)/len(accs),
        'loss': best_loss,
    }

if __name__ == "__main__":
    print("=" * 90)
    print("SEED SWEEP AT N=20 AND N=50 — do the clean-win claims survive seeds?")
    print("=" * 90)
    print(f"{'N':>3} {'gen_h':>5} {'seed':>5} {'gen_params':>11} {'ratio':>8} "
          f"{'solved':>9} {'min_acc':>8} {'avg_acc':>8} {'loss':>9} {'time':>6}")
    print("-" * 90)

    rows = []
    for N in [20, 50]:
        for seed in [42, 0, 1]:
            t0 = time.time()
            r = run_one(N=N, gen_h=8, seed=seed, epochs=1000)
            r['elapsed'] = time.time() - t0
            rows.append(r)
            print(f"{r['N']:>3} {r['gen_h']:>5} {r['seed']:>5} {r['gen_params']:>11} "
                  f"{r['ratio']:>7.2f}x {r['solved']:>4}/{N:<4} "
                  f"{r['min_acc']:>7.1f}% {r['avg_acc']:>7.1f}% "
                  f"{r['loss']:>9.4f} {r['elapsed']:>5.0f}s")

    print("-" * 90)
    print("\nAGGREGATE:")
    for N in [20, 50]:
        subset = [r for r in rows if r['N'] == N]
        solved_vals = [r['solved'] for r in subset]
        mean_solved = sum(solved_vals) / len(solved_vals)
        ratio = subset[0]['ratio']
        print(f"  N={N}: ratio={ratio:.2f}x | solved mean={mean_solved:.1f}/{N} "
              f"min={min(solved_vals)} max={max(solved_vals)}")

    print("\nVERDICT:")
    for N in [20, 50]:
        subset = [r for r in rows if r['N'] == N]
        all_full = all(r['solved'] == N for r in subset)
        any_fail = any(r['solved'] < N for r in subset)
        ratio = subset[0]['ratio']
        if all_full:
            print(f"  N={N}: ROBUST — all 3 seeds solved {N}/{N}. "
                  f"Compression claim ({ratio:.2f}x) survives seeds.")
        elif any_fail and any(r['solved'] == N for r in subset):
            print(f"  N={N}: FRAGILE — some seeds fail. Compression claim "
                  f"({ratio:.2f}x) needs seed-averaged framing.")
        else:
            print(f"  N={N}: FAILS — no seed solved all {N}. Prior single-seed "
                  f"result was overclaimed; README needs major correction.")

    print("\n  Interpretation guide:")
    print("  - ROBUST at both N=20 and N=50 = the clean-win claims are solid.")
    print("  - FRAGILE at either = README needs to say 'best of 3 seeds' not 'solved'.")
    print("  - FAILS at either = the claim was a single-seed artifact; correct immediately.")