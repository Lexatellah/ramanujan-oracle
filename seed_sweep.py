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
        'solved': sum(1 for a in accs if a >= 99.0),
        'min_acc': min(accs),
        'avg_acc': sum(accs)/len(accs),
        'loss': best_loss,
    }

if __name__ == "__main__":
    print("=" * 84)
    print("SEED SWEEP — is the 98 -> 95 non-monotonicity real or noise?")
    print("=" * 84)
    print(f"{'gen_h':>6} {'seed':>5} {'gen_params':>11} {'solved':>9} "
          f"{'min_acc':>8} {'avg_acc':>8} {'loss':>9} {'time':>7}")
    print("-" * 84)

    rows = []
    for gen_h in [8, 16, 32]:
        for seed in [42, 0, 1]:
            t0 = time.time()
            r = run_one(N=100, gen_h=gen_h, seed=seed, epochs=1000)
            elapsed = time.time() - t0
            r['elapsed'] = elapsed
            rows.append(r)
            print(f"{r['gen_h']:>6} {r['seed']:>5} {r['gen_params']:>11} "
                  f"{r['solved']:>4}/100  {r['min_acc']:>7.1f}% "
                  f"{r['avg_acc']:>7.1f}% {r['loss']:>9.4f} {elapsed:>6.0f}s")

    print("-" * 84)
    print("\nAGGREGATE PER gen_h:")
    for gen_h in [8, 16, 32]:
        subset = [r for r in rows if r['gen_h'] == gen_h]
        solved_vals = [r['solved'] for r in subset]
        mean_solved = sum(solved_vals) / len(solved_vals)
        min_solved = min(solved_vals)
        max_solved = max(solved_vals)
        print(f"  gen_h={gen_h:>2}: solved mean={mean_solved:>5.1f}/100 "
              f"min={min_solved:>3} max={max_solved:>3}")

    print("\nVERDICT:")
    mean_8 = sum(r['solved'] for r in rows if r['gen_h']==8) / 3
    mean_16 = sum(r['solved'] for r in rows if r['gen_h']==16) / 3
    mean_32 = sum(r['solved'] for r in rows if r['gen_h']==32) / 3

    if mean_16 < mean_8 - 2:
        print(f"  NON-MONOTONICITY CONFIRMED. gen_h=16 mean ({mean_16:.1f}) < gen_h=8 mean ({mean_8:.1f}).")
        print("  This is a real optimization pathology, not seed noise.")
    elif mean_16 > mean_8 + 2:
        print(f"  MONOTONIC. gen_h=16 mean ({mean_16:.1f}) > gen_h=8 mean ({mean_8:.1f}).")
        print("  Prior 98->95 result was seed noise. Current README needs correction.")
    else:
        print(f"  INCONCLUSIVE. gen_h=8 mean={mean_8:.1f}, gen_h=16 mean={mean_16:.1f}.")
        print("  Difference within seed noise. Need more seeds or a different test.")

    if mean_32 >= 99:
        print(f"\n  gen_h=32 robustly solves N=100 (mean {mean_32:.1f}/100).")
    else:
        print(f"\n  gen_h=32 also fails sometimes (mean {mean_32:.1f}/100).")
        print("  Suggests genuine architectural limit, not just generator width.")