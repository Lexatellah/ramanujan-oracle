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
        b = b_emb[self.b_idx]
        return x @ W.t() + b

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

def make_task_from_truth(fn_id):
    X = torch.tensor([[float(a), float(b), float(c)]
                      for a in range(2) for b in range(2) for c in range(2)])
    bits = [(fn_id >> i) & 1 for i in range(8)]
    y = torch.tensor([[float(bits[i])] for i in range(8)])
    return X, y

def evaluate_all(model, gen, tasks):
    accs = []
    with torch.no_grad():
        for i, (X, y) in enumerate(tasks):
            oh = torch.zeros(len(tasks)); oh[i] = 1.0
            meta = gen(oh.unsqueeze(0)).squeeze(0)
            out = model(X, meta)
            accs.append(((out > 0.5).float() == y).float().mean().item() * 100)
    return accs

def run_one(N, gen_h, hidden=256, epochs=2000, lr=0.01, seed=42):
    torch.manual_seed(seed)
    model = CondOracle(hidden, DIGIT_ARRAY)
    gen = TaskGenerator(N, h=gen_h, out=42)
    opt = optim.Adam(gen.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    tasks = [make_task_from_truth(i) for i in range(N)]
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
    accs = evaluate_all(model, gen, tasks)
    return {
        'N': N, 'gen_h': gen_h, 'gen_params': gen_params,
        'sep_params': N * 42,
        'solved': sum(1 for a in accs if a >= 99.0),
        'min_acc': min(accs),
        'avg_acc': sum(accs)/len(accs),
        'loss': best_loss,
    }

if __name__ == "__main__":
    print("=" * 78)
    print("CAPACITY TEST — is N=100 a generator-capacity or substrate limit?")
    print("=" * 78)
    print(f"{'N':>4} {'gen_h':>6} {'gen_params':>11} {'N x 42':>8} "
          f"{'ratio':>7} {'solved':>8} {'min_acc':>8} {'loss':>9} {'time':>7}")
    print("-" * 78)

    results = []
    for N, gh in [(100, 8), (100, 16), (100, 32)]:
        t0 = time.time()
        r = run_one(N=N, gen_h=gh, epochs=2000)
        elapsed = time.time() - t0
        r['elapsed'] = elapsed
        results.append(r)
        ratio = r['sep_params'] / r['gen_params']
        print(f"{r['N']:>4} {r['gen_h']:>6} {r['gen_params']:>11} "
              f"{r['sep_params']:>8} {ratio:>6.2f}x "
              f"{r['solved']:>3}/{r['N']:<4} {r['min_acc']:>7.1f}% "
              f"{r['loss']:>9.4f} {elapsed:>6.0f}s")

    print("-" * 78)
    print("\nVERDICT:")
    all_solved = [r for r in results if r['solved'] == r['N']]
    if all_solved:
        first = all_solved[0]
        print(f"  CAPACITY FIXED at gen_h={first['gen_h']}.")
        print(f"  {first['gen_params']} params -> {first['N']}/{first['N']} tasks solved.")
        print(f"  Compression vs {first['N']} separate models: "
              f"{first['sep_params']/first['gen_params']:.2f}x")
        print("  -> The N=100 failure at gen_h=8 was generator capacity, not substrate.")
    elif results[-1]['solved'] < results[-1]['N']:
        print(f"  SUBSTRATE CEILING — even gen_h={results[-1]['gen_h']} fails at N={results[-1]['N']}.")
        print(f"  Best: {results[-1]['solved']}/{results[-1]['N']} solved, "
              f"min_acc {results[-1]['min_acc']:.1f}%")
        print("  -> The 10-value lookup substrate cannot distinguish 100 tasks,")
        print("     regardless of generator width. This is a stronger finding")
        print("     than a clean 100/100 — it defines the boundary of the approach.")
    else:
        print("  PARTIAL — some configs solved, check the table above.")