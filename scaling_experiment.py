import torch
import torch.nn as nn
import torch.optim as optim
from math import factorial
from decimal import Decimal, getcontext

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

# --- Generate N synthetic tasks: each is a distinct 3-bit boolean function ---
def make_task_from_truth(fn_id):
    X = torch.tensor([[float(a), float(b), float(c)]
                      for a in range(2) for b in range(2) for c in range(2)])
    # Distinct boolean function based on fn_id bits
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

def run(n_tasks, gen_h=8, hidden=256, epochs=2000, lr=0.01, seed=42):
    torch.manual_seed(seed)
    model = CondOracle(hidden, DIGIT_ARRAY)
    gen = TaskGenerator(n_tasks, h=gen_h, out=42)
    opt = optim.Adam(gen.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    # pick n_tasks distinct boolean functions
    task_ids = list(range(n_tasks))
    tasks = [make_task_from_truth(i) for i in task_ids]

    gen_params = sum(p.numel() for p in gen.parameters())
    sep_params = n_tasks * 42
    n_synth = 3 * hidden + hidden

    best_loss = float('inf'); best_state = None
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        total = 0.0
        for i in range(n_tasks):
            oh = torch.zeros(n_tasks); oh[i] = 1.0
            meta = gen(oh.unsqueeze(0)).squeeze(0)
            total = total + loss_fn(model(tasks[i][0], meta), tasks[i][1])
        total = total / n_tasks
        total.backward()
        opt.step()
        if total.item() < best_loss:
            best_loss = total.item()
            best_state = {k: v.clone() for k, v in gen.state_dict().items()}

    gen.load_state_dict(best_state)
    accs = evaluate_all(model, gen, tasks)
    n_solved = sum(1 for a in accs if a >= 99.0)

    return {
        'n_tasks': n_tasks,
        'gen_params': gen_params,
        'sep_params': sep_params,
        'n_synth': n_synth,
        'solved': n_solved,
        'min_acc': min(accs),
        'avg_acc': sum(accs)/len(accs),
        'loss': best_loss,
    }

if __name__ == "__main__":
    print("=" * 78)
    print("SCALING EXPERIMENT — does the generator beat N separate models?")
    print("=" * 78)
    print(f"{'N':>4} {'gen_params':>11} {'N x 42':>8} {'ratio':>7} "
          f"{'solved':>7} {'min_acc':>8} {'avg_acc':>8} {'loss':>9}")
    print("-" * 78)

    results = []
    for n in [4, 10, 20, 50, 100]:
        r = run(n_tasks=n, gen_h=8, hidden=256, epochs=2000)
        ratio = r['sep_params'] / r['gen_params']
        winner = 'gen' if ratio > 1.0 else 'sep'
        print(f"{r['n_tasks']:>4} {r['gen_params']:>11} {r['sep_params']:>8} "
              f"{ratio:>6.2f}x {r['solved']:>3}/{n:<3} {r['min_acc']:>7.1f}% "
              f"{r['avg_acc']:>7.1f}% {r['loss']:>9.4f}  [{winner}]")
        results.append(r)

    print("-" * 78)
    print("\nVERDICT:")
    crossover = next((r['n_tasks'] for r in results
                      if r['sep_params'] > r['gen_params']), None)
    if crossover:
        print(f"  Crossover at N={crossover} tasks: generator becomes smaller.")
    else:
        print("  No crossover in tested range.")
    print()
    for r in results:
        if r['solved'] == r['n_tasks'] and r['sep_params'] > r['gen_params']:
            print(f"  N={r['n_tasks']:>3}: generator {r['gen_params']} params vs "
                  f"{r['sep_params']} separate = {r['sep_params']/r['gen_params']:.2f}x smaller, "
                  f"all {r['n_tasks']} tasks solved.")