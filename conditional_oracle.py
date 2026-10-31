import torch
import torch.nn as nn
import torch.optim as optim
from math import factorial
from decimal import Decimal, getcontext

# ============================================================
# 1. RAMANUJAN DIGIT STREAM
# ============================================================
getcontext().prec = 5000

def ramanujan_digits(num_terms=400, num_digits=4000):
    total = Decimal(0)
    for k in range(num_terms):
        total += Decimal(factorial(4*k)) * Decimal(1103 + 26390*k) / \
                 ((Decimal(factorial(k))**4) * Decimal(396)**(4*k))
    val = (Decimal(2).sqrt() * 2 / Decimal(9801)) * total
    return ''.join(c for c in format(val, 'f') if c.isdigit())[1:num_digits+1]

DIGITS = ramanujan_digits()
DIGIT_ARRAY = torch.tensor([int(c) for c in DIGITS], dtype=torch.long)
print(f"Ramanujan stream: {len(DIGITS)} digits of 1/pi\n")

# ============================================================
# 2. CONDITIONAL ZERO-MAP LAYER
# ============================================================
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
        self.hidden = hidden
        self.l1 = CondRamanujanLinear(3, hidden, 0, 1000, darr)
        self.l2 = CondRamanujanLinear(hidden, 1, 500, 1400, darr)

    def forward(self, x, meta):
        h = self.l1(x, meta[0:10], meta[10:20], meta[20:21])
        h = torch.nn.functional.leaky_relu(h, 0.1)
        return torch.sigmoid(self.l2(h, meta[21:31], meta[31:41], meta[41:42]))

# ============================================================
# 3. THE TASK-CONDITIONAL GENERATOR
# ============================================================
class TaskGenerator(nn.Module):
    def __init__(self, n_tasks, h=8, out=42):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_tasks, h),
            nn.ReLU(),
            nn.Linear(h, out),
        )
    def forward(self, onehot):
        return self.net(onehot)

# ============================================================
# 4. TASKS (all use 3-dim input; 2-input tasks padded with 0)
# ============================================================
def make_task(name):
    base2 = [[0.,0.,0.],[0.,1.,0.],[1.,0.,0.],[1.,1.,0.]]
    if name == 'xor':     return torch.tensor(base2), torch.tensor([[0.],[1.],[1.],[0.]])
    if name == 'and':     return torch.tensor(base2), torch.tensor([[0.],[0.],[0.],[1.]])
    if name == 'or':      return torch.tensor(base2), torch.tensor([[0.],[1.],[1.],[1.]])
    if name == 'xnor':    return torch.tensor(base2), torch.tensor([[1.],[0.],[0.],[1.]])
    if name == 'parity3':
        X = torch.tensor([[float(a),float(b),float(c)]
                          for a in range(2) for b in range(2) for c in range(2)])
        return X, (X.sum(1, keepdim=True) % 2)
    raise ValueError(name)

TRAIN = ['xor', 'and', 'or', 'parity3']
HELD = 'xnor'

# ============================================================
# 5. EVALUATION
# ============================================================
def evaluate(model, gen, tasks):
    results = {}
    with torch.no_grad():
        for i, name in enumerate(tasks):
            X, y = make_task(name)
            oh = torch.zeros(len(tasks)); oh[i] = 1.0
            meta = gen(oh.unsqueeze(0)).squeeze(0)
            out = model(X, meta)
            results[name] = ((out > 0.5).float() == y).float().mean().item() * 100
    return results

def eval_held(model, gen, held):
    with torch.no_grad():
        X, y = make_task(held)
        best_acc, best_task = 0, None
        for i, name in enumerate(TRAIN):
            oh = torch.zeros(len(TRAIN)); oh[i] = 1.0
            meta = gen(oh.unsqueeze(0)).squeeze(0)
            out = model(X, meta)
            acc = ((out > 0.5).float() == y).float().mean().item() * 100
            if acc > best_acc:
                best_acc, best_task = acc, name
        return best_acc, best_task

# ============================================================
# 6. TRAINING
# ============================================================
def train(hidden=256, epochs=3000, lr=0.01):
    torch.manual_seed(42)
    model = CondOracle(hidden, DIGIT_ARRAY)
    gen = TaskGenerator(len(TRAIN), h=8, out=42)
    opt = optim.Adam(gen.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    Xs, ys = zip(*[make_task(n) for n in TRAIN])

    best_loss, best_state = float('inf'), None
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        total = 0.0
        for i in range(len(TRAIN)):
            oh = torch.zeros(len(TRAIN)); oh[i] = 1.0
            meta = gen(oh.unsqueeze(0)).squeeze(0)
            out = model(Xs[i], meta)
            total = total + loss_fn(out, ys[i])
        total = total / len(TRAIN)
        total.backward()
        opt.step()
        if total.item() < best_loss:
            best_loss = total.item()
            best_state = {k: v.clone() for k, v in gen.state_dict().items()}
        if ep % 500 == 0 or ep == 1:
            accs = evaluate(model, gen, TRAIN)
            line = ' | '.join(f"{k}={v:3.0f}%" for k, v in accs.items())
            print(f"  ep {ep:4d} | loss {total.item():.4f} | {line}")
    if best_state:
        gen.load_state_dict(best_state)
    return model, gen, best_loss

# ============================================================
# 7. MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 72)
    print("CONDITIONAL ZERO-MAP WEIGHT GENERATION — Stage 1")
    print("=" * 72)
    print(f"Train tasks : {TRAIN}")
    print(f"Held out    : {HELD}\n")

    HIDDEN = 256
    n_synth = 3 * HIDDEN + HIDDEN
    gen_params = 4 * 8 + 8 + 8 * 42 + 42
    sep_params = 4 * 42

    print(f"Target network    : hidden={HIDDEN}, synthesized weights/task = {n_synth:,}")
    print(f"Generator         : 4 (task) -> 8 -> 42 | params = {gen_params}")
    print(f"4 separate models : {sep_params} params")
    print(f"Ratio             : {gen_params/sep_params:.1f}x more params for generator")
    print(f"(This tests MECHANISM, not efficiency — see verdict)\n")

    print("-" * 72)
    print("TRAINING")
    print("-" * 72)
    model, gen, loss = train(hidden=HIDDEN, epochs=3000, lr=0.01)

    print("\n" + "-" * 72)
    print("RESULTS ON TRAINING TASKS")
    print("-" * 72)
    accs = evaluate(model, gen, TRAIN)
    for name, acc in accs.items():
        mark = "OK" if acc >= 99.0 else "FAIL"
        print(f"  [{mark:4s}] {name:10s} : {acc:5.1f}%")
    all_ok = all(a >= 99.0 for a in accs.values())

    print("\n" + "-" * 72)
    print(f"HELD-OUT TEST ({HELD}) — does the generator generalize?")
    print("-" * 72)
    best_acc, best_task = eval_held(model, gen, HELD)
    print(f"  Best result: {best_acc:.0f}% (using embedding trained on '{best_task}')")
    if best_acc >= 99.0:
        print("  ** GENERALIZATION SUCCESS **")
        print("  (Unexpected for Stage 1 — this would be a strong result)")
    else:
        print("  (Expected — Stage 1 does not train for generalization)")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    if all_ok:
        print("  Conditional mechanism WORKS.")
        print(f"  One {gen_params}-param generator produced working weight sets")
        print(f"  for {len(TRAIN)} tasks, each with {n_synth:,} synthesized weights.")
        print()
        print("  Honest interpretation:")
        print(f"  - {gen_params} params > {sep_params} params (4 separate models).")
        print( "  - Conditional generation does NOT save parameters at 4 tasks.")
        print( "  - It WOULD save parameters at 100+ tasks (generator grows slowly).")
        print( "  - The real win: ONE mechanism for N tasks, with a path to")
        print( "    generalization via continuous task embeddings (Stage 3).")
        print()
        print("  Next experiments:")
        print( "  1. Replace one-hot with continuous task descriptor.")
        print( "  2. Scale to 20+ tasks — does generator beat separate models?")
        print( "  3. Try smaller generator (h=4 or h=2) — how small can it get?")
    else:
        failed = [n for n, a in accs.items() if a < 99.0]
        print(f"  PARTIAL SUCCESS — failed on: {failed}")
        print()
        print("  Tuning options:")
        print("  - Increase generator hidden (h=8 -> 16)")
        print("  - More epochs (3000 -> 8000)")
        print("  - Different lr (0.01 -> 0.005)")
        print("  - Larger target network (hidden=256 -> 512)")