import torch
import torch.nn as nn
import torch.optim as optim
from math import factorial
from decimal import Decimal, getcontext
import os, ast

# ============================================================
# 1. RAMANUJAN DIGITS
# ============================================================
getcontext().prec = 7000
def ramanujan_digits(num_terms, num_digits):
    total = Decimal(0)
    for k in range(num_terms):
        num = Decimal(factorial(4*k)) * Decimal(1103 + 26390*k)
        den = (Decimal(factorial(k))**4) * (Decimal(396)**(4*k))
        total += num / den
    val = (Decimal(2).sqrt() * 2 / Decimal(9801)) * total
    return ''.join(c for c in format(val,'f') if c.isdigit())[1:num_digits+1]

DIGITS = ramanujan_digits(700, 7000)
DIGIT_ARRAY = torch.tensor([int(c) for c in DIGITS], dtype=torch.long)

# ============================================================
# 2. ZERO-MAP LAYER
# ============================================================
class RamanujanLinear(nn.Module):
    def __init__(self, in_f, out_f, seed_offset, bias_offset, digit_array):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.num_weights = in_f * out_f
        self.digit_embed = nn.Parameter(torch.linspace(-1.0, 1.0, 10))
        self.bias_embed  = nn.Parameter(torch.linspace(-0.3, 0.3, 10))
        self.scale       = nn.Parameter(torch.tensor(0.5))
        w = torch.arange(seed_offset, seed_offset + self.num_weights) % len(digit_array)
        self.register_buffer('weight_digit_idx', digit_array[w])
        b = torch.arange(bias_offset, bias_offset + out_f) % len(digit_array)
        self.register_buffer('bias_digit_idx', digit_array[b])
    def forward(self, x):
        W = self.scale * self.digit_embed[self.weight_digit_idx].view(self.out_f, self.in_f)
        b = self.bias_embed[self.bias_digit_idx]
        return torch.matmul(x, W.t()) + b

class RamanujanOracle(nn.Module):
    def __init__(self, hidden, digit_array):
        super().__init__()
        self.hidden = hidden
        self.layer1 = RamanujanLinear(2, hidden, 0, 1000, digit_array)
        self.layer2 = RamanujanLinear(hidden, 1, 500, 1400, digit_array)
    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.layer1(x), 0.1)
        return torch.sigmoid(self.layer2(x))

# ============================================================
# 3. TRAIN / SAVE / LOAD
# ============================================================
X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([[0.],[1.],[1.],[0.]])

def train(model, epochs=4000, lr=0.02, clip=None):
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    for ep in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        if clip: torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        if loss.item() < 0.003:
            break
    with torch.no_grad():
        out = model(X)
        acc = ((out>0.5).float() == y).float().mean().item()*100
    return out, acc, loss_fn(out, y).item()

def save_model(model, path):
    with open(path,'w') as f:
        f.write(f"# Ramanujan Oracle — hidden={model.hidden}, activation=leaky_relu(0.1)\n")
        f.write(f"# Weights synthesized from 1/pi digits. This file IS the model.\n")
        for name,p in model.named_parameters():
            f.write(f"{name} = {[round(v,6) for v in p.detach().flatten().tolist()]}\n")
    return os.path.getsize(path)

def load_model(model, path):
    with open(path) as f:
        for line in f:
            if line.startswith('#') or '=' not in line: continue
            name, val = line.split('=',1)
            vals = ast.literal_eval(val.strip())
            parts = name.strip().split('.')
            obj = model
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], nn.Parameter(torch.tensor(vals, dtype=torch.float32)))

# ============================================================
# 4. TRAIN RAMANUJAN AT hidden=8192
# ============================================================
HIDDEN = 8192
print("=" * 80)
print(f"RAMANUJAN ORACLE — hidden={HIDDEN}, leaky ReLU(0.1), 7000-digit stream")
print("=" * 80)
torch.manual_seed(42)
m = RamanujanOracle(HIDDEN, DIGIT_ARRAY)
n_learn = sum(p.numel() for p in m.parameters())
n_synth = m.layer1.num_weights + m.layer2.num_weights
out, acc, loss = train(m, lr=0.02)
with torch.no_grad():
    h = m.layer1(X)
    dead = (h <= 0).all(dim=0).sum().item()
print(f"  learnable params : {n_learn}")
print(f"  weights synthesized : {n_synth:,} per forward pass")
print(f"  accuracy : {acc:.0f}% | loss : {loss:.4f}")
print(f"  dead neurons : {dead}/{HIDDEN} ({100*dead/HIDDEN:.1f}%)")

size = save_model(m, "ramanujan_oracle.txt")
m2 = RamanujanOracle(HIDDEN, DIGIT_ARRAY)
load_model(m2, "ramanujan_oracle.txt")
with torch.no_grad():
    out2 = m2(X)
print(f"  saved : ramanujan_oracle.txt ({size} bytes, {n_learn} numbers)")
print(f"  reload identical : {torch.allclose(out, out2, atol=1e-6)}")

# ============================================================
# 5. FAIR BASELINE SEARCH AT THE SAME WIDTH
# ============================================================
print("\n" + "=" * 80)
print(f"FAIR BASELINE — searching for a working MLP at hidden={HIDDEN}")
print("=" * 80)
best_baseline = None
for lr in [0.05, 0.02, 0.01, 0.005, 0.002]:
    torch.manual_seed(42)
    bl = nn.Sequential(nn.Linear(2,HIDDEN), nn.ReLU(), nn.Linear(HIDDEN,1), nn.Sigmoid())
    bl_n = sum(p.numel() for p in bl.parameters())
    _, bl_acc, _ = train(bl, lr=lr, clip=1.0)  # gradient clip keeps it stable
    print(f"  MLP hidden={HIDDEN}, lr={lr:.3f} -> acc={bl_acc:.0f}%")
    if bl_acc == 100.0 and best_baseline is None:
        best_baseline = bl_n

# ============================================================
# 6. HONEST COMPARISON
# ============================================================
print("\n" + "=" * 80)
print("THE SHOCK (honest version)")
print("=" * 80)
# Minimal working MLP for XOR: 2 hidden units -> 7 params
torch.manual_seed(0)
min_bl = nn.Sequential(nn.Linear(2,2), nn.ReLU(), nn.Linear(2,1), nn.Sigmoid())
min_n = sum(p.numel() for p in min_bl.parameters())
min_acc = 0
for lr in [0.5, 0.2, 0.1, 0.05]:
    torch.manual_seed(0)
    min_bl = nn.Sequential(nn.Linear(2,2), nn.ReLU(), nn.Linear(2,1), nn.Sigmoid())
    _, min_acc, _ = train(min_bl, lr=lr, epochs=8000)
    if min_acc == 100.0: break

print(f"  Minimum working MLP (2 hidden) : {min_n:5d} params | acc={min_acc:.0f}%")
print(f"  Matched-width MLP (hidden={HIDDEN}) : {best_baseline if best_baseline else 'FAILED':>5} params")
print(f"  Ramanujan (hidden={HIDDEN})      : {n_learn:5d} params | acc={acc:.0f}%")
print()
print(f"  At width {HIDDEN}, the standard MLP needs "
      f"{best_baseline if best_baseline else '?'} trainable numbers.")
print(f"  The Ramanujan Oracle needs {n_learn}. Weights are synthesized from 1/pi.")
if best_baseline:
    print(f"  Compression at this width : {best_baseline/n_learn:.1f}x")
print(f"  Model file : {size} bytes, containing {n_learn} numbers")
print(f"  The trained network fits in a text file. The weights do not exist.")