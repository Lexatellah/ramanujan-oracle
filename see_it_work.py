import torch
import torch.nn as nn
import torch.optim as optim
from math import factorial
from decimal import Decimal, getcontext

getcontext().prec = 2000
def ramanujan_digits(n_terms, n_digits):
    t = Decimal(0)
    for k in range(n_terms):
        t += Decimal(factorial(4*k)) * Decimal(1103 + 26390*k) / \
             ((Decimal(factorial(k))**4) * Decimal(396)**(4*k))
    v = (Decimal(2).sqrt() * 2 / Decimal(9801)) * t
    return ''.join(c for c in format(v,'f') if c.isdigit())[1:n_digits+1]

DIGITS = ramanujan_digits(300, 1500)
DIGIT_ARRAY = torch.tensor([int(c) for c in DIGITS], dtype=torch.long)

class RLinear(nn.Module):
    def __init__(self, in_f, out_f, seed_off, bias_off, darr):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.num_weights = in_f * out_f
        self.digit_embed = nn.Parameter(torch.linspace(-1,1,10))
        self.bias_embed  = nn.Parameter(torch.linspace(-0.3,0.3,10))
        self.scale       = nn.Parameter(torch.tensor(0.5))
        w = torch.arange(seed_off, seed_off + self.num_weights) % len(darr)
        self.register_buffer('w_idx', darr[w])
        b = torch.arange(bias_off, bias_off + out_f) % len(darr)
        self.register_buffer('b_idx', darr[b])
    def forward(self, x):
        W = self.scale * self.digit_embed[self.w_idx].view(self.out_f, self.in_f)
        b = self.bias_embed[self.b_idx]
        return x @ W.t() + b

class Oracle(nn.Module):
    def __init__(self, hidden, darr):
        super().__init__()
        self.hidden = hidden
        self.layer1 = RLinear(2, hidden, 0, 1000, darr)
        self.layer2 = RLinear(hidden, 1, 500, 1400, darr)
    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.layer1(x), 0.1)
        return torch.sigmoid(self.layer2(x))

X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([[0.],[1.],[1.],[0.]])

torch.manual_seed(42)
m = Oracle(256, DIGIT_ARRAY)
opt = optim.Adam(m.parameters(), lr=0.02)
for _ in range(2000):
    opt.zero_grad(); loss = nn.BCELoss()(m(X), y); loss.backward(); opt.step()
    if loss.item() < 0.003: break
print(f"Trained. acc = {((m(X)>0.5).float()==y).float().mean()*100:.0f}%\n")

# ---- DEMO 1: How one weight is born ----
print("=" * 70)
print("DEMO 1: How weight[0,0] is created")
print("=" * 70)
i0 = m.layer1.w_idx[0].item()
emb = m.layer1.digit_embed.detach()
scale = m.layer1.scale.item()
W_actual = (m.layer1.scale * m.layer1.digit_embed[m.layer1.w_idx].view(m.hidden, 2)).detach()
print(f"  First digit of 1/pi in our stream  : '{DIGITS[0]}'")
print(f"  Learned value for digit '{i0}'       : {emb[i0].item():+.4f}")
print(f"  Learned scale                      : {scale:.4f}")
print(f"  weight[0,0] = scale x embed        : {scale * emb[i0].item():+.4f}")
print(f"  Actual weight[0,0] in the matrix   : {W_actual[0,0].item():+.4f}   <- match\n")

# ---- DEMO 2: Only 10 distinct values ----
print("=" * 70)
print("DEMO 2: The weight matrix has only 10 distinct values")
print("=" * 70)
distinct = torch.unique(W_actual.round(decimals=5))
print(f"  Layer-1 weight matrix shape        : {tuple(W_actual.shape)}")
print(f"  Total weights                      : {W_actual.numel()}")
print(f"  Distinct values present            : {len(distinct)}")
print(f"  (Every weight is one of: scale x embed[d] for d in 0..9)")
print(f"  Distinct values: {[round(v,3) for v in distinct.tolist()][:10]}\n")

# ---- DEMO 3: Determinism ----
print("=" * 70)
print("DEMO 3: Same digits -> same weights, every time")
print("=" * 70)
W_again = (m.layer1.scale * m.layer1.digit_embed[m.layer1.w_idx].view(m.hidden, 2)).detach()
print(f"  Weights identical after recompute  : {torch.allclose(W_actual, W_again)}\n")

# ---- DEMO 4: One learnable number changes ALL weights ----
print("=" * 70)
print("DEMO 4: Perturbing one of 42 learnable numbers")
print("=" * 70)
before = W_actual.clone()
with torch.no_grad():
    m.layer1.digit_embed[5] += 0.5  # nudge the "meaning" of digit 5
after = (m.layer1.scale * m.layer1.digit_embed[m.layer1.w_idx].view(m.hidden, 2)).detach()
changed = (after != before).sum().item()
total = before.numel()
print(f"  We changed 1 number: embedding for digit '5'")
print(f"  Weights that changed: {changed} / {total} ({100*changed/total:.1f}%)")
print(f"  This is why 42 numbers can steer 6,000+ weights.\n")

# ---- DEMO 5: What a standard MLP looks like ----
print("=" * 70)
print("DEMO 5: Contrast with a standard MLP")
print("=" * 70)
torch.manual_seed(0)
mlp = nn.Sequential(nn.Linear(2,256), nn.ReLU(), nn.Linear(256,1), nn.Sigmoid())
opt = optim.Adam(mlp.parameters(), lr=0.05)
for _ in range(2000):
    opt.zero_grad(); l = nn.BCELoss()(mlp(X), y); l.backward(); opt.step()
    if l.item() < 0.003: break
W_mlp = mlp[0].weight.detach()
print(f"  Standard MLP layer-1 weights       : {W_mlp.numel()}")
print(f"  Distinct values                    : {len(torch.unique(W_mlp.round(decimals=5)))}")
print(f"  Every value is independent         : gradients only touch the weight they flow through")
print(f"  Learnable params                   : {sum(p.numel() for p in mlp.parameters())}\n")

# ---- TEST: Can it do a harder task? ----
print("=" * 70)
print("TEST: 3-bit parity (8 examples, harder than XOR)")
print("=" * 70)
X3 = torch.tensor([[float(i),float(j),float(k)] for i in range(2) for j in range(2) for k in range(2)])
y3 = (X3.sum(dim=1, keepdim=True) % 2)

# Our model needs 3 inputs now; adjust layer1
class Oracle3(nn.Module):
    def __init__(self, hidden, darr):
        super().__init__()
        self.hidden = hidden
        self.layer1 = RLinear(3, hidden, 0, 1000, darr)
        self.layer2 = RLinear(hidden, 1, 500, 1400, darr)
    def forward(self,x):
        x = torch.nn.functional.leaky_relu(self.layer1(x), 0.1)
        return torch.sigmoid(self.layer2(x))

torch.manual_seed(42)
m3 = Oracle3(256, DIGIT_ARRAY)
opt = optim.Adam(m3.parameters(), lr=0.02)
for ep in range(8000):
    opt.zero_grad(); l = nn.BCELoss()(m3(X3), y3); l.backward(); opt.step()
    if l.item() < 0.003: break
acc3 = ((m3(X3)>0.5).float()==y3).float().mean().item()*100
print(f"  Ramanujan (42 params) on 3-bit parity: acc = {acc3:.0f}% | loss = {l.item():.4f}")

torch.manual_seed(42)
mlp3 = nn.Sequential(nn.Linear(3,256), nn.ReLU(), nn.Linear(256,1), nn.Sigmoid())
opt = optim.Adam(mlp3.parameters(), lr=0.05)
for ep in range(8000):
    opt.zero_grad(); l3 = nn.BCELoss()(mlp3(X3), y3); l3.backward(); opt.step()
    if l3.item() < 0.003: break
acc3m = ((mlp3(X3)>0.5).float()==y3).float().mean().item()*100
print(f"  Standard MLP (1025 params) on 3-bit parity: acc = {acc3m:.0f}% | loss = {l3.item():.4f}")