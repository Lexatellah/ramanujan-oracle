import torch
import torch.nn as nn
import ast
from math import factorial
from decimal import Decimal, getcontext

getcontext().prec = 7000
def ramanujan_digits(num_terms, num_digits):
    total = Decimal(0)
    for k in range(num_terms):
        num = Decimal(factorial(4*k)) * Decimal(1103 + 26390*k)
        den = (Decimal(factorial(k))**4) * (Decimal(396)**(4*k))
        total += num / den
    val = (Decimal(2).sqrt() * 2 / Decimal(9801)) * total
    return ''.join(c for c in format(val,'f') if c.isdigit())[1:num_digits+1]

DIGIT_ARRAY = torch.tensor([int(c) for c in ramanujan_digits(700, 7000)], dtype=torch.long)

class RamanujanLinear(nn.Module):
    def __init__(self, in_f, out_f, seed_offset, bias_offset, digit_array):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.num_weights = in_f * out_f
        self.digit_embed = nn.Parameter(torch.zeros(10))
        self.bias_embed  = nn.Parameter(torch.zeros(10))
        self.scale       = nn.Parameter(torch.tensor(1.0))
        w = torch.arange(seed_offset, seed_offset + self.num_weights) % len(digit_array)
        self.register_buffer('weight_digit_idx', digit_array[w])
        b = torch.arange(bias_offset, bias_offset + out_f) % len(digit_array)
        self.register_buffer('bias_digit_idx', digit_array[b])
    def forward(self,x):
        W = self.scale * self.digit_embed[self.weight_digit_idx].view(self.out_f, self.in_f)
        b = self.bias_embed[self.bias_digit_idx]
        return torch.matmul(x, W.t()) + b

class RamanujanOracle(nn.Module):
    def __init__(self, hidden, digit_array):
        super().__init__()
        self.hidden = hidden
        self.layer1 = RamanujanLinear(2, hidden, 0, 1000, digit_array)
        self.layer2 = RamanujanLinear(hidden, 1, 500, 1400, digit_array)
    def forward(self,x):
        x = torch.nn.functional.leaky_relu(self.layer1(x), 0.1)
        return torch.sigmoid(self.layer2(x))

def load_model(path):
    hidden, params = None, {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') and 'hidden=' in line:
                hidden = int(line.split('hidden=')[1].split(',')[0])
            elif '=' in line and not line.startswith('#'):
                name, val = line.split('=',1)
                params[name.strip()] = ast.literal_eval(val.strip())
    m = RamanujanOracle(hidden, DIGIT_ARRAY)
    for name, vals in params.items():
        parts = name.split('.')
        obj = m
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], nn.Parameter(torch.tensor(vals, dtype=torch.float32)))
    return m

if __name__ == "__main__":
    m = load_model("ramanujan_oracle.txt")
    X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    y = torch.tensor([[0.],[1.],[1.],[0.]])
    with torch.no_grad():
        out = m(X)
        acc = ((out>0.5).float()==y).float().mean().item()*100
    print("Loaded model from ramanujan_oracle.txt")
    print(f"  hidden           : {m.hidden}")
    print(f"  learnable params : {sum(p.numel() for p in m.parameters())}")
    print(f"  probs            : {[round(v,4) for v in out.flatten().tolist()]}")
    print(f"  truth            : {y.flatten().tolist()}")
    print(f"  accuracy         : {acc:.0f}%")