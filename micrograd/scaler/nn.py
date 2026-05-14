import random
from micrograd.scaler.engine import Value


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return []
    

class Linear(Module):
    def __init__(self, nin, nout):
        # Kaiming/He initialization 
        limit = (2 / nin) ** 0.5
        self.w = [
            [Value(random.uniform(-limit, limit)) for _ in range(nin)]
            for _ in range(nout)
        ]
        self.b = [Value(0) for _ in range(nout)]
    
    def __call__(self, x):
        out = []
        for w_row, b in zip(self.w, self.b):
            act = sum((wi * xi for wi, xi in zip(w_row, x)), b)
            out.append(act)
        if len(out) == 1:
            return out[0]
        return out
    
    def parameters(self):
        return [p for w_row in self.w for p in w_row] + self.b
    
    def __repr__(self):
        return f"Linear: ({len(self.w[0])}, {len(self.w)})"


class ReLU(Module):
    def __call__(self, x):
        if isinstance(x, list):
            return [xi.relu() for xi in x]
        return x.relu()
    
    def __repr__(self):
        return f"ReLU()"


class Sigmoid(Module):
    def __call__(self, x):
        if isinstance(x, list):
            return [xi.sigmoid() for xi in x]
        return x.sigmoid()
    
    def __repr__(self):
        return f"Sigmoid()"


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        layers = ",\n    ".join(str(l) for l in self.layers)
        n_params = len(self.parameters())

        return (
            f"Sequential(\n"
            f"    {layers}\n"
            f")\n\n"
            f"Parameters: {n_params}"
        )


# Loss Function 
def binary_cross_entropy(y_pred, y_true):
    eps = 1e-8
    l = -(y_true * (y_pred + eps).log() + (1 - y_true) * (1 - y_pred + eps).log())
    return l

def evaluate(model, X, y):
    losses = []
    correct = 0.0
    for x_i, y_i in zip(X, y):
        y_pred = model(x_i)
        loss = binary_cross_entropy(y_pred, y_i)
        losses.append(loss.data)
        pred = 1 if y_pred.data >= 0.5 else 0
        correct += (pred == y_i)
    avg_loss = sum(losses) / len(losses)
    acc = correct / len(y)

    return avg_loss, acc
