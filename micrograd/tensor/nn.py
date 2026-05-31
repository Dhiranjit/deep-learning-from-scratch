import numpy as np
from micrograd.tensor.engine import Tensor


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.grad)
        
    def parameters(self):
        return []


class Linear(Module):
    def __init__(self, nin: int, nout: int):
        self.nin = nin
        self.nout = nout
        self.W = Tensor.randn(nout, nin) * (2 / nin) ** 0.5 # He init
        self.b = Tensor.zeros(nout,)
    
    def __call__(self, X: Tensor):
        out = X @ self.W.T + self.b 
        return out
    
    def parameters(self):
        return [self.W, self.b]
    
    def __repr__(self):
        return f"Linear: ({self.nin}, {self.nout})"


class ReLU(Module):
    def __call__(self, X: Tensor):
        return X.relu()
    
    def __repr__(self):
        return f"ReLU()"


class Sigmoid(Module):
    def __call__(self, X: Tensor):
        return X.sigmoid()
    
    def __repr__(self):
        return f"Sigmoid()"
    

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers
    
    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        layers = ",\n    ".join(str(l) for l in self.layers)

        return (
            f"Sequential(\n"
            f"    {layers}\n"
            f")\n"
        )

class CrossEntropyLossWithLogits:
    def __call__(self, logits: Tensor, targets: np.ndarray) -> Tensor:
        """
        logits: Tensor of shape (N, C)
        targets: np.ndarray of shape (N,) with integer class indices
        return: scaler Tensor (mean loss over batch)
        """
        x = logits.data
        N, C = x.shape

        # log-sum-exp trick for stability
        m = x.max(axis=1, keepdims=True) # (N, 1)
        x_shifted = x - m
        log_sum_exp = m.squeeze(1) + np.log(np.exp(x_shifted).sum(axis=1)) # (N,)
        nll = -x[np.arange(N), targets] + log_sum_exp
        loss = nll.mean()
        out = Tensor(loss, (logits,))
        
        def _backward():
            probs = np.exp(x_shifted) / np.exp(x_shifted).sum(axis=1, keepdims=True) # (N, C)
            probs[np.arange(N), targets] -= 1.0
            probs /= N
            logits.grad += probs * out.grad
        out._backward = _backward

        return out 

