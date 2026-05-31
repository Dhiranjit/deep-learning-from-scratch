import numpy as np


class Tensor():
    """
    A minimal autograd tensor engine built on numpy.

    Wraps an n-dimensional array with Tensor class to create a Directed Acyclic Graph
    which enables us to _backward recursively to calculate the gradients.

    Accepts any array-like (Python scalars, lists, numpy arrays); data is
    stored as float32.
    """

    def __init__(self, data, _prev=()):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self._prev = set(_prev)
        self._backward = lambda: None

    @classmethod
    def randn(cls, *shape):
        data = np.random.randn(*shape).astype(np.float32)
        return cls(data)
    
    @classmethod
    def zeros(cls, *shape):
        return  cls(np.zeros(shape).astype(np.float32))

    @staticmethod
    def unbroadcast(grad, target_shape):
        ndims_added = grad.ndim - len(target_shape)

        for _ in range(ndims_added):
            grad = grad.sum(axis=0)

        for i, dim in enumerate(target_shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def T(self):
        out = Tensor(self.data.T, (self,))

        def _backward():
            self.grad += out.grad.T
        out._backward = _backward
        return out


    # ---------- Binary ops ----------

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            self.grad += Tensor.unbroadcast(out.grad, self.shape)
            other.grad += Tensor.unbroadcast(out.grad, other.shape)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other))

        def _backward():
            self.grad += Tensor.unbroadcast(out.grad * other.data, self.shape)
            other.grad += Tensor.unbroadcast(out.grad * self.data, other.shape)
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return (other ** -1) * self

    def __matmul__(self, other):
        assert isinstance(other, Tensor)
        out = Tensor(self.data @ other.data, (self, other))

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    # ---------- Unary ops ----------

    def __neg__(self):
        out = Tensor(-self.data, (self,))

        def _backward():
            self.grad += -out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (float, int))
        out = Tensor(self.data ** other, (self,))

        def _backward():
            self.grad += out.grad * other * self.data ** (other - 1)
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,))

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    # ---------- Activations ----------

    def sigmoid(self):
        s = 1 / (1 + np.exp(-self.data))
        out = Tensor(s, (self,))

        def _backward():
            self.grad += out.grad * (s * (1 - s))
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0), (self,))

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    # ---------- Backward ----------

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()


    def __repr__(self):
        arr = np.array2string(self.data)
        indent = " " * 7
        arr = arr.replace("\n", "\n" + indent)
        return f"tensor({arr})"
