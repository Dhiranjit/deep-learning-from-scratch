""" Scaler Autograd Engine"""
import math


class Value:
    def __init__(self, data, _prev=()):
        self.data = data
        self.grad = 0.0
        self._prev = set(_prev)
        self._backward = lambda: None
        
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        # closure  
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __neg__(self):
        out = Value(-self.data, (self,))

        def _backward():
            self.grad += -1 * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)
    
    def __pow__(self, other):
        assert isinstance (other, (int, float))
        out = Value((self.data ** other), (self,))

        def _backward():
            self.grad += out.grad * other * (self.data ** (other - 1))
        out._backward = _backward

        return out
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def exp(self):
        out = Value(math.exp(self.data), (self,))

        def _backward():
            self.grad += out.data * out.grad 
        out._backward = _backward
        return out 
    
    def log(self):
        out = Value(math.log(self.data), (self,))

        def _backward():
            self.grad += (1/self.data) * out.grad 
        out._backward = _backward
        return out
    
    def sigmoid(self):
        s = 1 / (1 + (math.exp(-self.data)))
        out = Value(s, (self,))

        def _backward():
            self.grad += out.grad * (s * (1 - s))
        out._backward = _backward
        return out

    
    def relu(self):
        out = Value(self.data if self.data > 0 else 0, (self,))

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out 

    def backward(self):
        topo = []
        visited = set()
        # Build a topological sort
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        # call _backward() in reversed order
        for node in reversed(topo):
            node._backward()
    
    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return (-self) + other

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return other * (self ** -1)
    

    def __repr__(self):
        return f"Value: {self.data}"