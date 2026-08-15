import math

class AddBackward:
    def __repr__(self):
        return "<AddBackward>"

class MulBackward:
    def __repr__(self):
        return "<MulBackward>"

class PowBackward:
    def __repr__(self):
        return "<PowBackward>"
    
class NegBackward:
    def __repr__(self):
        return "<NegBackward>"
    
class SubBackward:
    def __repr__(self):
        return "<SubBackward>"
    
class ReLUBackward:
    def __repr__(self):
        return "<ReLUBackward>"

class TanhBackward:
    def __repr__(self):
        return "<TanhBackward>"


class Tensor:
    def __init__(self, data, _children=(), _op="", label="", requires_grad=False):
        self.data = data
        self._prev = set(_children)
        self._op = _op              # the op that produced this node (for debugging/graphviz)
        self.label = label
        self.grad = None            # None until first backward pass — distinguishes
                                    
        self._backward = lambda: None  # closure that knows how to propagate grad
                                        # to this tensor's children; set by each op
        self.requires_grad = requires_grad
        self.grad_fn = None         # None for leaves; set by each op for non-leaves

    @property
    def is_leaf(self):
        # A leaf is a tensor with no children — i.e. created directly by the user,
        # not produce
        return len(self._prev) == 0

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad

        out = Tensor(self.data + other.data, _children=(self, other), _op="+", requires_grad=requires_grad)
        out.grad_fn = AddBackward()

        def _backward():
            if self.requires_grad:
                self.grad = out.grad if self.grad is None else self.grad + out.grad
            if other.requires_grad:
                other.grad = out.grad if other.grad is None else other.grad + out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad

        out = Tensor(self.data * other.data, _children=(self, other), _op="*", requires_grad=requires_grad)
        out.grad_fn = MulBackward()

        def _backward():
            if self.requires_grad:
                self.grad = other.data * out.grad if self.grad is None else self.grad + other.data * out.grad
            if other.requires_grad:
                other.grad = self.data * out.grad if other.grad is None else other.grad + self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError(f"Exponent must be int or float, got {type(other).__name__}")

        out = Tensor(self.data ** other, _children=(self,), _op="**", requires_grad=self.requires_grad)
        out.grad_fn = PowBackward()

        def _backward():
            if self.requires_grad:
                local_grad = out.grad * other * self.data ** (other - 1)
                self.grad = local_grad if self.grad is None else self.grad + local_grad
        out._backward = _backward
        return out
    
    def __neg__(self):
        out = Tensor(-1 * self.data, _children=(self,), _op="-", requires_grad=self.requires_grad)
        out.grad_fn = NegBackward()

        def _backward():
            if self.requires_grad:
                self.grad = out.grad * -1 if self.grad is None else self.grad + out.grad * -1
        out._backward = _backward
        return out
    
    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(self.data - other.data, _children=(self, other), _op="-", requires_grad=requires_grad)
        out.grad_fn = SubBackward()

        def _backward():
            if self.requires_grad:
                self.grad = out.grad if self.grad is None else self.grad + out.grad
            if other.requires_grad:
                other.grad = out.grad * -1 if other.grad is None else other.grad + (out.grad * -1)
        out._backward = _backward
        return out  
    
    def __truediv__(self, other):
        pass
        
    def __rtruediv__(self, other):
        pass

    # Activation functions
    def tanh(self):
        x = self.data 
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Tensor(data=t, _children=(self, ), _op="tanh", requires_grad=self.requires_grad)
        out.grad_fn = TanhBackward()

        def _backward():
            if self.requires_grad:
                self.grad = out.grad * (1 - t**2) if self.grad is None else self.grad + out.grad * (1 - t**2)
        out._backward = _backward
        return out
    
    def relu(self):
        out = Tensor(0 if self.data < 0 else self.data, (self, ), "Relu", requires_grad=self.requires_grad)
        out.grad_fn = ReLUBackward()

        def _backward():
            if self.requires_grad:
                local_grad = out.grad if self.data > 0 else 0
                self.grad = local_grad if self.grad is None else self.grad + local_grad
        out._backward = _backward
        return out
    
    def leaky_relu(self):
        pass

    def sigmoid(self):
        pass 
            
    def exp(self):
        pass

    def log(self):
        ...

    def gelu(self):
        ...

    def softmax(self):
        ...

    def __rsub__(self, other):
        return self.__sub__(other)

    def __repr__(self):
        return (f"Tensor(data={self.data}, grad={self.grad}, "
                f"requires_grad={self.requires_grad}, is_leaf={self.is_leaf}, "
                f"grad_fn={self.grad_fn})")