import math

class Mini:
    def __init__(self, value, gradient=1, _parents=(None, None), _slope=1):
        self.value = value
        self.gradient = gradient 
        self._parents = _parents
        # self.operation = None
        self._slope = _slope # the derivative of it's child with respect to itself

    def __add__(self, other): # handles: self + other
        assert isinstance(other, (Mini, int, float)), "operand must be of type Mini/int/float"
        if not isinstance(other, Mini):
            other = Mini(other)

        child = Mini(self.value + other.value)
        self._slope = 1 if self.value > 0 else -1
        other._slope = 1 if other.value > 0 else -1
        child._parents = (self, other)
        return child

    def __radd__(self, other): # handles: other + self
        return self + other # runs __add__()
    
    def __mul__(self, other): # handles: self * other
        assert isinstance(other, (Mini, int, float)), "operand must be of type Mini/int/float"
        if not isinstance(other, Mini):
            other = Mini(other)
        
        child = Mini(self.value * other.value)
        self._slope = other.value if self.value > 0 else -other.value
        other._slope = self.value if other.value > 0 else -self.value
        child._parents = (self, other)
        return child
    
    def __rmul__(self, other): # handles: other * self
        return self * other # runs __mul__()
    
    def __pow__(self, other): # handles: self ** other
        assert isinstance(other, (Mini, int, float)), "operand must be of type Mini/int/float"
        if not isinstance(other, Mini):
            other = Mini(other)
        
        child = Mini(self.value ** other.value)
        self._slope = other.value * self.value**(other.value - 1)
        other._slope = math.log(self.value) * self.value**other.value
        child._parents = (self, other)
        return child
 
    def __rpow__(self, other): # handles: other ** self
        assert isinstance(other, (int, float)), "base must be of type Mini/int/float"
        return Mini(value=other) ** self # runs __pow__()

    def __truediv__(self, other): # hanldes self / other
        assert isinstance(other, (Mini, int, float)), "divisor must be of type Mini/int/float"
        return self * other**-1 # first runs __pow__(), then runs __mul__()
        
    def __rtruediv__(self, other): # handles other / self
        assert isinstance(other, (Mini, int, float)), "dividend must be of type Mini/int/float"
        return other * self**-1 # first runs __pow__(), then runs __mul__()

    def __neg__(self): # handles -obj
        self.value *= -1 
        self._slope *= -1
        self.gradient *= -1
        return self

    def __sub__(self, other): # handles self - other
        assert isinstance(other, (Mini, int, float)), "subtrahend must be of type Mini/int/float"
        return self + (-other) # may run __neg__() first, then runs __add__()

    def __rsub__(self, other): # handles other - self
        assert isinstance(other, (Mini, int, float)), "subtrahend must be of type Mini/int/float"
        return other + (-self) # first runs __neg__(), then runs __add__()

    def exp(self):
        self._slope = math.exp(self.value)
        child = Mini(math.exp(self.value))
        child._parents = (self, )
        return child

    def sigmoid(self):
        def calculate(x):
            return 1 / (1 + math.exp(-x))
        
        self._slope = calculate(self.value)*(1-calculate(self.value))
        child = Mini(calculate(self.value))
        child._parents = (self, )
        return child 

    def backprop(self):
        if not all(isinstance(e, Mini) for e in self._parents):
            return self.gradient
        for parent in self._parents:
            parent.gradient = self.gradient * parent._slope
            if all(isinstance(e, Mini) for e in parent._parents):
                parent.backprop()
