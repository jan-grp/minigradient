import math

class Mini:
    def __init__(self, value):
        self.value = value
        self.gradient = 1 # dy/dslef, where y is the final result where .backprop() is called on
        self._parents = (None, None)
        # self.operation = _operation
        self._slope = 1 # df/dself, where f is the immediate operation self is used for 

    def __add__(self, other): # handles: self + other
        assert isinstance(other, (Mini, int, float)), "operand must be of type Mini/int/float"
        if not isinstance(other, Mini):
            other = Mini(other)

        child = Mini(self.value + other.value)
        child._parents = (self, other)
        return child

    def __radd__(self, other): # handles: other + self
        return self + other # runs __add__()
    
    def __mul__(self, other): # handles: self * other
        assert isinstance(other, (Mini, int, float)), "operand must be of type Mini/int/float"
        if not isinstance(other, Mini):
            other = Mini(other)
        
        child = Mini(self.value * other.value)
        self._slope = other.value 
        other._slope = self.value 
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
        other._slope = math.log(self.value) * self.value**other.value if self.value > 0 else 1
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

    def __sub__(self, other): # handles self - other
        assert isinstance(other, (Mini, int, float)), "subtrahend must be of type Mini/int/float"
        return self + (-other) # may run __neg__() first, then runs __add__()

    def __rsub__(self, other): # handles other - self
        assert isinstance(other, (Mini, int, float)), "subtrahend must be of type Mini/int/float"
        return other + (-self) # first runs __neg__(), then runs __add__()
    
    def __neg__(self): # handles -obj
        child = Mini(-self.value)
        child._slope = -self._slope
        child.gradient = -self.gradient
        child._parents = (self, )
        return child

    def exp(self):
        self._slope = math.exp(self.value)
        child = Mini(math.exp(self.value))
        child._parents = (self, )
        return child

    def log(self):
        child = Mini(math.log(self.value))
        self._slope = 1 / self.value
        child._parents = (self, )
        return child

    def sigmoid(self):
        sigmoid_value = 1 / (1 + math.exp(-self.value))
        child = Mini(sigmoid_value)
        self._slope = sigmoid_value * (1 - sigmoid_value)
        child._parents = (self, )
        return child

    def backprop(self):
        if not all(isinstance(e, Mini) for e in self._parents):
            return self.gradient
        for parent in self._parents:
            parent.gradient = self.gradient * parent._slope
            if all(isinstance(e, Mini) for e in parent._parents):
                parent.backprop()
