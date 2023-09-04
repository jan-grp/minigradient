import math

class Mini:
    def __init__(self, value):
        self.value = value
        self.gradient = 1 
        self._parents = (None, None)
        # self.operation = None
        self._slope = 1 # the derivative it's child with respect to itself

    def __add__(self, other): # handles: self + other
        if isinstance(other, Mini):
            value = self.value + other.value
            child = Mini(value=value)

            self._slope = 1 if self.value > 0 else -1
            other._slope = 1 if other.value > 0 else -1
            child._parents = (self, other)
            return child
        else:
            # other is not an instance of Mini
            assert isinstance(other, (int, float)), "right operand must be of type Mini/int/float"
            other_instance = Mini(value=other)
            value = self.value + other_instance.value
            child = Mini(value=value)

            self._slope = 1 if self.value > 0 else -1
            other_instance._slope = 1 if other_instance.value > 0 else -1
            child._parents = (self, other_instance)
            return child

    def __radd__(self, other): # handles: other + self
        return self + other # runs __add__()
    
    def __mul__(self, other): # handles: self * other
        if isinstance(other, Mini):
            value = self.value * other.value
            child = Mini(value=value)

            self._slope = other.value
            other._slope = self.value
            child._parents = (self, other)
            return child
        else:
            # other is not an instance of Mini
            assert isinstance(other, (int, float)), "right operand must be of type Mini/int/float"
            other_instance = Mini(value=other)
            value = self.value * other_instance.value
            child = Mini(value=value)

            self._slope = other_instance.value
            other_instance._slope = self.value
            child._parents = (self, other_instance)
            return Mini(value=self.value * other)
    
    def __rmul__(self, other): # handles: other * self
        return self * other # runs __mul__()
    
    def __pow__(self, other): # handles: self ** other
        if isinstance(other, Mini):
            value = self.value ** other.value
            child = Mini(value=value)

            self._slope = other.value * self.value**(other.value - 1)
            other._slope = math.log(self.value) * self.value**other.value
            child._parents = (self, other)
            return child
        else:
            # other is not an instance of Mini
            assert isinstance(other, (int, float)), "right operand must be of type Mini/int/float"
            other_instance = Mini(value=other)
            value = self.value ** other_instance.value
            child = Mini(value=value)

            self._slope = other_instance.value * self.value**(other_instance.value - 1)
            other_instance._slope = math.log(self.value) * self.value**other_instance.value
            child._parents = (self, other_instance)
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
        self.value *= -1 # runs __mul__()
        return self

    def __sub__(self, other): # handles self - other
        assert isinstance(other, (Mini, int, float)), "subtrahend must be of type Mini/int/float"
        return self + (-other) # may run __neg__() first, then runs __add__()

    def __rsub__(self, other): # handles other - self
        assert isinstance(other, (Mini, int, float)), "subtrahend must be of type Mini/int/float"
        return other + (-self) # first runs __neg__(), then runs __add__()

    # backpropagation
    def backprop(self):
        assert all(isinstance(e, Mini) for e in self._parents), "object is not the result of calculations of the Mini class"
        for parent in self._parents:
            parent.gradient = self.gradient * parent._slope
            if all(isinstance(e, Mini) for e in parent._parents):
                parent.backprop()
