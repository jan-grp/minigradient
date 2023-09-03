import math

class Mini:
    def __init__(self, value):
        self.value = value
        self.gradient = 1
        self._parents = (None, None)
        # self.operation = None
        self._slope = 1

    # addition
    def __add__(self, other):
        if isinstance(other, Mini):
            # both are Mini
            value = self.value + other.value
            child = Mini(value=value)
            child._parents = (self, other)
            self._slope = 1
            other._slope = 1
            return child
        else:
            # only left is Mini
            assert isinstance(other, (int, float)), "right operand must be of type Mini/int/float"
            return Mini(value=self.value + other)
    
    def __radd__(self, other): # gets called when right operand is instance of class
        if isinstance(other, Mini):
            # both are Mini
            return Mini(value=self.value + other.value)
        else:
            # left is Mini
            return Mini(value=self.value + other)
    
    # multiplication
    def __mul__(self, other):
        if isinstance(other, Mini):
            # both are Mini
            value = self.value * other.value
            child = Mini(value=value)

            self._slope = other.value
            other._slope = self.value
            child._parents = (self, other)
            return child
        else:
            # only left is Mini
            assert isinstance(other, (int, float)), "right operand must be of type Mini/int/float"

            return Mini(value=self.value * other)
    
    def __rmul__(self, other):
        if isinstance(other, Mini):
            # both are Mini
            value = self.value * other.value
            child = Mini(value=value)

            self._slope = other.value
            other._slope = self.value
            child._parents = (self, other)
            return child
        else:
            # left is Mini
            return Mini(value=self.value * other)
    
    # exponentiation
    def __pow__(self, other):
        if isinstance(other, Mini):
            # both are Mini
            value = self.value ** other.value
            child = Mini(value=value)

            self._slope = other.value * self.value**(other.value - 1)
            other._slope = math.log(self.value) * self.value**(other.value)
            child._parents = (self, other)
            return child
        else:
            # only left is Mini
            assert isinstance(other, (int, float)), "right operand must be of type Mini/int/float"
            return Mini(value=self.value ** other)
        
    def __rpow__(self, other):
        return Mini(other**self.value)
    
    #backpropagation
    def backprop(self):
        assert all(isinstance(e, Mini) for e in self._parents), "object is not the result of calculations of the Mini class"
        for parent in self._parents:
            parent.gradient = self.gradient * parent._slope
            if all(isinstance(e, Mini) for e in parent._parents):
                parent.backprop()
        

w1 = Mini(2)
x1 = Mini(3)
w2 = Mini(4)
x2 = Mini(5)
b = Mini(6)

a1 = Mini(4)

y = w1*x1**a1 + w2*x2 + b

y.backprop()
print(a1.gradient)
print(f'expected: {w1.value*math.log(x1.value)*x1.value**a1.value}')