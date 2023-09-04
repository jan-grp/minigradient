import unittest
import math
from main import Mini

class TestMini(unittest.TestCase):
    def test_addition(self):
        a = Mini(10)
        b = Mini(4)
        result = a + b
        self.assertEqual(result.value, 14)
        result.backprop()
        self.assertEqual(a.gradient, 1)
    
    def test_multiplication(self):
        a = Mini(5)
        b = Mini(3)
        result = a * b
        self.assertEqual(result.value, 15)
        result.backprop()
        self.assertEqual(a.gradient, 3)

    def test_power(self):
        a = Mini(2)
        b = Mini(3)
        result = a ** b
        self.assertEqual(result.value, 8)
        result.backprop()
        result.backprop()
        self.assertEqual(a.gradient, 12)  # The gradient for a ** b with respect to a = b * a**(b-1)

    def test_division(self):
        a = Mini(10)
        b = Mini(2)
        result = a / b
        self.assertEqual(result.value, 5)
        result.backprop()
        result.backprop()
        self.assertEqual(a.gradient, 0.5)  # The gradient for a / b with respect to a = 1 / b

    def test_negation(self):
        a = Mini(7)
        neg_a = -a
        self.assertEqual(neg_a.value, -7)
        neg_a.backprop()
        self.assertEqual(neg_a.gradient, -1)

    def test_subtraction(self):
        a = Mini(10)
        b = Mini(4)
        result = a - b
        self.assertEqual(result.value, 6)
        result.backprop()
        self.assertEqual(a.gradient, 1)

    def test_radd(self):
        a = Mini(10)
        result = 5 + a
        self.assertEqual(result.value, 15)
        result.backprop()
        self.assertEqual(result.gradient, 1)

    def test_rmul(self):
        a = Mini(5)
        result = 3 * a
        self.assertEqual(result.value, 15)
        result.backprop()
        self.assertEqual(a.gradient, 3)

    def test_rpow(self):
        a = Mini(2)
        result = 3 ** a
        self.assertEqual(result.value, 9)
        result.backprop()
        self.assertEqual(a.gradient, 9 * math.log(3))  # The gradient for b ** a with respect to a = log(b) * b**a
    
    def test_rtruediv(self):
        a = Mini(4)
        result = 12 / a
        self.assertEqual(result.value, 3)
        result.backprop()
        self.assertEqual(a.gradient, -3/4)  # The gradient for a / b with respect to b = -a / b**2

    def test_backpropagation(self):
        a = Mini(3)
        b = Mini(2)
        c = a * b
        d = c + 1
        e = d ** 2
        e.backprop()
        self.assertEqual(a.gradient, 28)
        self.assertEqual(b.gradient, 42)
        self.assertEqual(c.gradient, 14)
        self.assertEqual(d.gradient, 14)

    def test_backpropagation_numerically(self):
        # numerical tests to calm my insecurities
        w1 = Mini(3)
        x1 = Mini(2)
        b = Mini(1)
        a = Mini(2)
        y = (w1*x1 + b)**a
        
        h = 0.00001
        d_w1 = (((w1.value + h)*x1.value + b.value)**a.value - y.value) / h
        d_x1 = ((w1.value*(x1.value + h) + b.value)**a.value - y.value) / h
        d_b = ((w1.value*x1.value + (b.value + h))**a.value - y.value) / h
        d_a = ((w1.value*x1.value + b.value)**(a.value + h) - y.value) / h

        y.backprop()
        self.assertAlmostEqual(w1.gradient, d_w1, 2)
        self.assertAlmostEqual(x1.gradient, d_x1, 2)
        self.assertAlmostEqual(b.gradient, d_b, 2)
        self.assertAlmostEqual(a.gradient, d_a, 2)
        


unittest.main()
