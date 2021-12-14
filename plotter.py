import neuro_math
import numpy as np


def degree_filling(seq, n):
    for i in range(5, n + 5):
        seq.append(2 ** i)


def mod_filling(seq, n):
    for i in range(5, n + 5):
        seq.append(i % 10)


def arithmetic_filling(seq, n):
    for i in range(5, n + 5):
        seq.append(i)


def fibonacci_filling(seq, n):
    for i in range(n):
        seq.append(fib_recursion(i))


def fib_recursion(n):
    if n <= 1:
        return n
    else:
        return fib_recursion(n - 1) + fib_recursion(n - 2)


window_size = 3

# create original dataset for training
original_data = []
arithmetic_filling(original_data, 30)

# create dataset for recognition
test_data = []
arithmetic_filling(test_data, window_size)
model = neuro_math.NeuroMath(original_data, window_size=window_size)
model.training()
result = model.guessing(np.array(test_data))
original = (window_size + 5)
print('input sequence:')
print(test_data)
print('result: ' + str(result[0][0]))
print('original: ' + str(original))
error = abs((result[0][0] - original) / original)
print('error = ' + str(error))
