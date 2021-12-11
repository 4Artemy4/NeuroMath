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


j = 15
model = main.NeuroMath(window_size=j, sequence='arithmetic')
model.training()
test_data = []
arithmetic_filling(test_data, j)
result = model.guessing(np.array(test_data))
original = (j + 5)
print('input sequence:')
print(test_data)
print('result:')
print(result)
print('original:')
print(original)
error = abs((result - original) / original)
print('error = ')
print(error)
