import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

def preprocess_expression(expr):
    expr = expr.replace("^", "**").replace(" ", "")
    expr = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expr)  # 4x -> 4*x, 2sin(x) -> 2*sin(x)
    expr = re.sub(r'([a-zA-Z)])(\d)', r'\1*\2', expr)  # x2 -> x*2
    return expr

def parse_function(expr):
    expr = preprocess_expression(expr)
    allowed_funcs = {
        "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log,
        "sqrt": np.sqrt, "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
        "pi": np.pi, "e": np.e
    }
    def f(x):
        return eval(expr, {"x": x, **allowed_funcs})
    return f

def trapezoidal(f, a, b, n):
    h = (b - a) / n
    s = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        s += f(a + i * h)
    return h * s

def romberg_table(f, a, b, max_order):
    R = np.zeros((max_order, max_order))
    for k in range(max_order):
        n = 2 ** k
        R[k, 0] = trapezoidal(f, a, b, n)
        for j in range(1, k + 1):
            R[k, j] = (4**j * R[k, j - 1] - R[k - 1, j - 1]) / (4**j - 1)
    return R

def plot_trapezoidal(f, a, b, n):
    x = np.linspace(a, b, 1000)
    plt.figure("Trapezoidal")
    plt.plot(x, f(x), label='f(x)')
    h = (b - a) / n
    for i in range(n):
        x0 = a + i * h
        x1 = a + (i + 1) * h
        plt.fill([x0, x0, x1, x1], [0, f(x0), f(x1), 0], 'lightblue', edgecolor='black', alpha=0.5)
    plt.title("Trapezoidal Area Approximation")
    plt.grid(True)
    plt.legend()

def plot_romberg(f, a, b, n):
    x = np.linspace(a, b, 1000)
    plt.figure("Romberg")
    plt.plot(x, f(x), label='f(x)')
    h = (b - a) / n
    for i in range(n):
        x0 = a + i * h
        x1 = a + (i + 1) * h
        plt.fill([x0, x0, x1, x1], [0, f(x0), f(x1), 0], 'lightgreen', edgecolor='black', alpha=0.5)
    plt.title("Romberg Area Approximation")
    plt.grid(True)
    plt.legend()

if __name__ == "__main__":
    try:
        user_input = input("Masukkan fungsi (gunakan variabel x, contoh: 4sin(x)-3x^2+2x): ")
        f = parse_function(user_input)
        a = float(input("Batas bawah (a): "))
        b = float(input("Batas atas (b): "))
        n = int(input("Jumlah subinterval (harus pangkat 2, contoh: 2, 4, 8, 16): "))
        max_order = int(np.log2(n)) + 1

        trap_data = [(2**k, trapezoidal(f, a, b, 2**k)) for k in range(max_order)]
        trap_df = pd.DataFrame(trap_data, columns=["n (subinterval)", "Trapezoidal Value"])

        romb_data = romberg_table(f, a, b, max_order)
        romb_df = pd.DataFrame(romb_data, columns=[f"R(k,{j})" for j in range(max_order)])
        romb_df.insert(0, "2^k intervals", [2**k for k in range(max_order)])

        print("\nTABEL METODE TRAPEZOIDAL:")
        print(trap_df.to_string(index=False))

        print("\nTABEL METODE ROMBERG:")
        print(romb_df.round(6).to_string(index=False))

        plot_trapezoidal(f, a, b, n)
        plot_romberg(f, a, b, n)
        plt.show()

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
