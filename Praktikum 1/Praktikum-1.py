import numpy as np
import re
import matplotlib.pyplot as plt


def regula_falsi(f, a, b, tol=1e-6, max_iter=100):
    if f(a) * f(b) >= 0:
        return None, []

    tabel = []

    for i in range(1, max_iter + 1):
        c = b - (f(b) * (b - a)) / (f(b) - f(a))
        fc = f(c)
        tabel.append((i, a, b, c, fc))

        if abs(fc) < tol:
            return c, tabel
        elif f(a) * fc < 0:
            b = c
        else:
            a = c

    return c, tabel


def tampilkan_tabel(tabel):
    print("Iterasi |     a     |     b     |     c     |    f(c)   ")
    print("--------------------------------------------------------")
    for i, a, b, c, fc in tabel:
        print(f"{i:^7} | {a:8.5f} | {b:8.5f} | {c:8.5f} | {fc:9.6f}")
    print()


def plot_function(f, a, b, roots):
    x = np.linspace(a, b, 400)
    y = f(x)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label='f(x)')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.scatter(roots, [0] * len(roots), color='red', zorder=5, label='Akar')

    plt.title('Grafik Fungsi dan Akar (Metode Regula Falsi)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def preprocess_expression(expr):
    expr = expr.replace("^", "**").replace(" ", "")
    expr = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expr)
    expr = re.sub(r'([a-zA-Z)])(\d)', r'\1*\2', expr)
    return expr


def parse_function(expr):
    expr = preprocess_expression(expr)
    allowed_funcs = {
        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
        'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan,
        'log': np.log, 'sqrt': np.sqrt, 'exp': np.exp,
        'abs': np.abs, 'pi': np.pi, 'e': np.e
    }

    def func(x):
        try:
            return eval(expr, {"__builtins__": {}}, {**allowed_funcs, 'x': x})
        except Exception as e:
            raise ValueError(f"Kesalahan dalam evaluasi fungsi: {e}")

    return func


def cari_akar(f, start=-10, end=10, step=0.5, max_extend=100):
    estimated_roots = []
    root_tabels = {}  # key = root, value = tabel iterasi

    def cari_di_rentang(x_range):
        local_roots = []
        for i in range(len(x_range) - 1):
            a, b = x_range[i], x_range[i + 1]
            try:
                if f(a) * f(b) < 0:
                    root, tabel = regula_falsi(f, a, b)
                    if root is not None:
                        root = round(root, 6)
                        local_roots.append(root)
                        root_tabels[root] = tabel
            except:
                continue
        return local_roots

    range_limit = max(abs(start), abs(end))
    x_range = np.arange(start, end, step)
    estimated_roots += cari_di_rentang(x_range)

    for x in x_range:
        try:
            if abs(f(x)) < 1e-6:
                root = round(x, 6)
                estimated_roots.append(root)
                root_tabels[root] = [(0, x, x, x, f(x))]
        except:
            continue

    while not estimated_roots and range_limit <= max_extend:
        range_limit += 10
        x_range = np.arange(-range_limit, range_limit, step)
        estimated_roots += cari_di_rentang(x_range)
        for x in x_range:
            try:
                if abs(f(x)) < 1e-6:
                    root = round(x, 6)
                    estimated_roots.append(root)
                    root_tabels[root] = [(0, x, x, x, f(x))]
            except:
                continue

    return sorted(set(estimated_roots)), -range_limit, range_limit, root_tabels


if __name__ == "__main__":
    try:
        user_input = input("Masukkan fungsi (gunakan variabel x, contoh: sin(x)-x/2): ")
        f = parse_function(user_input)
        _ = f(0)
    except Exception as e:
        print(f"Fungsi tidak valid: {e}")
        exit()

    roots, plot_min, plot_max, tabels = cari_akar(f)

    if roots:
        print(f"\nAkar-akar yang ditemukan: {roots}\n")
        for r in roots:
            print(f"Tabel iterasi untuk akar mendekati x = {r}:")
            tampilkan_tabel(tabels[r])

        plot_min = min(plot_min, min(roots) - 1)
        plot_max = max(plot_max, max(roots) + 1)
        plot_function(f, plot_min, plot_max, roots)
    else:
        print("Tidak ditemukan akar dalam rentang yang diberikan maupun setelah eksplorasi.")
