"""
Решение уравнения Пуассона методом конечных разностей (5-точечный шаблон).

    u_xx + u_yy = -f(x,y),  (x,y) in (0,1)^2
    u = 0 на границе

Тестовый пример:
    f(x,y) = 2*pi^2 * sin(pi*x) * sin(pi*y)
    u_exact(x,y) = sin(pi*x) * sin(pi*y)

"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

RESULTS_DIR = os.path.join("results", "fd")
os.makedirs(RESULTS_DIR, exist_ok=True)

def source_term(X, Y):
    """f(x, y) = 2*pi^2 * sin(pi*x) * sin(pi*y)."""
    return 2.0 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)


def exact_solution(X, Y):
    """Точное решение u(x, y) = sin(pi*x) * sin(pi*y)."""
    return np.sin(np.pi * X) * np.sin(np.pi * Y)


# 2. 5-точечный шаблон
def build_system(N):
    """
    Строит разреженную матрицу A и вектор правой части b
    для внутренних узлов сетки N x N.

    Неизвестные: u[i, j] для i, j = 1, ..., N-1
    Индексация вектора: k = (i-1)*(N-1) + (j-1)
    Размер системы: (N-1)^2 x (N-1)^2

    Уравнение для узла (i, j):
        u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j] = -h^2 * f[i,j]
    """
    h = 1.0 / N
    M = N - 1 
    size = M * M 

    rows, cols, vals = [], [], []

    def idx(i, j):
        return (i - 1) * M + (j - 1)

    for i in range(1, N):
        for j in range(1, N):
            k = idx(i, j)
            # Диагональ: -4
            rows.append(k); cols.append(k);    vals.append(-4.0)

            # Сосед слева: u[i-1, j]
            if i - 1 >= 1:
                rows.append(k); cols.append(idx(i - 1, j)); vals.append(1.0)

            # Сосед справа: u[i+1, j]
            if i + 1 <= N - 1:
                rows.append(k); cols.append(idx(i + 1, j)); vals.append(1.0)

            # Сосед снизу: u[i, j-1]
            if j - 1 >= 1:
                rows.append(k); cols.append(idx(i, j - 1)); vals.append(1.0)

            # Сосед сверху: u[i, j+1]
            if j + 1 <= N - 1:
                rows.append(k); cols.append(idx(i, j + 1)); vals.append(1.0)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(size, size))

    # Правая часть: -h^2 * f[i,j]
    x = np.linspace(0, 1, N + 1)
    y = np.linspace(0, 1, N + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")
    F = source_term(X, Y)

    b = np.zeros(size)
    for i in range(1, N):
        for j in range(1, N):
            k = idx(i, j)
            b[k] = -h**2 * F[i, j]

    return A, b, h


def solve_fd(N):
   
    A, b, h = build_system(N)

    u_inner = spla.spsolve(A, b)

    U = np.zeros((N + 1, N + 1))
    M = N - 1
    for i in range(1, N):
        for j in range(1, N):
            U[i, j] = u_inner[(i - 1) * M + (j - 1)]

    return U, h


def plot_solution(U, h, N):
    x = np.linspace(0, 1, N + 1)
    y = np.linspace(0, 1, N + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")

    U_exact = exact_solution(X, Y)
    err = np.abs(U - U_exact)

    # Метрики
    L2 = np.sqrt(np.mean((U - U_exact) ** 2))
    Linf = np.max(err)
    print(f"N = {N}, h = {h:.4e}")
    print(f"  L2-ошибка:  {L2:.4e}")
    print(f"  L∞-ошибка:  {Linf:.4e}")

    fig = plt.figure(figsize=(16, 5))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.plot_surface(X, Y, U_exact, cmap="viridis")
    ax1.set_title("Точное решение")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.plot_surface(X, Y, U, cmap="viridis")
    ax2.set_title(f"МКР (N={N})")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.plot_surface(X, Y, err, cmap="hot")
    ax3.set_title(f"Ошибка (max = {Linf:.2e})")
    ax3.set_xlabel("x"); ax3.set_ylabel("y")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"fd_poisson_N{N}.png"), dpi=150, bbox_inches="tight")
    plt.show()

    return L2, Linf


def convergence_study(Ns=(10, 20, 40, 80, 160)):
    L2_list, Linf_list, h_list = [], [], []

    for N in Ns:
        U, h = solve_fd(N)
        x = np.linspace(0, 1, N + 1)
        y = np.linspace(0, 1, N + 1)
        X, Y = np.meshgrid(x, y, indexing="ij")
        U_exact = exact_solution(X, Y)

        L2 = np.sqrt(np.mean((U - U_exact) ** 2))
        Linf = np.max(np.abs(U - U_exact))

        L2_list.append(L2)
        Linf_list.append(Linf)
        h_list.append(h)

        print(f"N = {N:4d}, h = {h:.4e}, L2 = {L2:.4e}, L∞ = {Linf:.4e}")

    h_arr = np.array(h_list)
    L2_arr = np.array(L2_list)
    Linf_arr = np.array(Linf_list)

    p_L2 = np.polyfit(np.log(h_arr), np.log(L2_arr), 1)[0]
    p_Linf = np.polyfit(np.log(h_arr), np.log(Linf_arr), 1)[0]

    print(f"\nПорядок сходимости по L2:  {p_L2:.3f}")
    print(f"Порядок сходимости по L∞:  {p_Linf:.3f}")

    plt.figure(figsize=(8, 6))
    plt.loglog(h_arr, L2_arr, "o-", label=f"L2 (порядок ≈ {p_L2:.2f})")
    plt.loglog(h_arr, Linf_arr, "s-", label=f"L∞ (порядок ≈ {p_Linf:.2f})")

    plt.loglog(h_arr, L2_arr[0] * (h_arr / h_arr[0]) ** 2,
               "k--", alpha=0.5, label="O(h²)")
    plt.xlabel("h"); plt.ylabel("Ошибка")
    plt.title("Сходимость МКР для уравнения Пуассона")
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "fd_convergence.png"), dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    N = 40
    U, h = solve_fd(N)
    plot_solution(U, h, N)

    print("\n--- Исследование сходимости ---")
    convergence_study(Ns=(10, 20, 40, 80, 160))