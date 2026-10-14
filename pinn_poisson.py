"""
PINN для уравнения Пуассона:
    u_xx + u_yy = -f(x,y),  (x,y) in (0,1)^2
    u = 0 на границе

Тестовый пример:
    f(x,y) = 2*pi^2 * sin(pi*x) * sin(pi*y)
    u_exact(x,y) = sin(pi*x) * sin(pi*y)

Параметры: 
    MLP-модель + tanh - функция-активатор
    Xavier-инициализация 
    Функция потерь: L = L_(PDE) + lamda * L_(BC), где
    L_(PDE) - невязка
    L_(BC) - граничные условия
    L = 1/M * sum(u_xx + u_yy + f)^2 + lamda * 1/K * sum(u^2) 
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device}")


class PINN(nn.Module):
    """MLP: (x, y) -> u(x, y)."""
    def __init__(self, hidden=4, neurons=64):
        super().__init__()
        layers = [nn.Linear(2, neurons), nn.Tanh()]
        for _ in range(hidden):
            layers += [nn.Linear(neurons, neurons), nn.Tanh()]
        layers.append(nn.Linear(neurons, 1))
        self.net = nn.Sequential(*layers)

        # Xavier инициализация
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


def laplacian(u, x, y):
    """u_xx + u_yy."""
    u_x = torch.autograd.grad(u, x, torch.ones_like(u),
                              create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u),
                              create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                               create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y),
                               create_graph=True)[0]
    return u_xx + u_yy


def source(x, y):
    return 2.0 * np.pi**2 * torch.sin(np.pi * x) * torch.sin(np.pi * y)

def exact(x, y):
    return torch.sin(np.pi * x) * torch.sin(np.pi * y)


def sample_interior(n):
    """n случайных точек внутри (0,1)^2."""
    x = torch.rand(n, 1, device=device, requires_grad=True)
    y = torch.rand(n, 1, device=device, requires_grad=True)
    return x, y


def sample_boundary(n_per_side):
    """Точки на 4 сторонах квадрата."""
    s = n_per_side
    x_b = torch.cat([
        torch.rand(s, 1, device=device),           # y=0
        torch.rand(s, 1, device=device),           # y=1
        torch.zeros(s, 1, device=device),          # x=0
        torch.ones(s, 1, device=device),           # x=1
    ])
    y_b = torch.cat([
        torch.zeros(s, 1, device=device),
        torch.ones(s, 1, device=device),
        torch.rand(s, 1, device=device),
        torch.rand(s, 1, device=device),
    ])
    return x_b, y_b


def loss_fn(model, x_int, y_int, x_b, y_b, lam=10.0):
    u = model(x_int, y_int)
    r = laplacian(u, x_int, y_int) + source(x_int, y_int)
    L_pde = torch.mean(r**2)

    u_b = model(x_b, y_b)
    L_bc = torch.mean(u_b**2)

    return L_pde + lam * L_bc, L_pde, L_bc

"""
Обучение:
Использован оптимизатор Adam - обновление весов, используя : 1) скользящее среднее градиентов
                                                             2) квадратов градиентов
Каждые 3000эпох learning rate падает /2.
Генерируем 2000 + 400 точек (100 x 4).
Перегенерируем каждые 200эпох.
"""
def train(model, n_epochs=8000, lr=1e-3,
          n_int=2000, n_bnd=400, resample_every=200):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3000, gamma=0.5)

    x_int, y_int = sample_interior(n_int)
    x_b, y_b = sample_boundary(n_bnd // 4)

    hist = {"loss": [], "pde": [], "bc": []}

    for ep in range(n_epochs):
        if ep % resample_every == 0 and ep > 0:
            x_int, y_int = sample_interior(n_int)
            x_b, y_b = sample_boundary(n_bnd // 4)

        opt.zero_grad()
        loss, L_pde, L_bc = loss_fn(model, x_int, y_int, x_b, y_b)
        loss.backward()
        opt.step()
        sched.step()

        hist["loss"].append(loss.item())
        hist["pde"].append(L_pde.item())
        hist["bc"].append(L_bc.item())

        if ep % 500 == 0:
            print(f"Epoch {ep:5d} | Loss={loss.item():.3e} "
                  f"| PDE={L_pde.item():.3e} | BC={L_bc.item():.3e}")

    return hist


def evaluate(model, N=100):
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y, indexing="ij")

    xt = torch.tensor(X.flatten(), dtype=torch.float32,
                      device=device).unsqueeze(1)
    yt = torch.tensor(Y.flatten(), dtype=torch.float32,
                      device=device).unsqueeze(1)

    with torch.no_grad():
        U = model(xt, yt).cpu().numpy().reshape(N, N)

    U_ex = np.sin(np.pi * X) * np.sin(np.pi * Y)

    L2 = np.sqrt(np.mean((U - U_ex) ** 2))
    Linf = np.max(np.abs(U - U_ex))
    print(f"\n--- Ошибка на сетке {N}x{N} ---")
    print(f"L2:  {L2:.4e}")
    print(f"L∞:  {Linf:.4e}")

    return X, Y, U, U_ex, L2, Linf


def plot_all(hist, X, Y, U, U_ex):
    fig = plt.figure(figsize=(16, 10))

    ax = fig.add_subplot(2, 3, 1)
    ax.semilogy(hist["loss"], label="Total")
    ax.semilogy(hist["pde"], label="PDE")
    ax.semilogy(hist["bc"], label="BC")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Обучение"); ax.legend(); ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 2, projection="3d")
    ax.plot_surface(X, Y, U_ex, cmap="viridis")
    ax.set_title("Точное решение")

    ax = fig.add_subplot(2, 3, 3, projection="3d")
    ax.plot_surface(X, Y, U, cmap="viridis")
    ax.set_title("PINN")

    err = np.abs(U - U_ex)
    ax = fig.add_subplot(2, 3, 4, projection="3d")
    ax.plot_surface(X, Y, err, cmap="hot")
    ax.set_title(f"|Ошибка| max={err.max():.2e}")

    plt.tight_layout()
    plt.savefig("pinn_poisson.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    model = PINN(hidden=4, neurons=64).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров: {n_params}")

    print("\n--- Обучение ---")
    hist = train(model, n_epochs=8000, lr=1e-3,
                 n_int=2000, n_bnd=400)

    print("\n--- Оценка ---")
    X, Y, U, U_ex, L2, Linf = evaluate(model, N=100)

    plot_all(hist, X, Y, U, U_ex)