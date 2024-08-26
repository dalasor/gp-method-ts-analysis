import matplotlib.pyplot as plt
import numpy as np


# Аттраткор Лоренца
def lorenz(xyz, s=10, r=28, b=2.667):
    x, y, z = xyz
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array([x_dot, y_dot, z_dot])


dt = 0.01
num_steps = [100, 1000, 2000, 5000]
xs, ys, zs = [], [], []

for i in range(4):
    xyzs = np.empty((num_steps[i] + 1, 3))  # Пустые массивы временных рядов
    xyzs[0] = (0., 1., 1.05)  # Подобранны начальные условия

    # Метод Эйлера
    for j in range(num_steps[i]):
        xyzs[j + 1] = xyzs[j] + lorenz(xyzs[j]) * dt

    # Достаем временные ряды x(t), y(t), z(t)
    x_t, y_t, z_t = [xyzs[:, i] for i in range(3)]
    xs.append(x_t)
    ys.append(y_t)
    zs.append(z_t)

if __name__ == '__main__':
    for i in range(4):
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(xs[i], ys[i], zs[i], lw=0.5)
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title(f"N = {num_steps[i]}")
        plt.show()
