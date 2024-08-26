# import lorenz
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import time


def phase_space_reconstruction(ts, t_delay, emb_dim):
    n = len(ts)
    return np.array([ts[i:i + emb_dim * t_delay:t_delay]
                     for i in range(n - emb_dim * t_delay + 1)])


def correlation_sum(phase_space, r):
    n = len(phase_space)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(phase_space[i] - phase_space[j]) < r:
                count += 1
    print(f'Done for r = {r}')
    return count / (n * (n - 1))


def grassberger_procaccia(ts, t_delay, emb_dim, rs):
    phase_space = phase_space_reconstruction(ts, t_delay, emb_dim)
    c_r = [correlation_sum(phase_space, r) for r in rs]
    print(f'Done for dim = {emb_dim}')
    return c_r


# Загрузка временного ряда радиоизлучения
time_series = np.genfromtxt("data_norh_m1.6_v2.txt", skip_header=1,
                            delimiter='\t', usecols=(1,))
print(len(time_series))

# Параметры
# time_series = lorenz.xs[1]  # [0] - 100, [1] - 1000, [2] - 2000, [3] - 5000
delay = 1  # Задержка
embedding_dim = 5  # Размерность встраивания
r_values = np.logspace(-6, -3, num=30)  # Диапазон значений r (-2, -1, num=40)
# r_values = np.logspace(-1, 1, num=40)  # Диапазон значений r для n = 100

# Выполнение анализа
start_time = time.time()
C_r = grassberger_procaccia(time_series, delay, embedding_dim, r_values)
end_time = time.time() - start_time
print(f"Время выполнения: {end_time} секунд")
print(C_r)

# Определение наклона линейной регрессией
log_r = np.log10(r_values)
log_c_r = np.log10(C_r)
slope, c, r_value, p_value, std_err = linregress(log_r[3:15], log_c_r[3:15])
y_line = slope * log_r[3:15] + c  # уравнение прямой с наклоном slope
print(f"Embedding Dimension = {embedding_dim}, Slope = {slope:.2f}")

# Построение графика
fig, ax = plt.subplots()

ax.scatter(r_values, C_r, marker='x', color='red')
ax.plot(r_values[3:15], 10 ** y_line, label=f'Slope: {slope:.2f}', linewidth=2)

ax.set_title(f'Correlation Sum vs. Radius '
             f'(EmbDim = {embedding_dim}, N = {len(time_series)})')
ax.set_ylabel('log(C(r))')
ax.set_xlabel('log(r)')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(loc='upper left')
ax.grid()
plt.show()
# plt.savefig("last_1.png")
