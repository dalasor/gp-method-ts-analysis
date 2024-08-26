import lorenz
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
    # print(f'Done for r = {r}')
    return count / (n * (n - 1))


def grassberger_procaccia(ts, t_delay, emb_dim, rs):
    dims = [i for i in range(7, emb_dim + 1)]
    c_r = []
    for dim in dims:
        phase_space = phase_space_reconstruction(ts, t_delay, dim)
        c_r.append([correlation_sum(phase_space, r) for r in rs])
        print(f'Done for dim = {dim}')
    return dims, c_r


# Параметры
time_series = lorenz.xs[1]  # [0] - 100, [1] - 1000, [1] - 2000, [3] - 5000
delay = 1  # Задержка
embedding_dim = 10  # Макс размерность встраивания
r_values = np.logspace(-1.8, 1.2, num=40)  # Диапазон значений r
# r_values = np.logspace(-1, 1, num=40)  # Диапазон значений r для n = 100

# Выполнение анализа
start_time = time.time()
Dims, C_r = grassberger_procaccia(time_series, delay, embedding_dim, r_values)
end_time = time.time() - start_time
print(f"Время выполнения: {end_time} секунд")

log_r = np.log10(r_values)  # figsize=(10, 7)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,
                                             sharex=True, sharey=True)
fig.suptitle(f'Correlation Sum vs. Radius (N = {len(time_series)-1})')
fig.tight_layout()
ax1.set_ylabel('log(C(r))')
ax3.set_ylabel('log(C(r))')
ax3.set_xlabel('log(r)')
ax4.set_xlabel('log(r)')
ax1.set_xscale('log')
ax1.set_yscale('log')

axs = [ax1, ax2, ax3, ax4]

for k in range(len(Dims)):
    log_c_r = np.log10(C_r[k])
    # Определение наклона линейной регрессией
    slope, c, r_value, p_value, std_err = linregress(log_r[5:20], log_c_r[5:20])
    y_line = slope * log_r[5:20] + c  # уравнение прямой с наклоном slope
    # Построение графиков
    axs[k].scatter(r_values, C_r[k], marker='x', color='red')
    axs[k].plot(r_values[5:20], 10 ** y_line, label=f'Slope: {slope:.2f}')
    axs[k].set_title(f'Embedding Dimension = {Dims[k]}')
    axs[k].legend(loc='upper left')
    axs[k].grid()
    # Вывод наклонов для каждой размерности
    print(f"Dimension {embedding_dim}: Slope = {slope:.2f}")

plt.show()


# # Или с помощью метода наименьших квадратов
# A = np.vstack([log_r, np.ones(len(log_r))]).T
# m, c = np.linalg.lstsq(A, log_c_r, rcond=None)[0]
# print("Наклон по МНК:", m)
# print("Пересечение по МНК:", c)
