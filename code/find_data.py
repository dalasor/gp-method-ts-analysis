from sunpy.net import Fido, attrs as a
import astropy.units as u
from sunpy.timeseries import TimeSeries
from sunpy.time import TimeRange
import matplotlib.pyplot as plt
import numpy as np

# Задание параметров запроса, вспышка M1.6
tstart = "2010-11-04 22:00"
tend = "2010-11-05 07:00"
frequency = 17 * u.GHz

''' Используем Fido для поиска данных NoRH. Данные - кривая блеска, 
    только вместо интенсивности корреляционные коэффициенты,
    1% корреляции = 30 SFU (Solar Flux Unit), 1 SFU = 1e-22 W / (Hz * m^2). '''
result = Fido.search(a.Time(tstart, tend), a.Wavelength(frequency),
                     a.Instrument('norh'))

# скачиваем второй файл, в который полностью уместилась вспышка
files = Fido.fetch(result[0][1])

ts = TimeSeries(files)  # распаковываем временной ряд

# Обрезаем временной ряд
tr = TimeRange('2010-11-04 23:54', '2010-11-05 01:20')
ts = ts.truncate(tr)

# Добавим новую колонку для интенсивности
intensity = 30 * ts.quantity("Correlation Coefficient") * 1.e-22 * 17 * 10**9 / 0.01
ts = ts.add_column("Intensity", intensity)

# Переводим из специфическго формата в датафрейм
ts_df = ts.to_dataframe()
print(ts_df.keys())
print(ts_df.mean())
print(ts_df.max().iloc[0])

# Выгружаем данные в файл для последующего анализа
ts_df.index.rename('Time', inplace=True)
ts_df = ts_df.rename(columns={"Correlation Coefficient": "Correlation"})
ts_df.to_csv('data_norh_m1.6_v2.txt', sep='\t', float_format='%.8e',
             date_format='%Y-%m-%d %H:%M:%S')

fig, ax = plt.subplots()
ts.plot(axes=ax)
plt.show()


