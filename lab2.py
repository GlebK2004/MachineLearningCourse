from builtins import len, print

import numpy as np
import matplotlib.pyplot as plt

# Исходные данные
x = np.array([15, 26, 65, 320, 652, 156, 896, 123, 16, 263, 459, 213])  # Входные значения x
y = np.array([125, 163, 162, 263, 563, 23, 463, 126, 133, 152, 213, 159])  # Выходные значения y

# Расчет коэффициентов регрессии
n = len(x)
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x_squared = np.sum(x**2)

a = (sum_y * sum_x_squared - sum_x * sum_xy) / (n * sum_x_squared - sum_x**2)
b = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)

# Построение линии тренда
y_predicted = a + b * x

# График точечной диаграммы и линии тренда
plt.scatter(x, y, color='blue', label='Данные')
plt.plot(x, y_predicted, color='red', label='Линия тренда')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Точечная диаграмма с линией тренда')
plt.legend()
plt.show()

# Расчет параметров проверки эффективности и состоятельности уравнения регрессии
correlation_coefficient = np.corrcoef(x, y)[0, 1]  # Расчет коэффициента корреляции
residuals = y - y_predicted  # Расчет остатков
standard_error_residual = np.sqrt(np.sum(residuals**2) / (n - 2))  # Расчет стандартной ошибки остатков
standard_error_a = standard_error_residual * np.sqrt(1 / n + (np.mean(x)**2) / (sum_x_squared - n * np.mean(x)**2))  # Расчет стандартной ошибки коэффициента a
standard_error_b = standard_error_residual / np.sqrt(sum_x_squared - n * np.mean(x)**2)  # Расчет стандартной ошибки коэффициента b

t_a = a / standard_error_a
t_b = b / standard_error_b
t_critical = 2.20 #табличное

# Вывод результатов
print(f"Уравнение линейной регрессии: y = {a:.2f} + {b:.2f}x")
print(f"Линейный коэффициент корреляции: {correlation_coefficient:.2f}")
print(f"Стандартная ошибка остаточной компоненты: {standard_error_residual:.2f}")
print(f"Стандартная ошибка коэффициента a: {standard_error_a:.2f}")
print(f"Стандартная ошибка коэффициента b: {standard_error_b:.2f}")
print(f"t-критерий Стьюдента для a: {t_a:.2f}")
print(f"t-критерий Стьюдента для b: {t_b:.2f}")

# Проверка статистической значимости коэффициентов регрессии
if abs(t_a) > t_critical:
    print("Коэффициент a является надежным")
else:
    print("Коэффициент a не является надежным")

if abs(t_b) > t_critical:
    print("Коэффициент b является надежным")
else:
    print("Коэффициент b не является надежным")