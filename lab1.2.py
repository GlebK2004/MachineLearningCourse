import numpy as np
import matplotlib.pyplot as plt

a_values = np.arange(-5, 12.5, 0.5)  # Создаем массив значений a от -5 до 12 с шагом 0.5
x_values = 3.567  # Заданный x

# Вычисляем значения функции для каждого значения a
y_values = 1 / np.tan(x_values**3) + 2.24 * a_values * x_values

# Выводим значения аргумента и значения функции
for a, y in zip(a_values, y_values):
    print(f"a = {a}, f(x) = {y}")

# Находим наибольшее, наименьшее и среднее значения функции
max_value = np.max(y_values)
min_value = np.min(y_values)
mean_value = np.mean(y_values)

print(f"Максимальное значение: {max_value}")
print(f"Минимальное значение: {min_value}")
print(f"Среднее значение: {mean_value}")
print(f"Количество элементов в массиве: {len(y_values)}")

# Сортируем массив (чётные варианты – по убыванию, нечётные – по возрастанию)
if len(a_values) % 2 == 0:
    sorted_values = np.sort(y_values)[::-1]  # Чётные варианты: сортировка по убыванию
else:
    sorted_values = np.sort(y_values)  # Нечётные варианты: сортировка по возрастанию

print("Отсортированный массив:")
print(sorted_values)

# Построение графика функции
plt.plot(a_values, y_values, marker='o', label='f(x)')  # График функции с маркером 'o'
plt.axhline(y=mean_value, color='r', linestyle='--', label='Среднее значение')  # График прямой с маркером '--' и красным цветом
plt.xlabel('Значение a')
plt.ylabel('Значение f(x)')
plt.title('График функции f(x)')
plt.legend()
plt.grid(True)
plt.show()