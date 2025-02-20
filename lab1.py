import pandas as pd
import matplotlib.pyplot as plt

# Загрузка датасета из CSV-файла
df = pd.read_csv('D:\\Глеб\\Учёба\\4КУРС\\MO\\Walmart.csv')

# Взятие первых 1000 значений
df = df.head(1000)

# Проверка на пропуски
missing_values = df.isnull().sum()
print("Пропущенные значения:")
print(missing_values)

# Проверка на нормальность распределения и обработка выбросов
columns_to_check = ['Store', 'Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
for column in columns_to_check:
    data = df[column].dropna()  # Удаление пропусков

    # Построение ящика с усами
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.boxplot(data, vert=False)  # Ящик с усами
    plt.title(f'Ящик с усами для {column}')

    # Построение гистограммы
    plt.subplot(1, 2, 2)
    plt.hist(data, bins=30)  # Гистограмма
    plt.title(f'Гистограмма для {column}')
    plt.show()

    # Заполнение пропусков средним значением
    df[column] = df[column].fillna(df[column].mean())

    # Обработка аномальных значений
    lower_bound = df[column].mean() - 3 * df[column].std()  # Нижняя граница
    upper_bound = df[column].mean() + 3 * df[column].std()  # Верхняя граница
    df = df.loc[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Построение сводной таблицы
pivot_table = pd.pivot_table(df, values='Weekly_Sales', index='Store', aggfunc=['mean', 'median', 'min', 'max'])
print("Сводная таблица:")
print(pivot_table)