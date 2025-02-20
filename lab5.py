from builtins import print, range, len

import numpy as np
import pandas as pd
import sns
import statsmodels.tsa.api as smt
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("D:\\Глеб\\Учёба\\4КУРС\\MO\\Naive-Bayes-Classification-Data.csv")
# Разделение данных на признаки (X) и целевую переменную (y)
X = data[['glucose', 'bloodpressure']]
y = data['diabetes']

# Обучение логистической регрессии
log_reg = LogisticRegression()
log_reg.fit(X, y)

# Визуализация данных
plt.scatter(x=data['glucose'], y=data['bloodpressure'], c=data['diabetes'])
# Настройка осей и меток
plt.xlabel('Glucose')
plt.ylabel('Blood Pressure')

# Визуализация разделяющей линии (предсказанные классы)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx, yy = np.meshgrid(np.linspace(*xlim, 100),
                     np.linspace(*ylim, 100))
Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.title('Logistic Regression Decision Boundary')
plt.show()
X = data[['bloodpressure']]
y = data['glucose']

# Обучение линейной регрессии
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Предсказание уровня глюкозы
glucose_pred = lin_reg.predict(X)

# Визуализация результата
plt.scatter(X, y, color='blue')
plt.plot(X, glucose_pred, color='red')
plt.xlabel('Blood Pressure')
plt.ylabel('Glucose Level')
plt.title('Linear Regression: Glucose Prediction')
plt.show()


def task2():
    X = data[['bloodpressure']]
    y = data['glucose']

    # Обучение линейной регрессии
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)

    # Предсказание уровня глюкозы
    glucose_pred = lin_reg.predict(X)

    # Визуализация результата
    plt.scatter(X, y, color='blue')
    plt.plot(X, glucose_pred, color='red')
    plt.xlabel('Blood Pressure')
    plt.ylabel('Glucose Level')
    plt.title('Linear Regression: Glucose Prediction')
    plt.show()


task2()