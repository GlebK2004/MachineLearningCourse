import multiprocessing
from builtins import print, range, len

import numpy as np
import pandas as pd
import sns
import statsmodels.tsa.api as smt
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("D:\\Глеб\\Учёба\\4КУРС\\MO\\drug200.csv")

data_encoded = pd.get_dummies(data, columns=['Sex', 'BP', 'Cholesterol'])


X = data_encoded.drop('Drug', axis=1)
y = data_encoded['Drug']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Визуализация тепловой карты
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap='Blues')

# Настройка меток осей
plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['drugA', 'drugB', 'drugC', 'drugX', 'drugY'])
plt.yticks(ticks=[0, 1, 2, 3, 4], labels=['drugA', 'drugB', 'drugC', 'drugX', 'drugY'])

# Добавление аннотаций
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, conf_matrix[i, j], ha='center', va='center', color='white')

# Настройка меток и заголовка
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Отображение тепловой карты
plt.show()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')