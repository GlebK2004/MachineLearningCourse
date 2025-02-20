from builtins import print, range, len

import numpy as np
import pandas as pd
import statsmodels.tsa.api as smt
from matplotlib import pyplot as plt

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sm import sm

data = pd.read_csv('D:\\Глеб\\Учёба\\4КУРС\\MO\\Витя\\MO_PYTHON\\lacity.org-website-traffic.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

monthly_sessions = data.groupby([pd.Grouper(freq='ME'), 'Device Category'])['Sessions'].sum().unstack().stack()
monthly_sessions_log = np.log(monthly_sessions)

train = monthly_sessions_log.loc[monthly_sessions_log.index.get_level_values(0) < '2019-01-01']
test = monthly_sessions_log.loc[(monthly_sessions_log.index.get_level_values(0) >= '2015-01-01') & (
            monthly_sessions_log.index.get_level_values(0) < '2020-04-01')]

fitted_model = smt.ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit()
test_predictions = fitted_model.forecast(len(test))

test_predictions_orig = np.exp(test_predictions)

mse = mean_squared_error(test, test_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(test, test_predictions)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R^2): {r2}')

plt.figure(figsize=(6, 6))
plt.plot(monthly_sessions.index.get_level_values(0), monthly_sessions.values, label='Actual')
plt.plot(test.index.get_level_values(0), test_predictions_orig, label='Predicted', linestyle='dashed',
             color='orange')
plt.title('Monthly Sessions Forecasting')
plt.xlabel('Date')
plt.ylabel('Sessions')
plt.legend()
plt.show()
