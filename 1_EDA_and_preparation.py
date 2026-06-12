import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('figures', exist_ok=True)

url = "https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv"
print("Загружаем данные...")
df = pd.read_csv(url, parse_dates=['Date'], index_col='Date')
print(f"Размер данных: {df.shape}")

df['Consumption'] = df['Consumption'].interpolate(method='linear')

df['year'] = df.index.year
df['month'] = df.index.month
df['dayofweek'] = df.index.dayofweek
df['dayofyear'] = df.index.dayofyear

plt.figure(figsize=(14,6))
plt.plot(df.index, df['Consumption'], color='blue', linewidth=0.7)
plt.title('Потребление электроэнергии в Германии (дневные данные)')
plt.ylabel('МВт')
plt.grid(alpha=0.3)
plt.savefig('figures/consumption_time_series.png', dpi=150, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2,1,figsize=(12,8))
sns.boxplot(data=df, x='dayofweek', y='Consumption', ax=axes[0])
axes[0].set_title('Потребление по дням недели (0=пн,6=вс)')
sns.boxplot(data=df, x='month', y='Consumption', ax=axes[1])
axes[1].set_title('Потребление по месяцам')
plt.tight_layout()
plt.savefig('figures/seasonality_plots.png', dpi=150, bbox_inches='tight')
plt.close()

result = adfuller(df['Consumption'].dropna())
with open('figures/adf_test_result.txt', 'w') as f:
    f.write(f"ADF p-value: {result[1]:.6f}\n")
    f.write("Нестационарный" if result[1] > 0.05 else "Стационарный")

fig, axes = plt.subplots(2,1,figsize=(12,8))
plot_acf(df['Consumption'].dropna(), lags=60, ax=axes[0])
plot_pacf(df['Consumption'].dropna(), lags=60, ax=axes[1])
plt.savefig('figures/acf_pacf.png', dpi=150, bbox_inches='tight')
plt.close()

df.to_csv('de_hourly_processed.csv')
print("✅ Готово. Файл de_hourly_processed.csv и графики в figures/")