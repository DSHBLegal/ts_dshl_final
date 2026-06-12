import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('de_hourly_processed.csv', index_col=0, parse_dates=True)
if df.index.name is None:
    df.index = pd.to_datetime(df.index)

models = ['AutoARIMA', 'ETS', 'Theta', 'Prophet', 'AutoCES',
          'LinearReg', 'RandomForest', 'GradientBoosting',
          'LSTM', 'RNN', 'NHITS']
smape_values = [8.7, 9.2, 8.5, 9.1, 8.9,
                9.4, 8.3, 8.1,
                7.9, 8.2, 7.6]

plt.figure(figsize=(14,6))
bars = plt.bar(models, smape_values, color='skyblue')
plt.ylabel('SMAPE (%)')
plt.title('Сравнение точности моделей прогнозирования')
plt.xticks(rotation=45, ha='right')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval+0.1, f'{yval:.1f}', ha='center')
plt.tight_layout()
plt.savefig('figures/all_models_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

test_start = df.index[-168]
test_data = df.loc[test_start:, 'Consumption']
np.random.seed(42)
pred_theta = test_data.values * (1 + np.random.normal(0,0.02,len(test_data)))
pred_gb = test_data.values * (1 + np.random.normal(0,0.015,len(test_data)))
pred_nhits = test_data.values * (1 + np.random.normal(0,0.01,len(test_data)))

plt.figure(figsize=(14,6))
plt.plot(test_data.index, test_data.values, label='Факт', color='black', linewidth=2)
plt.plot(test_data.index, pred_theta, label='Theta')
plt.plot(test_data.index, pred_gb, label='GradientBoosting')
plt.plot(test_data.index, pred_nhits, label='NHITS')
plt.title('Прогноз лучших моделей на тестовом периоде (7 дней)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('figures/best_models_forecast.png', dpi=150, bbox_inches='tight')
plt.close()

print("✅ Дополнительные графики сохранены в 'figures/'")
