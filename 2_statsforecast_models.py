import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, ETS, Theta, Prophet, AutoCES
from statsforecast.utils import smape

# Загрузка
df = pd.read_csv('de_hourly_processed.csv', index_col=0, parse_dates=True)
df = df[['Consumption']].dropna()
df_ts = df.reset_index()
df_ts.columns = ['ds', 'y']
df_ts['unique_id'] = 'DE'

train = df_ts[:-168]
test = df_ts[-168:]

models = [
    AutoARIMA(season_length=24),
    ETS(season_length=24),
    Theta(season_length=24),
    Prophet(season_length=24),
    AutoCES(season_length=24),
]
sf = StatsForecast(models=models, freq='H', n_jobs=-1)
sf.fit(train)
forecasts = sf.predict(h=168)

y_true = test['y'].values
results = {}
for model in forecasts.columns[1:]:
    y_pred = forecasts[model].values
    smape_val = smape(y_true, y_pred)
    results[model] = round(smape_val, 4)
    print(f'{model}: SMAPE = {smape_val:.4f}')

best = min(results, key=results.get)
print(f'\nBest model: {best} (SMAPE={results[best]})')

plt.figure(figsize=(12,5))
plt.plot(test['ds'], y_true, label='Actual')
plt.plot(test['ds'], forecasts[best], label=f'{best} forecast')
plt.legend()
plt.show()

pd.DataFrame(results.items(), columns=['Model','SMAPE']).to_csv('stats_models_results.csv', index=False)
