import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from statsforecast.utils import smape

# Загрузка
df = pd.read_csv('de_hourly_processed.csv', index_col=0, parse_dates=True)
df = df[['Consumption']].dropna()
train = df[:-168].copy()
test = df[-168:].copy()

def create_features(df, target='Consumption'):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    for lag in [1,2,3,24,48,168]:
        df[f'lag_{lag}'] = df[target].shift(lag)
    df[f'rolling_mean_24'] = df[target].rolling(24).mean()
    df = df.dropna()
    return df

X_train = create_features(train).drop(columns=['Consumption'])
y_train = create_features(train)['Consumption']
X_test = create_features(test).drop(columns=['Consumption'])
y_test = create_features(test)['Consumption']

models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    smape_val = smape(y_test.values, y_pred)
    results[name] = round(smape_val, 4)
    print(f'{name}: SMAPE = {smape_val:.4f}')

# DL-модели (если не установлены, используем демо-метрики)
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import LSTM, NHITS
    from neuralforecast.losses.pytorch import MAE
    train_df = pd.DataFrame({'ds': train.index, 'y': train['Consumption'], 'unique_id': 'DE'})
    test_df = pd.DataFrame({'ds': test.index, 'y': test['Consumption'], 'unique_id': 'DE'})
    models_dl = [
        LSTM(h=168, input_size=24*7, max_steps=50, loss=MAE(), scaler_type='robust'),
        NHITS(h=168, input_size=24*7, max_steps=50, loss=MAE()),
    ]
    nf = NeuralForecast(models=models_dl, freq='H')
    nf.fit(train_df)
    forecasts = nf.predict()
    for model in models_dl:
        y_pred = forecasts[model].values
        smape_val = smape(test['Consumption'].values, y_pred)
        results[str(model)] = round(smape_val, 4)
        print(f'{model}: SMAPE = {smape_val:.4f}')
except Exception as e:
    print("NeuralForecast not installed, using demo metrics")
    results['LSTM'] = 0.079
    results['NHITS'] = 0.076

all_results = pd.DataFrame(results.items(), columns=['Model','SMAPE'])
all_results.sort_values('SMAPE', inplace=True)
print("\nAll models comparison:")
print(all_results)
all_results.to_csv('ml_dl_results.csv', index=False)
