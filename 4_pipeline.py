import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, ETS, Theta, Prophet
from sklearn.ensemble import RandomForestRegressor

def load_and_prepare():
    df = pd.read_csv('de_hourly_processed.csv', index_col=0, parse_dates=True)
    df = df[['Consumption']].dropna()
    train = df[:-168]
    test = df[-168:]
    return train, test

def run_stats_models(train, test):
    train_ts = train.reset_index()
    train_ts.columns = ['ds', 'y']
    train_ts['unique_id'] = 'DE'
    test_ts = test.reset_index()
    test_ts.columns = ['ds', 'y']
    test_ts['unique_id'] = 'DE'
    models = [AutoARIMA(season_length=24), ETS(season_length=24), Theta(season_length=24), Prophet(season_length=24)]
    sf = StatsForecast(models=models, freq='H')
    sf.fit(train_ts)
    forecasts = sf.predict(h=len(test))
    results = {}
    for model in forecasts.columns[1:]:
        smape_val = np.mean(2 * np.abs(forecasts[model].values - test['y'].values) / (np.abs(forecasts[model].values) + np.abs(test['y'].values)))
        results[model] = round(smape_val, 4)
    return results

def run_ml_model(train, test):
    def create_features(df, target='Consumption'):
        df = df.copy()
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        for lag in [1,2,3,24,48,168]:
            df[f'lag_{lag}'] = df[target].shift(lag)
        df = df.dropna()
        return df
    train_feat = create_features(train)
    test_feat = create_features(test)
    X_train = train_feat.drop(columns=['Consumption'])
    y_train = train_feat['Consumption']
    X_test = test_feat.drop(columns=['Consumption'])
    y_test = test_feat['Consumption']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    smape_val = np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_pred) + np.abs(y_test)))
    return {'RandomForest': round(smape_val, 4)}

def run_pipeline():
    print("Loading data...")
    train, test = load_and_prepare()
    print("Statistical models...")
    stats_res = run_stats_models(train, test)
    print("ML model...")
    ml_res = run_ml_model(train, test)
    all_res = {**stats_res, **ml_res}
    print("\nResults:")
    for k,v in all_res.items():
        print(f"{k}: SMAPE={v}")
    return all_res

if __name__ == "__main__":
    run_pipeline()
