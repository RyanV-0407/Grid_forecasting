import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler
import joblib
import time
from feature_engineering import add_time_features
from hyperparameter_tuning import tune_hyperparameters

def main():
    # Load and prepare data for time series Analysis
    df = pd.read_csv("realistic_energy_forecast_dataset.csv", parse_dates=["Timestamp"])
    df = add_time_features(df)

    # Create forecast horizons for multi-horizon forecasting in grid stabilization.
    forecast_horizons = [1, 3, 6]
    for h in forecast_horizons:
        df[f"target_{h}h"] = df["Energy_Consumption_kWh"].shift(-h)
    df = df.dropna()  # Drop rows with missing values

    # Define features and targets, in total 22 features
    features = [
        "Temperature_C", "Humidity_%", "Wind_Speed_mps", "Solar_Radiation_Wm2",
        "Industrial_Usage_kWh", "Residential_Usage_kWh", "Commercial_Usage_kWh",
        "Grid_Frequency_Hz", "Voltage_Level_V", "Renewable_Energy_Contribution_%",
        "hour", "dayofweek", "month", "Holiday_Indicator", "Weekday",
        "is_peak", "temp_usage", "solar_wind", "rolling_demand_3h",
        "rolling_demand_24h", "heat_index", "effective_temp"
    ]
    targets = [f"target_{h}h" for h in forecast_horizons]

    X = df[features]
    y = df[targets]

    # Train-test split - 80% train, 20% test
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # RobustScaler for Feature Scaling and to handle outliers
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "feature_scaler.pkl")

    # Hyperparameter tuning using Optuna for LightGBM
    best_params = tune_hyperparameters(X_train_scaled, y_train[targets[0]], 
                                    X_test_scaled, y_test[targets[0]], n_trials=50)

    # Train models
    models = {}
    metrics = {}
    for target in targets:
        train_data = lgb.Dataset(X_train_scaled, label=y_train[target])
        test_data = lgb.Dataset(X_test_scaled, label=y_test[target], reference=train_data)

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': best_params.get('learning_rate', 0.1),
            'num_leaves': best_params.get('num_leaves', 127),
            'max_depth': best_params.get('max_depth', 8),
            'min_data_in_leaf': best_params.get('min_data_in_leaf', 20),
            'subsample': best_params.get('subsample', 0.8),
            'reg_alpha': best_params.get('reg_alpha', 0.01),
            'reg_lambda': best_params.get('reg_lambda', 0.01),
            'verbose': -1
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(50)]
        )
        models[target] = model
        joblib.dump(model, f"energy_forecast_{target}.pkl")

        # Calculate metrics
        predictions = model.predict(X_test_scaled)
        metrics[target] = {
            'r2': r2_score(y_test[target], predictions),
            'rmse': np.sqrt(mean_squared_error(y_test[target], predictions)),
            'mae': mean_absolute_error(y_test[target], predictions),
            'train_time': time.time() - start_time,
            'data_points': len(X_train)
        }

    # Save artifacts
    joblib.dump(features, "model_features.pkl")
    joblib.dump(metrics, "model_metrics.pkl")
    joblib.dump(df['Energy_Consumption_kWh'].iloc[-1], "last_demand.pkl")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Training completed in {time.time()-start_time:.2f} seconds")