import pandas as pd
import numpy as np

def add_time_features(df):
    """Enhanced feature engineering with temporal patterns"""
    # Basic time features
    df["hour"] = df["Timestamp"].dt.hour
    df["dayofweek"] = df["Timestamp"].dt.dayofweek
    df["month"] = df["Timestamp"].dt.month
    df["Weekday"] = df["dayofweek"].apply(lambda x: 1 if x < 5 else 0)
    df["is_peak"] = df["hour"].between(17, 21).astype(int)

    # Interaction features
    df["temp_usage"] = df["Temperature_C"] * (
        df["Residential_Usage_kWh"] + df["Commercial_Usage_kWh"]
    )
    df["solar_wind"] = df["Solar_Radiation_Wm2"] * df["Wind_Speed_mps"]
    
    # Rolling features
    df["rolling_demand_3h"] = df["Energy_Consumption_kWh"].rolling(3, min_periods=1).mean().shift(1)
    df["rolling_demand_24h"] = df["Energy_Consumption_kWh"].rolling(24, min_periods=1).mean().shift(1)
    
    # Weather indices
    df["heat_index"] = 0.5 * (
        df["Temperature_C"] + 61.0 + 
        ((df["Temperature_C"] - 68.0) * 1.2) + 
        (df["Humidity_%"] * 0.094)
    )
    df["effective_temp"] = df["Temperature_C"] * df["Humidity_%"] / 100
    
    return df.dropna()