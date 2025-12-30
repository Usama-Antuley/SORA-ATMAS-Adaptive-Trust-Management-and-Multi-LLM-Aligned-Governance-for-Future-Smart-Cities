# -*- coding: utf-8 -*-
"""
Train a lightweight XGBoost weather classifier for Karachi:
- Load -> Clean -> EDA -> Pseudo-label (rule-based) -> Train -> Evaluate -> Save
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, balanced_accuracy_score
)
import joblib

# Try XGBoost; fallback to HistGradientBoosting if not installed
use_xgb = True
try:
    from xgboost import XGBClassifier
except Exception:
    use_xgb = False
    from sklearn.ensemble import HistGradientBoostingClassifier

# -----------------------------
# PATHS (edit only filename)
# -----------------------------
DATA_DIR = Path(r"C:\Users\FMT COMPUTERS\Downloads\Code Paper 3")
FILE_PATH = DATA_DIR / "weather data open meteo.csv"
OUT_DIR  = DATA_DIR

print("Reading:", FILE_PATH)
raw = pd.read_csv(FILE_PATH)

# -----------------------------
# LOAD & CLEAN
# -----------------------------
# Find the first numeric 'time' row; skip metadata header rows
start_row = raw[raw.iloc[:, 0].astype(str).str.isnumeric()].index.min()
df = pd.read_csv(FILE_PATH, skiprows=start_row)

# Time to Karachi local date
df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
df['date'] = df['time'].dt.tz_convert('Asia/Karachi').dt.date
df = df.sort_values('time').reset_index(drop=True)

def pick_col(cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

col_temp_max   = pick_col(['temperature_2m_max_HiRAM_SIT_HR (°C)',
                           'temperature_2m_max_CMCC_CM2_VHR4 (°C)'])
col_precip_hrm = pick_col(['rain_sum_HiRAM_SIT_HR (mm)'])
col_precip_cmc = pick_col(['precipitation_sum_CMCC_CM2_VHR4 (mm)',
                           'precipitation_sum (mm)'])
col_rh_mean    = pick_col(['relative_humidity_2m_mean_HiRAM_SIT_HR (%)',
                           'relative_humidity_2m_mean_CMCC_CM2_VHR4 (%)'])
col_wind_mean  = pick_col(['wind_speed_10m_mean_HiRAM_SIT_HR (km/h)',
                           'wind_speed_10m_mean_CMCC_CM2_VHR4 (km/h)'])
col_soil_moist = pick_col(['soil_moisture_0_to_10cm_mean_CMCC_CM2_VHR4 (m³/m³)',
                           'soil_moisture_0_to_7cm_mean_CMCC_CM2_VHR4 (m³/m³)'])
col_rad_mj     = pick_col(['shortwave_radiation_sum_HiRAM_SIT_HR (MJ/m²)',
                           'shortwave_radiation_sum_CMCC_CM2_VHR4 (MJ/m²)'])

work = pd.DataFrame({
    'date': df['date'],
    'temp_max': pd.to_numeric(df[col_temp_max], errors='coerce') if col_temp_max else np.nan,
    'rain_hiram': pd.to_numeric(df[col_precip_hrm], errors='coerce') if col_precip_hrm else np.nan,
    'rain_cmcc': pd.to_numeric(df[col_precip_cmc], errors='coerce')  if col_precip_cmc else np.nan,
    'rh_mean': pd.to_numeric(df[col_rh_mean], errors='coerce') if col_rh_mean else np.nan,
    'wind_mean_kmh': pd.to_numeric(df[col_wind_mean], errors='coerce') if col_wind_mean else np.nan,
    'soil_moist': pd.to_numeric(df[col_soil_moist], errors='coerce') if col_soil_moist else np.nan,
    'rad_sum_mj': pd.to_numeric(df[col_rad_mj], errors='coerce') if col_rad_mj else np.nan,
})

# Daily aggregation (literature thresholds are daily)
daily = work.groupby('date', as_index=False).agg({
    'temp_max': 'max',
    'rain_hiram': 'sum',
    'rain_cmcc': 'sum',
    'rh_mean': 'mean',
    'wind_mean_kmh': 'mean',
    'soil_moist': 'mean',
    'rad_sum_mj': 'sum'
})

# Prefer HiRAM; fill with CMCC when HiRAM is 0/NaN but CMCC>0
daily['precip_mm'] = daily['rain_hiram']
mask = (daily['precip_mm'].isna()) | ((daily['precip_mm'] == 0) & (daily['rain_cmcc'].fillna(0) > 0))
daily.loc[mask, 'precip_mm'] = daily.loc[mask, 'rain_cmcc']

# Radiation MJ/m²/day -> mean W/m² over the day
daily['rad_wm2'] = (daily['rad_sum_mj'] * 1e6) / 86400.0

# DOY climatology for Tmax and anomaly
daily['date_ts'] = pd.to_datetime(daily['date'])
daily['doy'] = daily['date_ts'].dt.dayofyear
clim = daily.groupby('doy', as_index=False)['temp_max'].mean().rename(columns={'temp_max':'temp_max_clim'})
daily = daily.merge(clim, on='doy', how='left')
daily['tmax_anom'] = daily['temp_max'] - daily['temp_max_clim']

# Antecedent rainfall
daily = daily.sort_values('date').reset_index(drop=True)
daily['rain_3d'] = daily['precip_mm'].rolling(3, min_periods=1).sum()
daily['rain_5d'] = daily['precip_mm'].rolling(5, min_periods=1).sum()
daily['rain_prev5'] = daily['precip_mm'].shift(1).rolling(5, min_periods=1).sum()

print("Daily shape:", daily.shape)
print(daily.head(3))

# -----------------------------
# EDA (minimal but useful)
# -----------------------------
print("\nMissing values:")
print(daily.isna().sum().sort_values(ascending=False))

plt.figure()
plt.plot(daily['date_ts'], daily['precip_mm'])
plt.title('Daily Precipitation (mm)')
plt.xlabel('Date'); plt.ylabel('mm'); plt.tight_layout(); plt.show()

plt.figure()
plt.plot(daily['date_ts'], daily['temp_max'])
plt.title('Daily Max Temperature (°C)')
plt.xlabel('Date'); plt.ylabel('°C'); plt.tight_layout(); plt.show()

# -----------------------------
# RULES -> pseudo-labels
# -----------------------------
# Literature-backed defaults (tune in your paper if you prefer IMD heavy=64.5 mm)
RAINY_DAY_MM    = 2.5
HEAVY_RAIN_MM   = 50.0        # set 64.5 to match IMD strictly
FLOOD_RAIN_MM   = 50.0
FLOOD_3DAY_MM   = 100.0
FLOOD_5DAY_MM   = 150.0
SOIL_SAT        = 0.35        # 35% volumetric water
HEATWAVE_ABS_C  = 40.0        # PMD plains absolute criterion
HEATWAVE_ANOM_C = 5.0         # WMO anomaly criterion
USE_HUMIDITY_FOR_HEAT = False
HEAT_HUMIDITY_MAX = 30.0

def label_day(r):
    rain = 0.0 if pd.isna(r['precip_mm']) else r['precip_mm']
    # Heatwave
    is_heat = (pd.notnull(r['temp_max']) and pd.notnull(r['tmax_anom']) and
               (r['temp_max'] >= HEATWAVE_ABS_C) and (r['tmax_anom'] >= HEATWAVE_ANOM_C))
    if USE_HUMIDITY_FOR_HEAT and pd.notnull(r['rh_mean']):
        is_heat = is_heat and (r['rh_mean'] < HEAT_HUMIDITY_MAX)
    # Flooding (compound)
    is_flood = (rain >= FLOOD_RAIN_MM) and (
        (pd.notnull(r['soil_moist']) and r['soil_moist'] >= SOIL_SAT) or
        (pd.notnull(r['rain_3d']) and r['rain_3d'] >= FLOOD_3DAY_MM) or
        (pd.notnull(r['rain_5d']) and r['rain_5d'] >= FLOOD_5DAY_MM) or
        (pd.notnull(r['rain_prev5']) and r['rain_prev5'] >= FLOOD_5DAY_MM)
    )
    is_heavy = (rain >= HEAVY_RAIN_MM)
    is_rain  = (rain >= RAINY_DAY_MM) and (rain < HEAVY_RAIN_MM)

    # Priority
    if is_flood:   return 'Flooding'
    elif is_heavy: return 'Heavy Rain'
    elif is_rain:  return 'Rain'
    elif is_heat:  return 'Heatwave'
    else:          return 'Normal'

daily['label'] = daily.apply(label_day, axis=1)
print("\nClass counts:\n", daily['label'].value_counts())

plt.figure()
daily['label'].value_counts().sort_index().plot(kind='bar')
plt.title('Class Distribution (Pseudo-labels)')
plt.xlabel('Class'); plt.ylabel('Count'); plt.tight_layout(); plt.show()

# -----------------------------
# TRAIN model
# -----------------------------
candidate_features = ['precip_mm','rain_3d','rain_5d','tmax_anom','temp_max',
                      'rh_mean','wind_mean_kmh','soil_moist','rad_wm2']
feature_cols = [c for c in candidate_features if c in daily.columns]

X = daily[feature_cols].copy()
y = daily['label'].copy()

# Split (try to stratify; if class rare, fallback)
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

preprocess = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), feature_cols)
])

if use_xgb:
    model = XGBClassifier(
        objective='multi:softprob',
        num_class=5,
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        tree_method='hist'
    )
else:
    model = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.05, random_state=42
    )

clf = Pipeline([('prep', preprocess), ('model', model)])
clf.fit(X_train, y_train)

# -----------------------------
# EVALUATE (clean metrics)
# -----------------------------
y_pred_train = clf.predict(X_train)
y_pred_test  = clf.predict(X_test)

print("\n=== TRAIN METRICS ===")
print("Accuracy:", accuracy_score(y_train, y_pred_train))
print("Balanced Acc.:", balanced_accuracy_score(y_train, y_pred_train))
print("Macro F1:", f1_score(y_train, y_pred_train, average='macro'))

print("\n=== TEST METRICS ===")
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("Balanced Acc.:", balanced_accuracy_score(y_test, y_pred_test))
print("Macro F1:", f1_score(y_test, y_pred_test, average='macro'))
print("\nClassification report:\n", classification_report(y_test, y_pred_test, zero_division=0))

labels_order = ['Normal','Rain','Heavy Rain','Flooding','Heatwave']
cm = confusion_matrix(y_test, y_pred_test, labels=labels_order)
plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion Matrix (Test)')
plt.xticks(range(len(labels_order)), labels_order, rotation=45, ha='right')
plt.yticks(range(len(labels_order)), labels_order)
plt.xlabel('Predicted'); plt.ylabel('True')
plt.colorbar(); plt.tight_layout(); plt.show()

# -----------------------------
# SAVE artifacts
# -----------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
labeled_csv = OUT_DIR / "karachi_weather_labeled.csv"
daily[['date'] + feature_cols + ['label']].to_csv(labeled_csv, index=False)

model_path = OUT_DIR / ("xgb_weather_pipeline.pkl" if use_xgb else "weather_fallback_pipeline.pkl")
joblib.dump(clf, model_path)

print("\nSaved:")
print(" - Labeled dataset:", labeled_csv)
print(" - Trained pipeline:", model_path)
