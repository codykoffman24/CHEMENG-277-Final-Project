import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

# ========================
# 1. Load & Preprocess Data (Discharge, Charge, and OCV)
# ========================
cccv_discharge_file = "Copy of Oxford_battery_data_discharge.csv"
cccv_charge_file = "Copy of Oxford_battery_data_charge.csv"
ocv_discharge_file = "Copy of Oxford_battery_data_OCVdc.csv"
ocv_charge_file = "Copy of Oxford_battery_data_OCVch.csv"

# Load datasets
cccv_discharge_data = pd.read_csv(cccv_discharge_file, low_memory=False)
cccv_charge_data = pd.read_csv(cccv_charge_file, low_memory=False)
ocv_discharge_data = pd.read_csv(ocv_discharge_file, low_memory=False)
ocv_charge_data = pd.read_csv(ocv_charge_file, low_memory=False)

# Sort datasets by cycle order
cccv_discharge_data = cccv_discharge_data.sort_values(by=['cell_number', 'cycle_number', 'time'])
cccv_charge_data = cccv_charge_data.sort_values(by=['cell_number', 'cycle_number', 'time'])
ocv_discharge_data = ocv_discharge_data.sort_values(by=['cell_number', 'cycle_number', 'time'])
ocv_charge_data = ocv_charge_data.sort_values(by=['cell_number', 'cycle_number', 'time'])

# Print total rows in each dataset
print(f"🔹 Total CC-CV discharge rows: {len(cccv_discharge_data)}")
print(f"🔹 Total CC-CV charge rows: {len(cccv_charge_data)}")
print(f"🔹 Total OCV discharge rows: {len(ocv_discharge_data)}")
print(f"🔹 Total OCV charge rows: {len(ocv_charge_data)}")

# ========================
# 2. Verify Cycle Alignment in OCV Dataset
# ========================
print("\n🔍 OCV Data Alignment Check:")
ocv_charge_cycles = ocv_charge_data.groupby('cell_number')['cycle_number'].nunique()
ocv_discharge_cycles = ocv_discharge_data.groupby('cell_number')['cycle_number'].nunique()
ocv_alignment_df = pd.DataFrame({'Charge Cycles': ocv_charge_cycles, 'Discharge Cycles': ocv_discharge_cycles})
print(ocv_alignment_df)

# Merge OCV charge and discharge datasets for proper hysteresis computation
ocv_hysteresis_df = pd.merge(ocv_charge_data, ocv_discharge_data,
                             on=['cell_number', 'cycle_number'],
                             suffixes=('_ch', '_dis'))

# Compute voltage hysteresis as the difference between OCV charge and discharge
ocv_hysteresis_df['OCV_hysteresis'] = ocv_hysteresis_df['voltage_ch'] - ocv_hysteresis_df['voltage_dis']
ocv_hysteresis_df = ocv_hysteresis_df[['cell_number', 'cycle_number', 'OCV_hysteresis']]

# ========================
# 3. Compute SOH from Discharge Data
# ========================
cccv_discharge_data['type'] = 'discharge'
cccv_charge_data['type'] = 'charge'

# Combine both datasets into one continuous time series
full_data = pd.concat([cccv_discharge_data, cccv_charge_data]).sort_values(by=['cell_number', 'cycle_number', 'time'])

# Compute max discharge capacity per cycle
grouped = full_data.groupby(['cell_number', 'cycle_number']).agg({'discharge_capacity': 'max'}).reset_index()
grouped.rename(columns={'discharge_capacity': 'max_discharge_capacity'}, inplace=True)

# Compute SOH as a percentage of the first cycle capacity
def compute_soh(df):
    cycle_1_capacity = df.loc[df['cycle_number'] == 1, 'max_discharge_capacity']
    if cycle_1_capacity.empty or cycle_1_capacity.values[0] <= 0:
        df['SOH'] = np.nan
    else:
        baseline = cycle_1_capacity.values[0]
        df['SOH'] = (df['max_discharge_capacity'] / baseline) * 100
    return df

grouped = grouped.groupby('cell_number', group_keys=False).apply(compute_soh).dropna(subset=['SOH'])

# Merge OCV hysteresis into the dataset
grouped = pd.merge(grouped, ocv_hysteresis_df, on=['cell_number', 'cycle_number'], how='left')

# ========================
# 4. Define Features & Labels
# ========================
features = ['avg_voltage', 'cycle_number', 'OCV_hysteresis']
X = grouped[features].values
y = grouped['SOH'].values

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========================
# 5. Train/Test Split
# ========================
grouped_sorted = grouped.sort_values(by=['cell_number', 'cycle_number'])
X_seq = grouped_sorted[features].values
y_seq = grouped_sorted['SOH'].values
X_seq_scaled = scaler.transform(X_seq)

split_index = int(0.8 * len(X_seq_scaled))
X_train, X_test = X_seq_scaled[:split_index], X_seq_scaled[split_index:]
y_train, y_test = y_seq[:split_index], y_seq[split_index:]

# ========================
# 6. Train Models & Evaluate Performance
# ========================
models = {
    "Linear Regression": linear_model.LinearRegression(),
    "Ridge Regression (alpha=1)": linear_model.Ridge(alpha=1),
    "Lasso Regression (alpha=1)": linear_model.Lasso(alpha=1, max_iter=10000)
}

print("\n--- Model Performance on Time-Based Train/Test Split (80/20) ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name}: R² Test = {model.score(X_test, y_test):.4f}")

# ========================
# 7. Validate Impact of OCV Hysteresis
# ========================
features_no_ocv = ['avg_voltage', 'cycle_number']
X_no_ocv = grouped[features_no_ocv].values
X_no_ocv_scaled = scaler.fit_transform(X_no_ocv)

print("\n--- Impact of OCV Hysteresis on Model Performance ---")
for name, model in models.items():
    model.fit(X_no_ocv_scaled, y_train)
    print(f"{name} Without OCV Hysteresis: R² Test = {model.score(X_test, y_test):.4f}")

# ========================
# 8. Feature Importance & Visualization
# ========================
best_model_name = max(models, key=lambda name: models[name].score(X_test, y_test))
best_model = models[best_model_name]

print(f"\n🔹 Best Performing Model: {best_model_name}")
print(f"R² on Test Set: {best_model.score(X_test, y_test):.4f}")

coefficients = best_model.coef_
feature_importance = dict(zip(features, coefficients))
sorted_importance = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)

print("\n🔹 Feature Coefficients (Sorted by Impact):")
for feature, coef in sorted_importance:
    print(f"{feature}: {coef:.4f}")

# Plot feature importance
plt.figure(figsize=(8, 5))
plt.barh([f[0] for f in sorted_importance], [f[1] for f in sorted_importance], color='blue')
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.title(f"Feature Importance in {best_model_name}")
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()

