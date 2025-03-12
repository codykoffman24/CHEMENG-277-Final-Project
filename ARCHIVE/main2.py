import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import gc  # For garbage collection

# ========================
# 1. Load & Preprocess Data (Discharge, Charge, and OCV)
# ========================
discharge_file = "Copy of Oxford_battery_data_discharge.csv"
charge_file = "Copy of Oxford_battery_data_charge.csv"
ocv_discharge_file = "Copy of Oxford_battery_data_OCVdc.csv"
ocv_charge_file = "Copy of Oxford_battery_data_OCVch.csv"

# Define function to process data in chunks
def load_and_process_in_chunks(filename, usecols, chunk_size=250000):
    chunks = pd.read_csv(filename, chunksize=chunk_size, usecols=usecols, low_memory=False)
    result = pd.DataFrame()
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1} from {filename}...")
        result = pd.concat([result, chunk])
    return result

# Load discharge data (needed for full processing)
print("Loading discharge data...")
discharge_data = load_and_process_in_chunks(
    discharge_file,
    usecols=['cell_number', 'cycle_number', 'time', 'voltage', 'current', 'temperature']
    # Added temperature column
)

# Sort discharge data
discharge_data = discharge_data.sort_values(by=['cell_number', 'cycle_number', 'time'])

# Print total rows
print(f"🔹 Total CC-CV discharge rows: {len(discharge_data)}")

# Ensure current values are negative for discharge
if discharge_data['current'].max() > 0:
    discharge_data['current'] = -discharge_data['current']

# Set time_diff to 1 second as per requirement
discharge_data['time_diff'] = 1

# Compute discharge capacity incrementally
discharge_data['incremental_discharge'] = (discharge_data['current'] * discharge_data['time_diff']).abs() / 3600
discharge_data['discharge_capacity'] = discharge_data.groupby(['cell_number', 'cycle_number'])['incremental_discharge'].cumsum() * 1000

# ========================
# 2. Aggregate Discharge Data First
# ========================
print("Aggregating discharge data...")
agg_funcs = {
    'voltage': 'mean',
    'temperature': 'mean',  # Aggregate temperature
    'discharge_capacity': 'max'
}

grouped = discharge_data.groupby(['cell_number', 'cycle_number']).agg(agg_funcs).reset_index()
grouped.rename(columns={
    'voltage': 'avg_voltage',
    'temperature': 'avg_temperature',  # Rename temperature
    'discharge_capacity': 'max_discharge_capacity'
}, inplace=True)

# Free up memory
del discharge_data
gc.collect()

# ========================
# 3. Process OCV Data More Efficiently
# ========================
print("Processing OCV data for hysteresis calculation...")

# Load and pre-aggregate OCV data
print("Loading and aggregating OCV discharge data...")
ocv_discharge_agg = load_and_process_in_chunks(
    ocv_discharge_file,
    usecols=['cell_number', 'cycle_number', 'voltage'],
    chunk_size=500000
).groupby(['cell_number', 'cycle_number']).agg({'voltage': 'mean'}).reset_index()

print("Loading and aggregating OCV charge data...")
ocv_charge_agg = load_and_process_in_chunks(
    ocv_charge_file,
    usecols=['cell_number', 'cycle_number', 'voltage'],
    chunk_size=500000
).groupby(['cell_number', 'cycle_number']).agg({'voltage': 'mean'}).reset_index()

print(f"🔹 Total OCV discharge aggregated rows: {len(ocv_discharge_agg)}")
print(f"🔹 Total OCV charge aggregated rows: {len(ocv_charge_agg)}")

# Calculate hysteresis from aggregated data
ocv_hysteresis_df = pd.merge(
    ocv_charge_agg,
    ocv_discharge_agg,
    on=['cell_number', 'cycle_number'],
    suffixes=('_ch', '_dis')
)

# Compute hysteresis feature
ocv_hysteresis_df['hysteresis_voltage'] = ocv_hysteresis_df['voltage_ch'] - ocv_hysteresis_df['voltage_dis']
ocv_hysteresis_df.drop(columns=['voltage_ch', 'voltage_dis'], inplace=True)

# Free up memory
del ocv_charge_agg
del ocv_discharge_agg
gc.collect()

# ========================
# 4. Merge Aggregated Data and Calculate SOH
# ========================
print("Calculating SOH...")
grouped = pd.merge(grouped, ocv_hysteresis_df, on=['cell_number', 'cycle_number'], how='left')

# Free up memory
del ocv_hysteresis_df
gc.collect()

# Compute State of Health (SOH)
grouped['SOH'] = (
    grouped['max_discharge_capacity'] /
    grouped.groupby('cell_number')['max_discharge_capacity'].transform('first')
) * 100

# Drop NaN values in SOH
grouped.dropna(subset=['SOH'], inplace=True)

# ========================
# 5. Define Features & Labels
# ========================
print("Preparing for model training...")
features = ['avg_voltage', 'cycle_number', 'hysteresis_voltage', 'avg_temperature']
X = grouped[features].values
y = grouped['SOH'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========================
# 6. Time-Based Train/Test Split
# ========================
grouped_sorted = grouped.sort_values(by=['cell_number', 'cycle_number'])
X_seq = grouped_sorted[features].values
y_seq = grouped_sorted['SOH'].values
X_seq_scaled = scaler.transform(X_seq)

split_index = int(0.8 * len(X_seq_scaled))
X_train, X_test = X_seq_scaled[:split_index], X_seq_scaled[split_index:]
y_train, y_test = y_seq[:split_index], y_seq[split_index:]

# ========================
# 7. Train Models & Evaluate Performance
# ========================
print("Training models...")
# Initial models
linear_reg = linear_model.LinearRegression()
ridge_reg = linear_model.Ridge(alpha=1)
lasso_reg = linear_model.Lasso(alpha=1, max_iter=10000)

# Train and evaluate initial models
linear_reg.fit(X_train, y_train)
ridge_reg.fit(X_train, y_train)
lasso_reg.fit(X_train, y_train)

print("\n--- Model Performance on Time-Based Train/Test Split (80/20) ---")
print("Linear Regression:")
print(f"  R2 on Training Set = {linear_reg.score(X_train, y_train):.4f}")
print(f"  R2 on Test Set = {linear_reg.score(X_test, y_test):.4f}")

print("Ridge Regression (alpha=1):")
print(f"  R2 on Training Set = {ridge_reg.score(X_train, y_train):.4f}")
print(f"  R2 on Test Set = {ridge_reg.score(X_test, y_test):.4f}")

print("Lasso Regression (alpha=1):")
print(f"  R2 on Training Set = {lasso_reg.score(X_train, y_train):.4f}")
print(f"  R2 on Test Set = {lasso_reg.score(X_test, y_test):.4f}")

# Test different Lasso alphas
print("--- Lasso Regression: Effect of Alpha ---")
lasso_alphas = [0.1, 0.25, 0.5, 0.75, 1, 5, 10]
lasso_results = {}

for alpha in lasso_alphas:
    lasso = linear_model.Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    score = lasso.score(X_test, y_test)
    lasso_results[alpha] = score
    print(f"Alpha = {alpha}: R2 Test = {score:.4f}")

# Test different Ridge alphas
print("--- Ridge Regression: Effect of Alpha ---")
ridge_alphas = [0.1, 0.25, 0.5, 0.75, 1, 5, 10]
ridge_results = {}

for alpha in ridge_alphas:
    ridge = linear_model.Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    score = ridge.score(X_test, y_test)
    ridge_results[alpha] = score
    print(f"Alpha = {alpha}: R2 Test = {score:.4f}")

# Find best Lasso alpha
best_lasso_alpha = max(lasso_results, key=lasso_results.get)
best_lasso = linear_model.Lasso(alpha=best_lasso_alpha, max_iter=10000)
best_lasso.fit(X_train, y_train)

# Find best Ridge alpha
best_ridge_alpha = max(ridge_results, key=ridge_results.get)
best_ridge = linear_model.Ridge(alpha=best_ridge_alpha)
best_ridge.fit(X_train, y_train)

# Choose overall best model
models = {
    "Linear Regression": linear_reg,
    f"Ridge Regression (alpha={best_ridge_alpha})": best_ridge,
    f"Lasso Regression (alpha={best_lasso_alpha})": best_lasso
}

best_model_name = max(models, key=lambda name: models[name].score(X_test, y_test))
best_model = models[best_model_name]

# Print best model
print(f"\n🔹 Best Performing Model: {best_model_name}")
print(f"R² on Test Set: {best_model.score(X_test, y_test):.4f}")

# Feature Coefficients
coefficients = best_model.coef_
feature_importance = dict(zip(features, coefficients))
sorted_importance = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)

print("\n🔹 Feature Coefficients (Sorted by Impact):")
for feature, coef in sorted_importance:
    print(f"{feature}: {coef:.4f}")

# ========================
# 8. Visualization
# ========================
print("Creating visualizations...")
# Sample data for visualization (to avoid memory issues)
sample_step = max(1, len(grouped) // 1000)  # Show at most 1000 points

plt.figure(figsize=(8, 5))
plt.scatter(grouped['cycle_number'][::sample_step], grouped['hysteresis_voltage'][::sample_step], alpha=0.5, color='blue')
plt.xlabel('Cycle Number')
plt.ylabel('OCV Hysteresis (V)')
plt.title('OCV Hysteresis vs. Cycle Number')
plt.grid(True)

plt.figure(figsize=(8, 5))
plt.scatter(grouped['cycle_number'][::sample_step], grouped['SOH'][::sample_step], alpha=0.5, color='red')
plt.xlabel('Cycle Number')
plt.ylabel('State of Health (SOH %)')
plt.title('State of Health vs. Cycle Number')
plt.grid(True)

plt.figure(figsize=(8, 5))
plt.scatter(grouped['cycle_number'][::sample_step], grouped['avg_voltage'][::sample_step], alpha=0.5, color='green')
plt.xlabel('Cycle Number')
plt.ylabel('Average Voltage (V)')
plt.title('Average Voltage vs. Cycle Number')
plt.grid(True)

# Create feature importance bar chart
plt.figure(figsize=(10, 6))
features_abs = [abs(coef) for _, coef in sorted_importance]
features_names = [feature for feature, _ in sorted_importance]

plt.bar(features_names, features_abs, color='royalblue')
plt.ylabel('Absolute Coefficient Value')
plt.xlabel('Feature')
plt.title('Feature Importance (Absolute Coefficient Values)')
plt.xticks(rotation=45)
plt.tight_layout()

# Add value labels on top of bars
for i, v in enumerate(features_abs):
    plt.text(i, v + 0.1, f"{v:.4f}", ha='center')

plt.show()
print("Analysis complete!")