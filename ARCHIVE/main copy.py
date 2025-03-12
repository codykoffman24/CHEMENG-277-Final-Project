import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

# ========================
# 1. Load & Preprocess Data (Discharge and Charge)
# ========================
discharge_file = "Copy of Oxford_battery_data_discharge.csv"
charge_file = "Copy of Oxford_battery_data_charge.csv"

# Load datasets
discharge_data = pd.read_csv(discharge_file, low_memory=False)
charge_data = pd.read_csv(charge_file, low_memory=False)

# Sort datasets by cycle order, preps for processing
discharge_data = discharge_data.sort_values(by=['cell_number', 'cycle_number', 'time'])
charge_data = charge_data.sort_values(by=['cell_number', 'cycle_number', 'time'])

# Print total rows in each dataset, provides insight into the size of datasets
print(f"Total discharge rows: {len(discharge_data)}")
print(f"Total charge rows: {len(charge_data)}")

# Ensure current values are negative for discharge, in case data has incorrect sign
if discharge_data['current'].max() > 0:
    discharge_data['current'] = -discharge_data['current']

# Assume 1 second has passed between each row, this was explicitly stated in the dataset readme and the report of source material
discharge_data['time_diff'] = 1
charge_data['time_diff'] = 1

# Compute discharge capacity incrementally, sets the units to mAh for industry comparisons
discharge_data['incremental_discharge'] = (discharge_data['current'] * discharge_data['time_diff']).abs() / 3600
discharge_data['discharge_capacity'] = discharge_data.groupby(['cell_number', 'cycle_number'])['incremental_discharge'].cumsum() * 1000

# ========================
# 2. Combine Charge & Discharge for Continuous Timeline
# ========================
discharge_data['type'] = 'discharge'
charge_data['type'] = 'charge'
# merges both datasets while maintaining cycle order, so the 1 second between rows assumption is maintained even as the script switches between csv files
full_data = pd.concat([discharge_data, charge_data]).sort_values(by=['cell_number', 'cycle_number', 'time'])

# Re-index time to reflect a continuous sequence within each cycle
full_data['time'] = full_data.groupby(['cell_number', 'cycle_number']).cumcount()

#Verifies that each battery has a corresponding charge-discharge cycle, test first 3 battery cells.
print("\n🔍 Sample Time Alignment Check:")
for cell in discharge_data['cell_number'].unique()[:3]:
    discharge_cycles = discharge_data[discharge_data['cell_number'] == cell]['cycle_number'].unique()
    charge_cycles = charge_data[charge_data['cell_number'] == cell]['cycle_number'].unique()
    print(f"Cell {cell}: {len(discharge_cycles)} discharge cycles, {len(charge_cycles)} charge cycles")

#remove unrealistic temp values (inaccurate outliers), prints a summary to verify no crazy temps
full_data_filtered = full_data[full_data['temperature'] >= 35].copy()
print("\n🔍 Temperature Data Summary Before Aggregation:")
print(full_data_filtered['temperature'].describe().apply(lambda x: f"{x:.2f} °C"))


# ========================
# 4. Compute Voltage Hysteresis, difference between change and discharge voltages per cycle, used for additional input feature
# ========================
hysteresis_df = full_data_filtered.pivot_table(index=['cell_number', 'cycle_number'],
                                               columns='type',
                                               values='voltage',
                                               aggfunc='mean')

hysteresis_df['hysteresis_voltage'] = hysteresis_df['charge'] - hysteresis_df['discharge']
hysteresis_df.reset_index(inplace=True)

# ========================
# 5. Aggregate Features for SOH Calculation
# ========================
#maximum discharge capacity in a cycle represents the total usable energy the battery can provide before it is fully depleted
agg_funcs = {
    'voltage': 'mean',
    'temperature': 'mean',
    'discharge_capacity': 'max'
}

grouped = full_data_filtered.groupby(['cell_number', 'cycle_number']).agg(agg_funcs).reset_index()

grouped.rename(columns={
    'voltage': 'avg_voltage',
    'temperature': 'avg_temperature',
    'discharge_capacity': 'max_discharge_capacity'
}, inplace=True)

grouped = pd.merge(grouped, hysteresis_df[['cell_number', 'cycle_number', 'hysteresis_voltage']],
                   on=['cell_number', 'cycle_number'], how='left')

# ========================
# 6. Compute State of Health (SOH)
# normalizes battery capacity relative to its first cycle
def compute_soh(df):
    cycle_1_capacity = df.loc[df['cycle_number'] == 1, 'max_discharge_capacity']
    if cycle_1_capacity.empty or cycle_1_capacity.values[0] <= 0:
        df['SOH'] = np.nan
    else:
        baseline = cycle_1_capacity.values[0]
        df['SOH'] = (df['max_discharge_capacity'] / baseline) * 100
    return df

grouped = grouped.groupby('cell_number', group_keys=False).apply(compute_soh).dropna(subset=['SOH'])

# ========================
# 7. Define Features & Labels
# ========================
features = ['avg_voltage', 'avg_temperature', 'cycle_number', 'hysteresis_voltage']
X = grouped[features].values
y = grouped['SOH'].values
#scales the features, Scaling helps prevent numerical instability and speeds up convergence in ridge and lasso regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========================
# 8. Time-Based Train/Test Split
# Ensures sequential ordering for time-based splitting of data (train on first 80%, test on sequential 20%, only the discharge dataset values)
grouped_sorted = grouped.sort_values(by=['cell_number', 'cycle_number'])
X_seq = grouped_sorted[features].values
y_seq = grouped_sorted['SOH'].values
X_seq_scaled = scaler.transform(X_seq)

split_index = int(0.8 * len(X_seq_scaled))
X_train, X_test = X_seq_scaled[:split_index], X_seq_scaled[split_index:]
y_train, y_test = y_seq[:split_index], y_seq[split_index:]

# ========================
# 9. Train Models & Evaluate Performance
# same methods from class
models = {
    "Linear Regression": linear_model.LinearRegression(),
    "Ridge Regression (alpha=1)": linear_model.Ridge(alpha=1),
    "Lasso Regression (alpha=1)": linear_model.Lasso(alpha=1, max_iter=10000)
}
# Trains models and prints their performance.
print("\n--- Model Performance on Time-Based Train/Test Split (80/20) ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name}:")
    print(f"  R2 on Training Set = {model.score(X_train, y_train):.4f}")
    print(f"  R2 on Test Set = {model.score(X_test, y_test):.4f}")

# Evaluate different alpha values for Lasso and Ridge
lasso_alphas = [0.1, 0.25, 0.5, 0.75, 1, 5, 10]
ridge_alphas = [0.1, 0.25, 0.5, 0.75, 1, 5, 10]

print("\n--- Lasso Regression: Effect of Alpha ---")
for alpha in lasso_alphas:
    model = linear_model.Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train, y_train)
    print(f"Alpha = {alpha}: R2 Test = {model.score(X_test, y_test):.4f}")

print("\n--- Ridge Regression: Effect of Alpha ---")
for alpha in ridge_alphas:
    model = linear_model.Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    print(f"Alpha = {alpha}: R2 Test = {model.score(X_test, y_test):.4f}")

# ========================
# 10. Visualization
# ========================
#visualizes hysteris over cycle number

plt.figure(figsize=(8, 5))
plt.scatter(grouped['cycle_number'], grouped['hysteresis_voltage'], alpha=0.5, color='blue')
plt.xlabel('Cycle Number')
plt.ylabel('Hysteresis Voltage (V)')
plt.title('Hysteresis Voltage vs. Cycle Number')
plt.grid(True)

#visualizes state of health over cycle number
plt.figure(figsize=(8, 5))
plt.scatter(grouped['cycle_number'], grouped['SOH'], alpha=0.5, color='red')
plt.xlabel('Cycle Number')
plt.ylabel('State of Health (SOH %)')
plt.title('State of Health vs. Cycle Number')
plt.grid(True)

#visualizes average voltage over cycle number

plt.figure(figsize=(8, 5))
plt.scatter(grouped['cycle_number'], grouped['avg_voltage'], alpha=0.5, color='green')
plt.xlabel('Cycle Number')
plt.ylabel('Average Voltage (V)')
plt.title('Average Voltage vs. Cycle Number')
plt.grid(True)

plt.show()

import matplotlib.pyplot as plt

# Identify the best-performing model based on test R² score
best_model_name = max(models, key=lambda name: models[name].score(X_test, y_test))
best_model = models[best_model_name]

# Print the best model name and its test performance
print(f"\n🔹 Best Performing Model: {best_model_name}")
print(f"R² on Test Set: {best_model.score(X_test, y_test):.4f}")

# Extract feature coefficients
coefficients = best_model.coef_
feature_importance = dict(zip(features, coefficients))

# Print the feature coefficients in descending order of impact
sorted_importance = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
print("\n🔹 Feature Coefficients (Sorted by Impact):")
for feature, coef in sorted_importance:
    print(f"{feature}: {coef:.4f}")

# Visualization: Feature Importance Bar Plot
plt.figure(figsize=(8, 5))
plt.barh([f[0] for f in sorted_importance], [f[1] for f in sorted_importance], color='blue')
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.title(f"Feature Importance in {best_model_name}")
plt.gca().invert_yaxis()  # Invert to show most impactful at the top
plt.grid(True)

# Show the plot
plt.show()

