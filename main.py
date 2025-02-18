#test test
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
#Jessica's test text

# =============================================================================
# 1. Data Input and Preprocessing
# =============================================================================
# Input: CSV file(s) from online datasets (e.g., NASA, Oxford, Battery Archive)
# The CSV is expected to include columns: time, voltage, current, cycle_count, temperature.
# For this example, assume the file 'battery_data.csv' is in the working directory.

DATA_FILE = 'battery_data.csv'  # <-- Update this path with your actual CSV file


def load_and_preprocess_data(file_path):
    """
    Loads battery data from a CSV file and preprocesses it.

    Expected CSV columns:
      - time: measurement timestamp (optional, not used directly in HMM)
      - voltage: measured battery voltage
      - current: measured battery current
      - cycle_count: number of charge/discharge cycles
      - temperature: battery temperature

    Returns:
      observed_data: A NumPy array of shape (n_samples, 4) with the observed features.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Data cleaning: Remove any rows with missing values
    df.dropna(inplace=True)

    # (Optional) Normalize or standardize the observed features if necessary
    # Uncomment the following lines to apply z-score normalization:
    # features = ['voltage', 'current', 'cycle_count', 'temperature']
    # df[features] = (df[features] - df[features].mean()) / df[features].std()

    # Extract the observed variables: voltage, current, cycle_count, and temperature
    observed_data = df[['voltage_load', 'current_load', 'time', 'temperature_battery']].values
    return observed_data


# Load the observed battery data
observed_data = load_and_preprocess_data(DATA_FILE)
print("Loaded observed data shape:", observed_data.shape)
print("First 5 observations:\n", observed_data[:5])

# =============================================================================
# 2. HMM Setup and Training
# =============================================================================
# In our framework, the HMM's observed inputs are the multivariate battery measurements:
#    - Voltage
#    - Current
#    - Cycle Count
#    - Temperature
#
# The hidden state we wish to predict is the battery's "State of Health" (SOH).
# For this example, we assume three possible SOH levels (e.g., Good, Moderate, Poor).
#
# The HMM uses the Baum-Welch algorithm internally (via model.fit) to estimate the transition
# and emission probabilities from the observed data.

# Define the number of hidden states (battery SOH levels)
n_hidden_states = 3  # This can be adjusted based on domain knowledge

# Initialize a Gaussian HMM with diagonal covariance matrices
hmm_model = GaussianHMM(n_components=n_hidden_states, covariance_type="diag",
                        n_iter=100, random_state=42)

# Fit the HMM to the observed data
hmm_model.fit(observed_data)

# =============================================================================
# 3. Prediction and Output
# =============================================================================
# Output: The HMM will predict the hidden states (battery SOH) for each time step.
# It also provides state probability distributions which indicate the likelihood
# that the battery is in a specific SOH level given the observed measurements.

# Predict the sequence of hidden states (i.e., battery state of health)
predicted_soh = hmm_model.predict(observed_data)

# Obtain the state probability distributions for each observation
state_probabilities = hmm_model.predict_proba(observed_data)

# Display the predictions for verification
print("\nPredicted Battery State of Health (first 10 observations):")
print(predicted_soh[:10])
print("\nState Probability Distributions (first 5 observations):")
print(state_probabilities[:5])

# =============================================================================
# 4. Visualization (Optional)
# =============================================================================
# Plot the predicted battery state of health over time for a visual inspection.
plt.figure(figsize=(12, 6))
plt.plot(predicted_soh, marker='o', linestyle='-', label='Predicted SOH')
plt.title('Battery State of Health Prediction Using HMM')
plt.xlabel('Time Index')
plt.ylabel('Predicted Hidden State (Battery SOH)')
plt.legend()
plt.show()
