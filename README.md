# CHEMENG-277-Final-Project

## Overview
This project analyzes battery State of Health (SOH) using the Oxford Battery Degradation Dataset. The analysis focuses on discharge trends and incorporates Open Circuit Voltage (OCV) hysteresis as a key feature to improve SOH predictions. Multivariable regression is applied and optimized with lasso and ridge regression to select the most important features impacting battery degradation. 
## Repository Structure
##

##
/CHEMENG-277-Final-Project
│── Copy of Oxford_battery_data_discharge.csv
│── Copy of Oxford_battery_data_charge.csv
│── Copy of Oxford_battery_data_OCVdc.csv
│── Copy of Oxford_battery_data_OCVch.csv
│── main3.py
│── README.md
│── requirements.txt
│── .gitignore
│── .gitattributes
│── ARCHIVE/
│── .idea/
│── avg_voltage_vs_cycle.png
│── feature_importance.png
│── feature_importance_table.png
│── model_performance_table.png
│── ocv_hysteresis_vs_cycle.png
│── soh_vs_cycle.png

##
- **CSV Files**: The dataset includes discharge, charge, and OCV data, stored in the root directory.
- **`main3.py`**: The primary analysis script that preprocesses data, trains models, and generates visualizations.
- **Visualizations**: Figures generated from the analysis, including feature importance, SOH degradation trends, and hyperparameter tuning results.
- **`.gitattributes`**: Enables **Git Large File Storage (LFS)** for handling large CSV files.
- **`requirements.txt`**: List of dependencies for easy setup.

## Installation
Ensure you have **Python 3.13** installed, then run:

pip install -r requirements.txt



## Running the Analysis
To execute the analysis:
python main3.py

This script:
Loads and processes the discharge, charge, and OCV datasets.
Computes hysteresis voltage and SOH.
Implements k-fold cross-validation and LOOCV for robust model evaluation.
Trains and evaluates Linear, Ridge, and Lasso Regression models.
Optimizes hyperparameters for Ridge and Lasso Regression.
Generates key plots for cycle degradation trends, feature importance, and model performance.
##

## Dependencies
The project requires:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `scipy.stats `
- `seaborn `
- `gc`

## Results & Key Findings
Battery SOH Estimation Using Data-Driven Modeling

1. Feature Selection & Modeling Approach
- A ridge regression model (α = 0.43) was selected after extensive hyperparameter tuning and cross-validation.
- Key features included OCV hysteresis, cycle number, temperature (Tavg), and current metrics (Iavg, IVar, Ikurt).
- OCV hysteresis had the strongest predictive power, confirming its importance in SOH estimation.

2. Cross-Validation & Model Performance
- Group k-fold (k = 4) and LOOCV ensured generalizability while avoiding data leakage.
- A nested k-fold CV (outer k = 4, inner k = 3) optimized regularization parameters.
- The final ridge regression model achieved high R^2 and low RMSE on both training and test sets.

3. Residual Analysis & Model Limitations
- Residual plots revealed non-random trends, indicating that battery degradation is not purely linear.
- The model slightly underestimated SOH for test set batteries, likely due to slower degradation rates compared to the training set.
- Future improvements may include higher-order polynomial models or additional predictors to refine SOH estimation.
- 
## Contributing
For improvements or extensions, feel free to fork and submit a pull request.

