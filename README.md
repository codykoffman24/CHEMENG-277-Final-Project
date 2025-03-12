# CHEMENG-277-Final-Project

## Overview
This project analyzes battery State of Health (SOH) using the Oxford Battery Degradation Dataset. The analysis focuses on discharge trends and incorporates Open Circuit Voltage (OCV) hysteresis as a key feature to improve SOH predictions. Multiple regression models are evaluated, including Linear, Ridge, and Lasso Regression, to determine the most effective method for modeling battery degradation over time.
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

## Results & Key Findings
-Hysteresis voltage increases with cycle count, indicating rising internal resistance.
-SOH degrades over cycles, aligning with expected battery aging behavior.
-Ridge Regression (α=10) was selected as the best model, with an R² test score of 0.8988.
-Hyperparameter tuning confirmed the best α values for Ridge and Lasso models.
-Discharge dataset trends are prioritized, as they offer better insights into battery health.


## Contributing
For improvements or extensions, feel free to fork and submit a pull request.

