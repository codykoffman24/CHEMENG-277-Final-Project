# CHEMENG-277-Final-Project

## Overview
This project analyzes battery **State of Health (SOH)** using **discharge dataset trends** and various regression models. It incorporates **hysteresis voltage** as a key feature and compares multiple modeling techniques, including **Linear, Ridge, and Lasso Regression**, to evaluate battery degradation over time.

## Repository Structure
```
/CHEMENG-277-Final-Project
│── Copy of Oxford_battery_data_discharge.csv
│── Copy of Oxford_battery_data_charge.csv
│── main.py
│── README.md
│── requirements.txt
│── .gitignore
│── .gitattributes
│── ARCHIVE/
│── .idea/
```

- **CSV files**: Raw data files stored in the **root directory**.
- **`main.py`**: Primary analysis script. Run this file to preprocess data, train models, and generate visualizations.
- **`requirements.txt`**: List of dependencies for easy setup.

## Installation
Ensure you have **Python 3.x** installed, then run:
```bash
pip install -r requirements.txt
```

## Running the Analysis
To execute the analysis:
```bash
python main.py
```
This script:
- Loads and processes the **discharge and charge datasets**.
- Computes **hysteresis voltage** and **SOH**.
- Trains and evaluates **Linear, Ridge, and Lasso Regression models**.
- Generates key plots for **cycle degradation trends**.

## Dependencies
The project requires:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

## Results & Key Findings
- **Hysteresis voltage increases with cycle count**, indicating rising internal resistance.
- **SOH degrades over cycles**, aligning with expected battery aging behavior.
- **Lasso regression performs best at α=0.25**, with an **R² test score of 0.9884**.
- **Discharge dataset trends are prioritized**, as they offer better insights into battery health.

## Contributing
For improvements or extensions, feel free to fork and submit a pull request.

