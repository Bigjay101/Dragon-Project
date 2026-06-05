# Dragon-Project

from pathlib import Path

readme = """# Dragon Flame Heat Output Regression Project

## Overview

This project investigates the prediction of dragon **Flame Heat Output (FHO)** using observable physical and biological characteristics. The predictors include:

- `AGE` — age in years
- `MASS` — mass in metric tons
- `WSP` — wingspan in metres
- `HID` — hide thickness in centimetres
- `SPD` — sustained flight speed in kilometres per hour
- `SPC` — dragon species

The project includes data cleaning, exploratory data analysis, single-variable regression, multiple-variable regression, model comparison, coefficient interpretation, model evaluation, and feature engineering.

## Project Objectives

The main objectives are to:

1. Determine which individual variables are most informative for predicting FHO.
2. Compare single-variable and multiple-variable regression models.
3. Assess whether adding predictors improves generalisation.
4. Interpret the signs and magnitudes of regression coefficients.
5. Evaluate model reliability using test metrics and residual analysis.
6. Improve predictive performance through feature engineering.

## Repository Structure

```text
.
├── final_dragon_regression_project.ipynb
├── dragon_data.csv
├── README.md
├── excel_section1/
│   └── Dragon_Question1.xlsx
|   ├── FHOvsAge.png
│   ├── FHOvsMass.png
│   ├── FHOvsWSP.png
├── excel_section1_spc/
│   ├── spc_test.xlsx
|   ├── spc_train.xlsx
|   └── FHODistributionBySpecies.png
│   └── split.py
│   
└── supporting_files/
    └── Earlier EDA and multiple-regression working files