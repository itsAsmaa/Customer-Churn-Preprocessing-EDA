# Customer Churn Analysis: Data Preprocessing & EDA Pipeline

A comprehensive machine learning data preprocessing and exploratory data analysis pipeline for customer churn prediction. This project demonstrates industry-standard data cleaning techniques, statistical analysis, and visualization methods.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset Description](#dataset-description)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a complete data preprocessing pipeline for a customer churn prediction dataset. It addresses real-world data quality issues including missing values, outliers, and feature scaling while maintaining data integrity and documenting all decisions with clear justifications.

**Course**: Machine Learning and Data Science (ENCS5341)  
**Institution**: Electrical and Computer Engineering Department  
**Assignment**: Data Preprocessing & Exploratory Data Analysis

## Features

### Data Quality Management
- Missing Value Handling: Intelligent imputation strategies based on data distribution (median/mode)
- Outlier Detection: Dual-method approach using IQR and Z-score techniques
- Outlier Treatment: Winsorizing for extreme values while preserving data points
- Feature Scaling: Both Min-Max normalization and standardization implementations

### Analysis & Visualization
- Univariate Analysis: Distribution analysis for all features
- Bivariate Analysis: Relationship exploration between features and target variable
- Correlation Analysis: Heatmaps and correlation matrices
- Advanced Visualizations: Pair plots, box plots, scatter plots, and histograms

### Documentation
- Detailed justifications for every preprocessing decision
- Before/after comparisons with statistical metrics
- Comprehensive phase reports with key findings

## Dataset Description

The dataset simulates a customer database with the following features:

| Feature | Type | Description |
|---------|------|-------------|
| CustomerID | Numeric | Unique identifier for each customer |
| Age | Numeric | Customer age (years) |
| Gender | Categorical | 0 = Male, 1 = Female |
| Income | Numeric | Annual income (USD) |
| Tenure | Numeric | Years with company |
| ProductType | Categorical | 0 = Basic, 1 = Premium |
| SupportCalls | Numeric | Support calls in last year |
| ChurnStatus | Binary | 0 = Stayed, 1 = Churned (Target) |

**Original Dataset**: 3,500 customers  
**Final Clean Dataset**: 3,165 customers  
**Data Completeness**: 100% (after preprocessing)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer-churn-analysis.git
cd customer-churn-analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Required Libraries
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

## Usage

### Running the Complete Pipeline

1. **Execute the main script**:
```bash
python assignment1_encs5341.py
```

2. **Or use Jupyter Notebook** (recommended for interactive exploration):
```bash
jupyter notebook Assignment1_ENCS5341.ipynb
```

### Phase-by-Phase Execution

The pipeline is organized into 6 phases:

```python
# Phase 1: Data Loading and Inspection
# Phase 2: Missing Values Handling
# Phase 3: Outlier Detection and Treatment
# Phase 4: Feature Scaling
# Phase 5: Exploratory Data Analysis
# Phase 6: Advanced Visualizations
```

### Output Files

The pipeline generates the following files:
- `clean_customers_v1.csv` - After missing value handling
- `clean_customers_v2.csv` - After outlier treatment
- `MinMax.csv` - Min-Max scaled features
- `standerdization.csv` - Standardized features

## Project Structure

```
customer-churn-analysis/
│
├── assignment1_encs5341.py          # Main Python script
├── Assignment1_ENCS5341.ipynb       # Jupyter notebook version
├── assignment_1.pdf                 # Assignment specifications
│
├── data/
│   ├── customer_data.csv            # Original dataset
│   ├── clean_customers_v1.csv       # After missing value handling
│   ├── clean_customers_v2.csv       # After outlier handling
│   ├── MinMax.csv                   # Min-Max scaled data
│   └── standerdization.csv          # Standardized data
│
├── visualizations/                  # Generated plots and charts
│   ├── missing_values_analysis.png
│   ├── outlier_boxplots.png
│   ├── correlation_heatmap.png
│   └── feature_distributions.png
│
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
└── LICENSE                          # MIT License
```

## Methodology

### Phase 1: Data Loading & Initial Inspection
- Load dataset using pandas
- Display first rows with `.head()`
- Generate summary statistics with `.describe()`
- Check data types and initial quality with `.info()`

### Phase 2: Missing Values Handling

**Strategy Decision Framework**:
- **< 5% missing**: Row deletion (minimal data loss)
- **5-30% missing**: Intelligent imputation
  - Numerical: Median (robust to outliers)
  - Categorical: Mode (most frequent value)
- **> 30% missing**: Consider dropping column

**Results**:
- Original missing values: 693 (2.48%)
- Final missing values: 0 (100% resolved)
- Rows retained: 3,165 (90.43%)

### Phase 3: Outlier Detection & Handling

**Detection Methods**:
1. **IQR Method**: Q1 - 1.5×IQR to Q3 + 1.5×IQR
2. **Z-Score Method**: |z-score| > 3

**Treatment Strategy**:
- **Winsorizing**: Cap extreme values at specified percentiles
  - Age: 1st-99th percentile
  - Income: 1st-99th percentile
  - SupportCalls: Keep (business-relevant)
- **Keep**: Retain all values for business-critical features (Tenure)

### Phase 4: Feature Scaling

**Two Approaches Implemented**:
1. **Min-Max Scaling**: Transforms features to [0, 1] range
2. **Standardization**: Z-score normalization (μ=0, σ=1)

### Phase 5 & 6: EDA & Visualizations

**Analyses Performed**:
- Univariate: Distribution histograms and count plots
- Bivariate: Feature relationships with churn status
- Multivariate: Correlation matrices and pair plots

## Results

### Key Findings

#### 1. Data Quality Improvement
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Completeness | 97.52% | 100% | +2.48% |
| Missing Values | 693 | 0 | -100% |
| Outlier Impact | High | Controlled | Winsorized |

#### 2. Predictive Feature Insights

**Strong Churn Indicators**:
- Tenure (r = -0.30): Negative correlation - newer customers more likely to churn
- ProductType: Premium users show slightly higher churn rate
- Age, Gender, Income: Weak correlations with churn

**Correlations with ChurnStatus**:
```
Tenure:        -0.30  (Strong negative)
Income:        -0.05  (Weak negative)
SupportCalls:   0.003 (Very weak)
Age:           -0.002 (Negligible)
Gender:        -0.002 (Negligible)
```

#### 3. Data Distribution Characteristics
- **Class Imbalance**: 95.5% retained vs 4.5% churned customers
- **Product Distribution**: 70.1% Basic, 29.9% Premium
- **Gender Balance**: 50.4% Male, 49.6% Female

### Visualizations

Key visualizations generated include:
- Missing value patterns and percentages
- Box plots showing outlier treatment effectiveness
- Correlation heatmaps
- Feature distributions (before/after preprocessing)
- Churn rate analysis by categorical features
- Pair plots showing feature relationships


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Course: Machine Learning and Data Science (ENCS5341)
- Electrical and Computer Engineering Department
- Assignment guidance and specifications provided by course instructors

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an educational project demonstrating data preprocessing and EDA techniques for machine learning applications. The dataset is synthetic and created for learning purposes.

## Related Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Data Preprocessing Best Practices](https://scikit-learn.org/stable/modules/preprocessing.html)
