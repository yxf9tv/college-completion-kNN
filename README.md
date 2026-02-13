# Data Pipeline Documentation

## Overview

The `data_pipelines.py` module contains reusable data preparation functions organized as a DAG (Directed Acyclic Graph) of operations. Each function represents a discrete step in the data preparation process and can be composed together to create complete pipelines.

## Quick Start

### Running the Complete Pipelines

```python
from data_pipelines import prepare_college_aid_pipeline, prepare_placement_pipeline

# Run College Aid pipeline
train, tune, test, stats = prepare_college_aid_pipeline()

# Run Placement Salary pipeline
train, tune, test, stats = prepare_placement_pipeline()
```

### Using Individual Functions

```python
from data_pipelines import (
    load_csv_data,
    convert_to_categorical,
    normalize_numeric_features,
    one_hot_encode_categoricals,
    split_train_tune_test
)

# Load data
df = load_csv_data("mydata.csv")

# Convert categorical columns
df = convert_to_categorical(df, ['gender', 'category'])

# Normalize numeric features (except target)
df = normalize_numeric_features(df, exclude_cols=['target'])

# One-hot encode
df = one_hot_encode_categoricals(df)

# Split data
train, tune, test = split_train_tune_test(df)
```

## Pipeline Architecture

### Function Categories

#### 1. Data Loading
- `load_csv_data()` - Load data from CSV files

#### 2. Type Conversion
- `convert_to_categorical()` - Convert columns to categorical type

#### 3. Factor Level Manipulation
- `collapse_categorical_levels()` - Reduce number of categories

#### 4. Missing Value Handling
- `drop_rows_with_missing_target()` - Remove rows with missing target
- `filter_by_categorical_value()` - Filter to specific category values
- `fill_numeric_missing_with_median()` - Impute numeric missing values

#### 5. Column Management
- `drop_columns()` - Drop specified columns
- `drop_high_missing_columns()` - Drop columns with high missing rates

#### 6. Feature Scaling
- `normalize_numeric_features()` - MinMax scaling (0-1 range)

#### 7. Encoding
- `one_hot_encode_categoricals()` - Create dummy variables

#### 8. Statistics & Baseline
- `calculate_baseline_stats()` - Compute target variable statistics

#### 9. Data Splitting
- `split_train_tune_test()` - Create Train/Tune/Test partitions

## Pipeline DAG Structure

### College Aid Pipeline Flow
```
Load Data
    ↓
Convert Categorical Types
    ↓
Collapse Factor Levels (basic, state)
    ↓
Drop Missing Target
    ↓
Drop Unnecessary Columns
    ↓
Fill Missing with Median
    ↓
Normalize Numeric Features
    ↓
One-Hot Encode
    ↓
Calculate Baseline Stats
    ↓
Split Train/Tune/Test
```

### Placement Pipeline Flow
```
Load Data
    ↓
Convert Categorical Types
    ↓
Filter to Placed Students
    ↓
Drop Unnecessary Columns
    ↓
Normalize Numeric Features
    ↓
One-Hot Encode
    ↓
Calculate Baseline Stats
    ↓
Split Train/Tune/Test
```

## Expected Outputs

### College Aid Pipeline
- **Target Variable:** `aid_value` (continuous)
- **Train Set:** ~2,657 rows, 52 features
- **Tune Set:** ~570 rows, 52 features
- **Test Set:** ~570 rows, 52 features
- **Baseline RMSE:** ~$6,420

### Placement Pipeline
- **Target Variable:** `salary` (continuous)
- **Train Set:** ~103 rows, 22 features
- **Tune Set:** ~22 rows, 22 features
- **Test Set:** ~23 rows, 22 features
- **Baseline RMSE:** ~$93,457

## Design Principles

1. **Modularity:** Each function does one thing well
2. **Composability:** Functions can be chained together
3. **Immutability:** Functions return new DataFrames (don't modify in place)
4. **Logging:** Each step prints progress information
5. **Type Hints:** All functions have type annotations
6. **Documentation:** Comprehensive docstrings

## Customization

To create a new pipeline for a different dataset:

```python
def prepare_my_custom_pipeline(filepath: str):
    """Custom pipeline for my dataset."""

    # Step 1: Load
    df = load_csv_data(filepath)

    # Step 2: Type conversion
    df = convert_to_categorical(df, ['cat1', 'cat2'])

    # Step 3: Handle missing values
    df = drop_rows_with_missing_target(df, 'my_target')

    # Step 4: Feature engineering (add custom logic here)
    # ...

    # Step 5: Normalize
    df = normalize_numeric_features(df, exclude_cols=['my_target'])

    # Step 6: Encode
    df = one_hot_encode_categoricals(df)

    # Step 7: Split
    train, tune, test = split_train_tune_test(df)

    return train, tune, test
```

## Testing

To test the pipelines:

```bash
python data_pipelines.py
```

This will run both pipelines and display detailed output for each step.

## Requirements

- pandas
- numpy
- scikit-learn

## Notes

- All numeric features are normalized to [0, 1] range using MinMaxScaler
- Target variables are NOT normalized to preserve interpretability
- Random state is set to 42 for reproducibility
- Data split: 70% Train, 15% Tune, 15% Test
