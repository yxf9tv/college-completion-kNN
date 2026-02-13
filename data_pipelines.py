import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Data Loading Functions
def load_csv_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded data from {filepath}")
    print(f"Shape: {df.shape}")
    return df



# Type Conversion Functions
def convert_to_categorical(df, columns):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
            print(f"Converted '{col}' to categorical type")
    return df

# Factor Level Manipulation Functions
def collapse_categorical_levels(df, column, top_n, other_label='Other'):
    df = df.copy()
    top_categories = df[column].value_counts().head(top_n).index.tolist()
    df[column] = df[column].apply(
        lambda x: x if x in top_categories else other_label
    ).astype('category')

    print(f"Collapsed '{column}' to top {top_n} categories + '{other_label}'")
    print(f"New value counts:\n{df[column].value_counts()}\n")
    return df


# Missing Value Handling Functions
def drop_rows_with_missing_target(df, target_col):
    df = df.copy()
    initial_count = len(df)
    df = df.dropna(subset=[target_col])
    dropped_count = initial_count - len(df)

    print(f"Dropped {dropped_count} rows with missing '{target_col}'")
    print(f"Remaining rows: {len(df)}")
    return df


def filter_by_categorical_value(df, column, value):
    df = df.copy()
    initial_count = len(df)
    df = df[df[column] == value].copy()

    print(f"Filtered to keep only '{value}' in '{column}'")
    print(f"Rows before: {initial_count}, after: {len(df)}")
    return df


def fill_numeric_missing_with_median(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled {missing_count} missing values in '{col}' with median: {median_val}")

    return df


# Column Dropping Functions
def drop_columns(df, columns):
    df = df.copy()
    columns_to_drop = [col for col in columns if col in df.columns]
    df = df.drop(columns=columns_to_drop)

    print(f"Dropped {len(columns_to_drop)} columns: {columns_to_drop}")
    print(f"Remaining columns: {len(df.columns)}")
    return df


def drop_high_missing_columns(df, threshold=0.5):
    df = df.copy()
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing values")

    return df


# Feature Scaling Functions
def normalize_numeric_features(df, exclude_cols=None):
    df = df.copy()
    exclude_cols = exclude_cols or []

    # Get numeric columns excluding specified ones
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]

    if cols_to_scale:
        scaler = MinMaxScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        print(f"Normalized {len(cols_to_scale)} numeric columns")
        print(f"Excluded from normalization: {exclude_cols}")

    return df



# Encoding Functions
def one_hot_encode_categoricals(df, drop_first=False):
    df = df.copy()
    categorical_cols = df.select_dtypes(include='category').columns.tolist()

    if categorical_cols:
        original_shape = df.shape
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)
        print(f"One-hot encoded {len(categorical_cols)} categorical columns")
        print(f"Shape change: {original_shape} -> {df.shape}")

    return df



# Baseline and Statistics Functions
def calculate_baseline_stats(df, target_col):
    stats = {
        'mean': df[target_col].mean(),
        'median': df[target_col].median(),
        'std': df[target_col].std(),
        'min': df[target_col].min(),
        'max': df[target_col].max()
    }

    print(f"\nBaseline Statistics for '{target_col}':")
    print(f"  Mean: {stats['mean']:,.2f}")
    print(f"  Median: {stats['median']:,.2f}")
    print(f"  Std Dev: {stats['std']:,.2f}")
    print(f"  Range: [{stats['min']:,.2f}, {stats['max']:,.2f}]")
    print(f"\nBaseline RMSE (always predicting mean): {stats['std']:,.2f}\n")

    return stats


# Data Splitting Functions
def split_train_tune_test(df, target_col=None, train_size=0.7, tune_size=0.15, random_state=42):
    # Calculate actual sizes
    train_rows = int(train_size * len(df))

    # Create stratification bins from continuous target using quantiles
    stratify_col = None
    if target_col:
        stratify_col = pd.qcut(df[target_col], q=5, duplicates='drop')

    # First split: Train vs (Tune + Test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_rows,
        stratify=stratify_col,
        random_state=random_state
    )

    # Second split: use tune_size to compute the proper ratio of remaining data
    tune_ratio = tune_size / (1 - train_size)
    stratify_temp = None
    if target_col:
        stratify_temp = pd.qcut(temp_df[target_col], q=5, duplicates='drop')

    tune_df, test_df = train_test_split(
        temp_df,
        train_size=tune_ratio,
        stratify=stratify_temp,
        random_state=random_state
    )

    print(f"\nData Split:")
    print(f"  Train: {train_df.shape} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Tune:  {tune_df.shape} ({len(tune_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {test_df.shape} ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, tune_df, test_df



# Pipeline Orchestration Functions
def prepare_college_aid_pipeline(filepath="cc_institution_details.csv"):
    print("="*70)
    print("COLLEGE AID PREDICTION - DATA PREPARATION PIPELINE")
    print("="*70)

    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    df = load_csv_data(filepath)

    # Step 2: Convert categorical columns
    print("\n[Step 2] Converting categorical columns...")
    categorical_cols = ['state', 'level', 'control', 'basic', 'hbcu', 'flagship']
    df = convert_to_categorical(df, categorical_cols)

    # Step 3: Collapse factor levels
    print("\n[Step 3] Collapsing categorical levels...")
    df = collapse_categorical_levels(df, 'basic', top_n=5, other_label='Other')
    df = collapse_categorical_levels(df, 'state', top_n=10, other_label='Other')

    # Step 4: Handle missing target values
    print("\n[Step 4] Handling missing target values...")
    df = drop_rows_with_missing_target(df, 'aid_value')

    # Step 5: Drop unnecessary columns
    print("\n[Step 5] Dropping unnecessary columns...")
    cols_to_drop = [
        'index', 'unitid', 'chronname', 'city', 'site', 'long_x', 'lat_y',
        'similar', 'nicknames', 'counted_pct', 'cohort_size', 'aid_percentile'
    ]
    # Also drop VSA columns
    vsa_cols = [col for col in df.columns if col.startswith('vsa_')]
    cols_to_drop.extend(vsa_cols)
    df = drop_columns(df, cols_to_drop)

    # Step 6: Fill remaining missing values
    print("\n[Step 6] Filling remaining missing values...")
    df = fill_numeric_missing_with_median(df)

    # Step 7: Normalize numeric features (except target)
    print("\n[Step 7] Normalizing numeric features...")
    df = normalize_numeric_features(df, exclude_cols=['aid_value'])

    # Step 8: One-hot encode categorical features
    print("\n[Step 8] One-hot encoding categorical features...")
    df = one_hot_encode_categoricals(df, drop_first=False)

    # Step 9: Calculate baseline statistics
    print("\n[Step 9] Calculating baseline statistics...")
    baseline_stats = calculate_baseline_stats(df, 'aid_value')

    # Step 10: Split into Train/Tune/Test
    print("\n[Step 10] Splitting data...")
    train_df, tune_df, test_df = split_train_tune_test(df, target_col='aid_value')

    print("\n" + "="*70)
    print("PIPELINE COMPLETE - Data ready for modeling!")
    print("="*70)

    return train_df, tune_df, test_df, baseline_stats


def prepare_placement_pipeline(filepath="Placement_Data_Full_Class.csv"):
    print("="*70)
    print("JOB PLACEMENT SALARY PREDICTION - DATA PREPARATION PIPELINE")
    print("="*70)

    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    df = load_csv_data(filepath)

    # Step 2: Convert categorical columns
    print("\n[Step 2] Converting categorical columns...")
    categorical_cols = [
        'gender', 'ssc_b', 'hsc_b', 'hsc_s',
        'degree_t', 'workex', 'specialisation', 'status'
    ]
    df = convert_to_categorical(df, categorical_cols)

    # Step 3: Filter to only placed students (those with salary values)
    print("\n[Step 3] Filtering to placed students only...")
    df = filter_by_categorical_value(df, 'status', 'Placed')

    # Step 4: Drop unnecessary columns
    print("\n[Step 4] Dropping unnecessary columns...")
    cols_to_drop = ['sl_no', 'status']  # status is now constant
    df = drop_columns(df, cols_to_drop)

    # Step 5: Normalize numeric features (except target)
    print("\n[Step 5] Normalizing numeric features...")
    df = normalize_numeric_features(df, exclude_cols=['salary'])

    # Step 6: One-hot encode categorical features
    print("\n[Step 6] One-hot encoding categorical features...")
    df = one_hot_encode_categoricals(df, drop_first=False)

    # Step 7: Calculate baseline statistics
    print("\n[Step 7] Calculating baseline statistics...")
    baseline_stats = calculate_baseline_stats(df, 'salary')

    # Step 8: Split into Train/Tune/Test
    print("\n[Step 8] Splitting data...")
    train_df, tune_df, test_df = split_train_tune_test(df, target_col='salary')

    print("\n" + "="*70)
    print("PIPELINE COMPLETE - Data ready for modeling!")
    print("="*70)

    return train_df, tune_df, test_df, baseline_stats



# Main Execution (for testing)
if __name__ == "__main__":
    print("\n\nTesting Pipeline 1: College Aid Prediction")
    print("="*70)
    train1, tune1, test1, stats1 = prepare_college_aid_pipeline()

    print("\n\n")

    print("Testing Pipeline 2: Job Placement Salary Prediction")
    print("="*70)
    train2, tune2, test2, stats2 = prepare_placement_pipeline()

    print("\n\nAll pipelines executed successfully!")
