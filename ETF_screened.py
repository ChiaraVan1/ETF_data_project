import pandas as pd
import numpy as np

# Load the data
df_funds = pd.read_csv('df_funds_with_all_metrics.csv')

# --- Step 1: Data Cleaning and Preprocessing ---

# Convert relevant columns to numeric, coercing errors to NaN
columns_to_convert = ['excess_return_mean', 'tracking_error', 'turnover_rate', 'turnover_6m_vs_3y']
for col in columns_to_convert:
    df_funds[col] = pd.to_numeric(df_funds[col], errors='coerce')

# Drop rows with missing values in the key metrics
df_funds.dropna(subset=columns_to_convert, inplace=True)

# --- Step 2: Define the thresholds based on the user's updated strategy ---

# Threshold for `turnover_rate`: 10% percentile of ALL funds (already handled)
turnover_rate_threshold = df_funds['turnover_rate'].quantile(0.1)

# Threshold for `turnover_6m_vs_3y`: greater than 1
turnover_6m_vs_3y_threshold = 1

# Thresholds for `excess_return_mean` and `tracking_error`: industry means
# First, group by industry and calculate the means
industry_metrics_mean = df_funds.groupby('industry').agg({
    'excess_return_mean': 'mean',
    'tracking_error': 'mean'
}).to_dict('index')

# --- Step 3: Apply the combined screening strategy ---

# Create a list to store the final screened ETFs
screened_etfs_list = []

# Iterate through each ETF to apply the combined conditions
for index, row in df_funds.iterrows():
    # Get the industry-specific thresholds
    industry = row['industry']
    if industry in industry_metrics_mean:
        excess_return_threshold = industry_metrics_mean[industry]['excess_return_mean']
        tracking_error_threshold = industry_metrics_mean[industry]['tracking_error']

        # Apply the four conditions
        cond1 = row['turnover_6m_vs_3y'] > turnover_6m_vs_3y_threshold
        cond2 = row['tracking_error'] < tracking_error_threshold
        cond3 = row['excess_return_mean'] > excess_return_threshold
        cond4 = row['turnover_rate'] > turnover_rate_threshold

        # If all conditions are met, add the ETF to the list
        if cond1 and cond2 and cond3 and cond4:
            screened_etfs_list.append(row)

# Create the final DataFrame from the list of results
df_final_screened = pd.DataFrame(screened_etfs_list)

# --- Step 4: Present the results ---

if not df_final_screened.empty:
    # Select key columns for the final display
    df_final_screened = df_final_screened[['ts_code', 'name', 'industry', 'invest_type', 'turnover_rate', 'turnover_6m_vs_3y', 'excess_return_mean', 'tracking_error']]
    output_filename = 'ETF_final_screened.csv'
    df_final_screened.to_csv(output_filename, index=False, encoding='utf-8-sig')
else:
    print("根据你的4条策略，没有筛选出符合条件的ETF。")
