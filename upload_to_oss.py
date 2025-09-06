import os
import oss2
import pandas as pd

# Get environment variables from GitHub Secrets
access_key_id = os.environ.get('ALIYUN_ACCESS_KEY_ID')
access_key_secret = os.environ.get('ALIYUN_ACCESS_KEY_SECRET')
endpoint = os.environ.get('OSS_ENDPOINT')
bucket_name = os.environ.get('OSS_BUCKET_NAME')

# Define the columns to select for the final data display
# We keep ts_code for merging and name for identification
screener_columns_to_keep = [
    'ts_code', 'name', '换手率(%)', '换手率6个月比3年', '超额收益均值(%)',
    '追踪误差(%)', '超额收益趋势斜率(万分之)', '行业内成交额占比(%)',
    '行业内成交额占比(百分位)'
]

metrics_columns_to_keep = [
    'ts_code', 'latest_discount_rate', 'annualized_volatility', 'max_drawdown'
]

# Initialize OSS authentication
auth = oss2.Auth(access_key_id, access_key_secret)
bucket = oss2.Bucket(auth, endpoint, bucket_name)

print("Starting data processing and upload...")

try:
    # Read the two CSV files
    df_screener = pd.read_csv('etf_screener_results_normal_mode.csv')
    df_metrics = pd.read_csv('etf_metrics_daily_report.csv')

    # Select the desired columns
    df_screener_selected = df_screener[screener_columns_to_keep]
    df_metrics_selected = df_metrics[metrics_columns_to_keep]

    # Merge the two dataframes on the common key 'ts_code'
    df_final = pd.merge(df_screener_selected, df_metrics_selected, on='ts_code', how='left')

    # Define the final output file name
    final_output_file = 'final_etf_data.csv'

    # Write the final merged DataFrame to a new CSV file
    df_final.to_csv(final_output_file, index=False, encoding='utf-8')

    # Upload the final file to OSS
    bucket.put_object_from_file(final_output_file, final_output_file)
    print(f"Successfully uploaded {final_output_file}.")

except Exception as e:
    print(f"Error during upload: {e}")

print("Upload complete!")
