import os
import oss2
import pandas as pd

# Get environment variables from GitHub Secrets
access_key_id = os.environ.get('ALIYUN_ACCESS_KEY_ID')
access_key_secret = os.environ.get('ALIYUN_ACCESS_KEY_SECRET')
endpoint = os.environ.get('OSS_ENDPOINT')
bucket_name = os.environ.get('OSS_BUCKET_NAME')

# Initialize OSS authentication
auth = oss2.Auth(access_key_id, access_key_secret)
bucket = oss2.Bucket(auth, endpoint, bucket_name)

print("Starting data upload...")

try:
    # Define the final output file name
    final_output_file = 'etf_screener_final_report.csv'

    # Check if the file exists before attempting to read and upload
    if os.path.exists(final_output_file):
        # The file is already prepared by ETF_screened.py
        # Read it to ensure it's valid, although not strictly necessary for upload
        df_final = pd.read_csv(final_output_file)
        
        # Upload the final file to OSS
        bucket.put_object_from_file(final_output_file, final_output_file)
        print(f"Successfully uploaded {final_output_file} to OSS.")
    else:
        print(f"Error: The file '{final_output_file}' was not found. Please ensure ETF_screened.py was run successfully.")

except Exception as e:
    print(f"Error during upload: {e}")

print("Upload complete!")
