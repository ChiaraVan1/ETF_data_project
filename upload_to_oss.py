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
    # Define the files to be uploaded
    files_to_upload = [
        'etf_metrics_daily_report.csv',      # The raw metrics report
        'etf_screener_final_report.csv'      # The final screening report
    ]

    for file_name in files_to_upload:
        # Check if the file exists before attempting to upload
        if os.path.exists(file_name):
            bucket.put_object_from_file(file_name, file_name)
            print(f"Successfully uploaded {file_name} to OSS.")
        else:
            print(f"Warning: The file '{file_name}' was not found. Skipping upload.")

except Exception as e:
    print(f"Error during upload: {e}")

print("Upload complete!")
