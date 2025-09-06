import os
import oss2

# 从环境变量中获取密钥和配置
access_key_id = os.environ.get('ALIYUN_ACCESS_KEY_ID')
access_key_secret = os.environ.get('ALIYUN_ACCESS_KEY_SECRET')
endpoint = os.environ.get('OSS_ENDPOINT')
bucket_name = os.environ.get('OSS_BUCKET_NAME')

# 初始化 OSS 客户端
# 使用密钥进行身份验证
auth = oss2.Auth(access_key_id, access_key_secret)
# 这行代码创建一个 bucket 对象，用于后续操作
bucket = oss2.Bucket(auth, endpoint, bucket_name)

# 遍历当前目录，找到所有 .csv 文件
# 你的 ETF 脚本运行后，.csv 文件就在当前目录中
print("Starting file upload...")
for filename in os.listdir('.'):
    # 检查文件是否以 .csv 结尾
    if filename.endswith('.csv'):
        # 确定在 OSS 存储桶中的路径
        # os.environ.get('OSS_DEST_DIR', '') 默认是 '/'
        oss_path = os.path.join(os.environ.get('OSS_DEST_DIR', ''), filename)
        # 核心步骤：将文件从本地上传到 OSS
        # put_object_from_file() 方法就是用来实现这个功能的
        bucket.put_object_from_file(oss_path, filename)
        print(f"Successfully uploaded {filename} to OSS bucket {bucket_name}.")

print("Upload complete!")
