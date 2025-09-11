import pandas as pd
import os

# 定义数据文件名
DATA_FILE_NAME = 'etf_screener_final_report.csv'

# 定义新的输出文件名，用于运营
DAILY_VIDEO_FILE = 'daily_video_top1.csv'
WEEKLY_VIDEO_FILE = 'weekly_video_top3.csv'

# 定义需要保留的列
COLUMNS_TO_KEEP = [
    'ts_code',
    'name',
    'industry',
    'Strategy',
    'Reason',
    '超额收益均值(%)',
    '超额收益趋势斜率(万分之)'
]

def process_etf_data(df, top_n):
    """
    处理并筛选ETF数据。
    
    Args:
        df (pd.DataFrame): 原始ETF数据。
        top_n (int): 需要筛选出的ETF数量，用于日更或周总结。
    
    Returns:
        pd.DataFrame: 筛选并排序后的ETF数据。
    """
    print(f"--- 正在筛选前 {top_n} 只ETF ---")
    
    # 按照 超额收益均值(%) 降序排序
    df['超额收益均值(%)'] = pd.to_numeric(df['超额收益均值(%)'], errors='coerce')
    df_sorted = df.sort_values(by='超额收益均值(%)', ascending=False)
    
    # 选取 Top N 的数据
    df_top_n = df_sorted.head(top_n)
    
    # 选取需要的列
    df_processed = df_top_n[COLUMNS_TO_KEEP]
    
    return df_processed

def main():
    """
    主函数，执行数据处理流程。
    """
    if not os.path.exists(DATA_FILE_NAME):
        print(f"错误: 找不到文件 {DATA_FILE_NAME}。请确保该文件已存在。")
        return

    print("--- 1. 正在从本地文件读取数据 ---")
    try:
        df_raw = pd.read_csv(DATA_FILE_NAME)
        print("数据读取成功！")
        
        # --- 规则1：日更短视频（Top 1） ---
        print("\n--- 2. 处理日更短视频数据（Top 1） ---")
        df_daily_top1 = process_etf_data(df_raw, 1)
        print("日更Top 1数据:")
        print(df_daily_top1.to_string())
        
        # --- 规则1：周总结视频（Top 3） ---
        print("\n--- 3. 处理周总结视频数据（Top 3） ---")
        df_weekly_top3 = process_etf_data(df_raw, 3)
        print("周总结Top 3数据:")
        print(df_weekly_top3.to_string())
        
        # 保存为新的、更具描述性的文件名
        df_daily_top1.to_csv(DAILY_VIDEO_FILE, index=False, encoding='utf-8-sig')
        df_weekly_top3.to_csv(WEEKLY_VIDEO_FILE, index=False, encoding='utf-8-sig')
        print(f"\n已将日更数据保存至: {DAILY_VIDEO_FILE}")
        print(f"已将周总结数据保存至: {WEEKLY_VIDEO_FILE}")
        
    except Exception as e:
        print(f"处理数据时出错: {e}")

if __name__ == '__main__':
    main()
