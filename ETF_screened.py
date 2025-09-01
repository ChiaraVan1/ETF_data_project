"""
ETF监控模型 - 筛选策略执行
--------------------
该脚本用于构建ETF监控系统的第二步：
1. 加载ETF数据报告（由ETF.py生成）。
2. 根据预设的量化策略，筛选出符合条件的ETF。
3. 策略综合考量了基金管理能力（超额收益、跟踪误差）和市场情绪（流动性、成交额比值）。

作者：[ChiaraVan]
创建日期：[30/08/2025]
"""

import pandas as pd
import os

def run_screener_strategy():
    """
    执行ETF筛选策略。
    """
    # 确保数据文件存在
    data_file = 'etf_metrics_daily_report.csv'
    if not os.path.exists(data_file):
        print(f"错误: 找不到数据文件 '{data_file}'。请先运行 ETF_data_fetcher.py。")
        return

    # --- Step 1: 加载数据并预处理 ---
    df_funds = pd.read_csv(data_file)
    print("已成功加载数据报告。")

    columns_to_convert = ['excess_return_mean', 'tracking_error', 'turnover_rate', 'turnover_6m_vs_3y']
    for col in columns_to_convert:
        df_funds[col] = pd.to_numeric(df_funds[col], errors='coerce')

    df_funds.dropna(subset=columns_to_convert, inplace=True)

    # --- Step 2: 定义筛选阈值 ---
    # 策略核心：找到管理能力优秀且有资金流入的ETF。
    
    # 条件1: 流动性加速 - 近6个月成交额均值高于近3年均值
    turnover_6m_vs_3y_threshold = 1

    # 条件2: 相对换手率 - 换手率高于所有基金的10%分位数，确保基本流动性
    turnover_rate_threshold = df_funds['turnover_rate'].quantile(0.1)

    # 条件3 & 4: 相对管理能力 - 超额收益高于行业平均，跟踪误差低于行业平均
    industry_metrics_mean = df_funds.groupby('industry').agg({
        'excess_return_mean': 'mean',
        'tracking_error': 'mean'
    }).to_dict('index')

    # --- Step 3: 应用组合筛选策略 ---
    screened_etfs_list = []

    for index, row in df_funds.iterrows():
        industry = row['industry']
        
        # 确保基金所属行业在行业均值字典中
        if industry in industry_metrics_mean:
            excess_return_threshold = industry_metrics_mean[industry]['excess_return_mean']
            tracking_error_threshold = industry_metrics_mean[industry]['tracking_error']

            # 应用4个组合条件
            cond1 = row['turnover_6m_vs_3y'] > turnover_6m_vs_3y_threshold
            cond2 = row['tracking_error'] < tracking_error_threshold
            cond3 = row['excess_return_mean'] > excess_return_threshold
            cond4 = row['turnover_rate'] > turnover_rate_threshold

            if cond1 and cond2 and cond3 and cond4:
                screened_etfs_list.append(row)

    df_final_screened = pd.DataFrame(screened_etfs_list)

    # --- Step 4: 结果展示与保存 ---
    if not df_final_screened.empty:
        df_final_screened = df_final_screened[['ts_code', 'name', 'industry', 'invest_type', 'turnover_rate', 'turnover_6m_vs_3y', 'excess_return_mean', 'tracking_error']]
        df_final_screened.rename(columns={
            'turnover_rate': '换手率',
            'turnover_6m_vs_3y': '换手率6个月比3年',
            'excess_return_mean': '超额收益均值',
            'tracking_error': '追踪误差'
        }, inplace=True)
        
        output_filename = 'etf_screener_results_strategy1.csv'
        df_final_screened.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n根据你的策略，成功筛选出 {len(df_final_screened)} 只ETF。结果已保存到 {output_filename} 文件中。")
    else:
        print("\n根据你的4条策略，没有筛选出符合条件的ETF。")

if __name__ == '__main__':
    run_screener_strategy()
