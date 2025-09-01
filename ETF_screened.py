"""
ETF监控模型 - 筛选策略执行 (双模式版)
--------------------
该脚本用于构建ETF监控系统的第二步：
1. 加载ETF数据报告（由ETF.py生成）。
2. 根据预设的“常规”或“反转”模式，筛选出符合条件的ETF。
3. 策略综合考量了基金管理能力（超额收益、跟踪误差）、近期表现趋势、流动性以及潜在的市场背离信号。

作者：[ChiaraVan]
创建日期：[30/08/2025]
"""

import pandas as pd
import numpy as np
import os

# --- 辅助函数：量化超额收益移动平均线趋势 ---
def calculate_ma_slope(row):
    """
    计算超额收益移动平均线趋势的斜率。
    斜率为正表示近期表现改善，为负表示恶化。
    """
    y = np.array([row['excess_return_5d_ma'], 
                  row['excess_return_10d_ma'], 
                  row['excess_return_15d_ma'], 
                  row['excess_return_20d_ma']])
    x = np.array([5, 10, 15, 20])
    try:
        slope = np.polyfit(x, y, 1)[0]
    except np.linalg.LinAlgError:
        slope = np.nan
    return slope

# --- 核心筛选逻辑函数 ---
def perform_screening(df_funds, mode='normal'):
    """
    执行具体的ETF筛选逻辑，可选择模式。
    Args:
        df_funds (pd.DataFrame): 包含所有ETF指标的DataFrame。
        mode (str): 筛选模式，可选 'normal' (常规模式) 或 'reversal' (反转模式)。
    Returns:
        pd.DataFrame: 筛选后的DataFrame。
    """
    industry_metrics_mean = df_funds.groupby('industry').agg({
        'excess_return_mean': 'mean',
        'tracking_error': 'mean'
    }).to_dict('index')
    
    screened_etfs_list = []

    for index, row in df_funds.iterrows():
        industry = row['industry']
        
        if industry in industry_metrics_mean:
            excess_return_threshold = industry_metrics_mean[industry]['excess_return_mean']
            tracking_error_threshold = industry_metrics_mean[industry]['tracking_error']

            # --- 常规筛选模式 ('normal') ---
            if mode == 'normal':
                cond1 = row['excess_return_mean'] > excess_return_threshold
                cond2 = row['tracking_error'] < tracking_error_threshold
                cond3 = row['ma_trend_slope'] > 0
                cond4 = row['is_price_turnover_divergence'] == False

                if cond1 and cond2 and cond3 and cond4:
                    screened_etfs_list.append(row)

            # --- 反转机会捕捉模式 ('reversal') ---
            elif mode == 'reversal':
                cond1 = row['is_price_turnover_divergence'] == True
                cond2 = row['excess_return_mean'] < excess_return_threshold
                cond3 = row['turnover_acceleration'] > 1.2
                cond4 = row['turnover_quantile'] > 0.75
                if cond1 and cond2 and cond3 and cond4:
                    screened_etfs_list.append(row)

    return pd.DataFrame(screened_etfs_list)

# --- 主函数：运行所有模式 ---
def run_all_modes():
    """
    加载数据并依次运行常规模式和反转模式。
    """
    data_file = 'etf_metrics_daily_report.csv'
    if not os.path.exists(data_file):
        print(f"错误: 找不到数据文件 '{data_file}'。请先运行 ETF_data_fetcher.py。")
        return

    # 加载数据并进行预处理
    df_funds = pd.read_csv(data_file)
    print("已成功加载数据报告。")
    columns_to_convert = ['excess_return_mean', 'tracking_error', 'turnover_rate', 'turnover_6m_vs_3y',
                          'excess_return_5d_ma', 'excess_return_10d_ma', 'excess_return_15d_ma', 'excess_return_20d_ma',
                          'turnover_acceleration', 'turnover_quantile']
    for col in columns_to_convert:
        df_funds[col] = pd.to_numeric(df_funds[col], errors='coerce')
    df_funds['ma_trend_slope'] = df_funds.apply(calculate_ma_slope, axis=1)
    df_funds.dropna(subset=['excess_return_mean', 'tracking_error', 'turnover_rate', 'ma_trend_slope', 'turnover_quantile'], inplace=True)

    # 运行常规模式
    print("\n--- 正在运行常规筛选模式 ---")
    df_normal = perform_screening(df_funds, mode='normal')
    if not df_normal.empty:
        # 数值格式化和列名优化
        df_normal['超额收益均值'] = (df_normal['excess_return_mean'] * 100).round(2)
        df_normal['追踪误差'] = (df_normal['tracking_error'] * 100).round(2)
        df_normal['换手率'] = (df_normal['turnover_rate'] * 100).round(2)
        df_normal['超额收益趋势斜率'] = df_normal['ma_trend_slope'].round(4)
        df_normal['换手率6个月比3年'] = df_normal['turnover_6m_vs_3y'].round(2)
        df_normal['行业内成交额占比'] = (df_normal['turnover_pct_in_industry'] * 100).round(2)
        
        df_normal = df_normal[['ts_code', 'name', 'industry', 'invest_type',
                               '换手率', '换手率6个月比3年', '超额收益均值', '追踪误差',
                               '超额收益趋势斜率', 'is_price_turnover_divergence', '行业内成交额占比']]
        df_normal.rename(columns={'is_price_turnover_divergence': '价格成交额背离'}, inplace=True)

        output_filename = 'etf_screener_results_normal_mode.csv'
        df_normal.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"在'normal'模式下，成功筛选出 {len(df_normal)} 只ETF。结果已保存到 {output_filename} 文件中。")
    else:
        print("在'normal'模式下，没有筛选出符合条件的ETF。")

    # 运行反转模式
    print("\n--- 正在运行反转机会捕捉模式 ---")
    df_reversal = perform_screening(df_funds, mode='reversal')
    if not df_reversal.empty:
        # 数值格式化和列名优化
        df_reversal['超额收益均值'] = (df_reversal['excess_return_mean'] * 100).round(2)
        df_reversal['追踪误差'] = (df_reversal['tracking_error'] * 100).round(2)
        df_reversal['换手率'] = (df_reversal['turnover_rate'] * 100).round(2)
        df_reversal['超额收益趋势斜率'] = df_reversal['ma_trend_slope'].round(4)
        df_reversal['资金流加速度'] = df_reversal['turnover_acceleration'].round(2)
        df_reversal['资金流分位数'] = (df_reversal['turnover_quantile'] * 100).round(2)
        df_reversal['换手率6个月比3年'] = df_reversal['turnover_6m_vs_3y'].round(2)
        df_reversal['行业内成交额占比'] = (df_reversal['turnover_pct_in_industry'] * 100).round(2)
        
        df_reversal = df_reversal[['ts_code', 'name', 'industry', 'invest_type',
                                   '价格成交额背离', '资金流加速度', '资金流分位数',
                                   '换手率', '换手率6个月比3年', '超额收益均值', '追踪误差',
                                   '超额收益趋势斜率', '行业内成交额占比']]
        df_reversal.rename(columns={'is_price_turnover_divergence': '价格成交额背离'}, inplace=True)

        output_filename = 'etf_screener_results_reversal_mode.csv'
        df_reversal.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"在'reversal'模式下，成功筛选出 {len(df_reversal)} 只ETF。结果已保存到 {output_filename} 文件中。")
    else:
        print("在'reversal'模式下，没有筛选出符合条件的ETF。")

if __name__ == '__main__':
    run_all_modes()
