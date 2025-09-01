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

def run_screener_strategy(mode='normal'):
    """
    执行ETF筛选策略，可选择模式。

    Args:
        mode (str): 筛选模式，可选 'normal' (常规模式) 或 'reversal' (反转模式)。
    """
    data_file = 'etf_metrics_daily_report.csv'
    if not os.path.exists(data_file):
        print(f"错误: 找不到数据文件 '{data_file}'。请先运行 ETF.py。")
        return

    # --- Step 1: 加载数据并预处理 ---
    df_funds = pd.read_csv(data_file)
    print("已成功加载数据报告。")

    # 转换所有关键列为数值类型
    columns_to_convert = ['excess_return_mean', 'tracking_error', 'turnover_rate', 'turnover_6m_vs_3y',
                          'excess_return_5d_ma', 'excess_return_10d_ma', 'excess_return_15d_ma', 'excess_return_20d_ma',
                          'turnover_acceleration', 'turnover_quantile']
    for col in columns_to_convert:
        df_funds[col] = pd.to_numeric(df_funds[col], errors='coerce')

    # 计算超额收益移动平均线斜率
    df_funds['ma_trend_slope'] = df_funds.apply(calculate_ma_slope, axis=1)

    # 剔除关键指标缺失的行
    df_funds.dropna(subset=['excess_return_mean', 'tracking_error', 'turnover_rate', 'ma_trend_slope', 'turnover_quantile'], inplace=True)

    # --- Step 2: 定义筛选条件 ---
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

            # --- 常规筛选模式 ---
            if mode == 'normal':
                # 条件1: 长期超额收益高于行业均值
                cond1 = row['excess_return_mean'] > excess_return_threshold
                # 条件2: 长期跟踪误差低于行业均值
                cond2 = row['tracking_error'] < tracking_error_threshold
                # 条件3: 超额收益移动平均线趋势向好
                cond3 = row['ma_trend_slope'] > 0
                # 条件4: 不存在价格与成交额背离
                cond4 = row['is_price_turnover_divergence'] == False

                if cond1 and cond2 and cond3 and cond4:
                    screened_etfs_list.append(row)

            # --- 反转机会捕捉模式 ---
            elif mode == 'reversal':
                # 条件1: 存在价格与成交额背离
                cond1 = row['is_price_turnover_divergence'] == True
                # 条件2: 长期超额收益低于行业均值（可能处于底部）
                cond2 = row['excess_return_mean'] < excess_return_threshold
                # 条件3: 资金流向加速（成交额加速比大于1.2，比常规策略更严格）
                cond3 = row['turnover_acceleration'] > 1.2
                # 条件4: 近期成交额处于高位（分位数大于75%）
                cond4 = row['turnover_quantile'] > 0.75
                
                if cond1 and cond2 and cond3 and cond4:
                    screened_etfs_list.append(row)

    df_final_screened = pd.DataFrame(screened_etfs_list)

    # --- Step 3: 结果展示与保存 ---
    if not df_final_screened.empty:
        df_final_screened = df_final_screened[['ts_code', 'name', 'industry', 'invest_type',
                                               'turnover_rate', 'turnover_6m_vs_3y', 'excess_return_mean', 'tracking_error',
                                               'ma_trend_slope', 'is_price_turnover_divergence', 'turnover_pct_in_industry',
                                               'turnover_acceleration', 'turnover_quantile']]
        df_final_screened.rename(columns={
            'turnover_rate': '换手率', 'turnover_6m_vs_3y': '换手率6个月比3年', 'excess_return_mean': '超额收益均值', 
            'tracking_error': '追踪误差', 'ma_trend_slope': '超额收益趋势斜率', 
            'is_price_turnover_divergence': '价格成交额背离', 'turnover_pct_in_industry': '行业内成交额占比',
            'turnover_acceleration': '资金流加速度', 'turnover_quantile': '资金流分位数'
        }, inplace=True)
        
        output_filename = f'etf_screener_results_{mode}_mode.csv'
        df_final_screened.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n在'{mode}'模式下，成功筛选出 {len(df_final_screened)} 只ETF。结果已保存到 {output_filename} 文件中。")
    else:
        print(f"\n在'{mode}'模式下，没有筛选出符合条件的ETF。请考虑调整筛选条件或切换模式。")

if __name__ == '__main__':
    # 示例用法：
    # 运行常规筛选模式
    run_screener_strategy(mode='normal')

    # 运行反转机会捕捉模式
    # run_screener_strategy(mode='reversal')
