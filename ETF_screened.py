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

# --- 辅助函数：计算超额收益移动平均线趋势斜率 ---
def calculate_ma_slope(row):
    """
    计算超额收益移动平均线趋势的斜率。
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

# --- 核心策略执行函数 ---
def perform_strategies(df_funds):
    """
    根据所有策略逻辑，为每只ETF打标签，并提供筛选理由。
    """
    results_list = []
    
    # 计算行业平均值，用于筛选基准
    industry_metrics_mean = df_funds.groupby('industry').agg({
        'excess_return_mean': 'mean',
        'tracking_error': 'mean'
    }).to_dict('index')
    
    for index, row in df_funds.iterrows():
        # 初始化结果
        strategy = ""
        reason = ""

        # --- 买入机会 (Buy_Signal) 策略 ---
        buy_cond = (
            # 绝对值：安全门槛
            (row['issue_amount'] >= 2.0) and
            (pd.notna(row['turnover_rate']) and row['turnover_rate'] >= df_funds['turnover_rate'].quantile(0.2)) and
            (pd.notna(row['latest_discount_rate']) and row['latest_discount_rate'] <= 0.02) and # 折溢价小于2%
            (pd.notna(row['annualized_volatility']) and row['annualized_volatility'] <= 0.40) and # 年化波动率小于40%
            (pd.notna(row['max_drawdown']) and row['max_drawdown'] <= 0.50) and # 最大回撤小于50%
            (pd.notna(row['discount_quantile_1y']) and row['discount_quantile_1y'] <= 0.8) and
            (pd.notna(row['volatility_quantile_1y']) and row['volatility_quantile_1y'] <= 0.9) and
            (pd.notna(row['max_drawdown_quantile_1y']) and row['max_drawdown_quantile_1y'] <= 0.9)
        ) and (
            # 相对值：动态机会
            (pd.notna(row['turnover_acceleration']) and row['turnover_acceleration'] > 1.2) or
            (pd.notna(row['turnover_quantile']) and row['turnover_quantile'] >= 0.75) or
            (pd.notna(row['ma_trend_slope']) and row['ma_trend_slope'] > 0) or
            (pd.notna(row['excess_return_vs_yoy']) and row['excess_return_vs_yoy'] > 0) or
            (pd.notna(row['excess_return_mean']) and row['excess_return_mean'] > industry_metrics_mean.get(row['industry'], {}).get('excess_return_mean', -100))
        )

        if buy_cond:
            strategy = "买入机会"
            reasons = []
            if pd.notna(row['turnover_acceleration']) and row['turnover_acceleration'] > 1.2: reasons.append("资金流入加速")
            if pd.notna(row['turnover_quantile']) and row['turnover_quantile'] >= 0.75: reasons.append("成交分位高位")
            if pd.notna(row['ma_trend_slope']) and row['ma_trend_slope'] > 0: reasons.append("超额收益趋势改善")
            if pd.notna(row['excess_return_vs_yoy']) and row['excess_return_vs_yoy'] > 0: reasons.append("超额收益优于去年同期")
            if pd.notna(row['excess_return_mean']) and row['excess_return_mean'] > industry_metrics_mean.get(row['industry'], {}).get('excess_return_mean', -100): reasons.append("长期超额为正")
            reason = "、".join(reasons)
        
        # --- 低买 (Low_Buy) 策略 ---
        low_buy_cond = (
            (pd.notna(row['discount_quantile_1y']) and row['discount_quantile_1y'] < 0.2) and
            (pd.notna(row['change_10d_discount']) and row['change_10d_discount'] < -0.01) and # 10日折价率加深超过1%
            (pd.notna(row['turnover_quantile']) and row['turnover_quantile'] <= 0.3) and
            (pd.notna(row['issue_amount']) and row['issue_amount'] >= 5.0) and
            (pd.notna(row['volatility_quantile_1y']) and row['volatility_quantile_1y'] <= 0.8) and
            (pd.notna(row['max_drawdown_quantile_1y']) and row['max_drawdown_quantile_1y'] <= 0.8)
        )
        if not strategy and low_buy_cond:
            strategy = "低买"
            reasons = []
            if pd.notna(row['discount_quantile_1y']) and row['discount_quantile_1y'] < 0.2: reasons.append("估值历史低位")
            if pd.notna(row['change_10d_discount']) and row['change_10d_discount'] < -0.01: reasons.append("折价率快速加深")
            if pd.notna(row['turnover_quantile']) and row['turnover_quantile'] <= 0.3: reasons.append("资金流处于冰点")
            reason = "、".join(reasons)

        # --- 卖出警示 (Sell_Alert) 策略 ---
        # 仅针对满足买入条件的基金进行警示
        if strategy == "买入机会":
            valuation_cond = (pd.notna(row['discount_quantile_1y']) and row['discount_quantile_1y'] > 0.9) or (pd.notna(row['latest_discount_rate']) and row['latest_discount_rate'] > 0.02) # 估值高位
            capital_cond = (
                (pd.notna(row['turnover_acceleration']) and row['turnover_acceleration'] < 0.8) or
                (pd.notna(row['turnover_quantile']) and row['turnover_quantile'] <= 0.3) or
                (pd.notna(row['is_price_turnover_divergence']) and row['is_price_turnover_divergence'] == True)
            )
            risk_cond = (
                (pd.notna(row['volatility_slope']) and row['volatility_slope'] > 0.0001) or
                (pd.notna(row['max_drawdown_slope']) and row['max_drawdown_slope'] < -0.0001)
            )
            
            if valuation_cond or capital_cond or risk_cond:
                strategy = "卖出警示"
                reasons = []
                if valuation_cond: reasons.append("估值高位")
                if capital_cond: reasons.append("资金撤退或背离")
                if risk_cond: reasons.append("风险恶化")
                reason = "、".join(reasons)

        row['Strategy'] = strategy
        row['Reason'] = reason
        results_list.append(row)
    
    return pd.DataFrame(results_list)

# --- 主函数：运行所有策略并生成报告 ---
def run_all_strategies():
    data_file = 'etf_metrics_daily_report.csv'
    if not os.path.exists(data_file):
        print(f"错误: 找不到数据文件 '{data_file}'。请先运行 ETF.py 生成数据报告。")
        return

    df_funds = pd.read_csv(data_file)
    print("已成功加载数据报告。")

    df_result = perform_strategies(df_funds)
    
    df_final = df_result[df_result['Strategy'] != ""].copy()
    
    # 最终报告格式化
    df_final['超额收益均值(%)'] = (df_final['excess_return_mean'] * 100).round(2)
    df_final['追踪误差(%)'] = (df_final['tracking_error'] * 100).round(2)
    df_final['换手率(%)'] = (df_final['turnover_rate'] * 100).round(2)
    df_final['超额收益趋势斜率(万分之)'] = df_final['ma_trend_slope'].round(4)
    df_final['资金流加速度(倍)'] = (df_final['turnover_acceleration']).round(2)
    df_final['资金流分位数(%)'] = (df_final['turnover_quantile'] * 100).round(2)
    df_final['价格成交额背离'] = df_final['is_price_turnover_divergence'].apply(lambda x: '是' if x else '否')
    df_final['最新折价率(%)'] = (df_final['latest_discount_rate'] * 100).round(2)
    df_final['折价率1年分位(%)'] = (df_final['discount_quantile_1y'] * 100).round(2)
    df_final['波动率1年分位(%)'] = (df_final['volatility_quantile_1y'] * 100).round(2)
    df_final['最大回撤1年分位(%)'] = (df_final['max_drawdown_quantile_1y'] * 100).round(2)
    df_final['波动率斜率'] = df_final['volatility_slope'].round(4)
    df_final['最大回撤斜率'] = df_final['max_drawdown_slope'].round(4)
    
    output_columns = [
        'ts_code', 'name', 'industry', 'invest_type', 'Strategy', 'Reason',
        '超额收益均值(%)', '追踪误差(%)', '超额收益趋势斜率(万分之)', '换手率(%)', 
        '资金流加速度(倍)', '资金流分位数(%)', '价格成交额背离',
        '最新折价率(%)', '折价率1年分位(%)', '波动率1年分位(%)', '最大回撤1年分位(%)',
        '波动率斜率', '最大回撤斜率'
    ]
    df_final = df_final[output_columns]

    output_filename = 'etf_screener_final_report.csv'
    df_final.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n成功筛选出 {len(df_final)} 只ETF。最终报告已保存到 {output_filename} 文件中。")

if __name__ == '__main__':
    run_all_strategies()
