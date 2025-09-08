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
    
    # 获取整个数据集的换手率分位数，用于基准比较
    turnover_rate_quantile_20 = df_funds['turnover_rate'].quantile(0.2)
    
    # --- 调试代码开始 ---
    print("\n--- 调试模式: 正在检查数据和筛选条件 ---")
    print(f"数据总行数: {len(df_funds)}")
    print(f"行业超额收益平均值: {industry_metrics_mean}")
    print(f"全市场换手率 20% 分位点: {turnover_rate_quantile_20:.6f}")
    # --- 调试代码结束 ---

    for index, row in df_funds.iterrows():
        # 初始化结果
        strategy = ""
        reason = ""

        # --- 买入机会 (Buy_Signal) 策略 ---
        # 综合所有买入条件
        
        # --- 调试代码开始 ---
        print(f"\n--- 正在检查基金: {row['name']} ({row['ts_code']}) ---")
        # --- 调试代码结束 ---

        # 检查安全（Safe）条件
        cond_safe_1 = pd.notna(row['issue_amount']) and row['issue_amount'] >= 2.0
        cond_safe_2 = pd.notna(row['turnover_rate']) and row['turnover_rate'] >= turnover_rate_quantile_20
        cond_safe_3 = pd.notna(row['latest_discount_rate']) and row['latest_discount_rate'] <= 0.02
        cond_safe_4 = pd.notna(row['annualized_volatility']) and row['annualized_volatility'] <= 30
        cond_safe_5 = pd.notna(row['max_drawdown']) and row['max_drawdown'] <= 0.40
        cond_safe_6 = pd.notna(row['discount_quantile_1y']) and row['discount_quantile_1y'] <= 0.8
        cond_safe_7 = pd.notna(row['volatility_quantile_1y']) and row['volatility_quantile_1y'] <= 0.9
        cond_safe_8 = pd.notna(row['max_drawdown_quantile_1y']) and row['max_drawdown_quantile_1y'] <= 0.9
        
        is_safe = (cond_safe_1 and cond_safe_2 and cond_safe_3 and cond_safe_4 and 
                   cond_safe_5 and cond_safe_6 and cond_safe_7 and cond_safe_8)

        # 检查动态（Dynamic）条件
        industry_excess_mean = industry_metrics_mean.get(row['industry'], {}).get('excess_return_mean', -100)
        cond_dynamic_1 = pd.notna(row['turnover_acceleration']) and row['turnover_acceleration'] > 1.2
        cond_dynamic_2 = pd.notna(row['turnover_quantile']) and row['turnover_quantile'] >= 0.75
        cond_dynamic_3 = pd.notna(row['ma_trend_slope']) and row['ma_trend_slope'] > 0
        cond_dynamic_4 = pd.notna(row['excess_return_vs_yoy']) and row['excess_return_vs_yoy'] > 0
        cond_dynamic_5 = pd.notna(row['excess_return_mean']) and row['excess_return_mean'] > industry_excess_mean

        is_dynamic = (cond_dynamic_1 or cond_dynamic_2 or cond_dynamic_3 or 
                      cond_dynamic_4 or cond_dynamic_5)

        # --- 调试代码开始 ---
        print(f"  --- 安全条件检查 (is_safe) ---")
        print(f"  1. 发行规模 >= 2.0亿: {cond_safe_1} (当前值: {row['issue_amount']})")
        print(f"  2. 换手率 >= 20%分位点: {cond_safe_2} (当前值: {row['turnover_rate']:.6f})")
        print(f"  3. 最新折价率 <= 2%: {cond_safe_3} (当前值: {row['latest_discount_rate']:.4f})")
        print(f"  4. 历史波动率 <= 40%: {cond_safe_4} (当前值: {row['annualized_volatility']:.4f})")
        print(f"  5. 历史最大回撤 <= 50%: {cond_safe_5} (当前值: {row['max_drawdown']:.4f})")
        print(f"  6. 折价率1年分位 <= 80%: {cond_safe_6} (当前值: {row['discount_quantile_1y']:.2f})")
        print(f"  7. 波动率1年分位 <= 90%: {cond_safe_7} (当前值: {row['volatility_quantile_1y']:.2f})")
        print(f"  8. 最大回撤1年分位 <= 90%: {cond_safe_8} (当前值: {row['max_drawdown_quantile_1y']:.2f})")
        print(f"  -> 综合安全条件结果: {'✅' if is_safe else '❌'}")
        
        print(f"  --- 动态条件检查 (is_dynamic) ---")
        print(f"  1. 资金流加速度 > 1.2: {cond_dynamic_1} (当前值: {row['turnover_acceleration']:.2f})")
        print(f"  2. 成交分位 >= 75%: {cond_dynamic_2} (当前值: {row['turnover_quantile']:.2f})")
        print(f"  3. 超额收益趋势斜率 > 0: {cond_dynamic_3} (当前值: {row['ma_trend_slope']:.4f})")
        print(f"  4. 超额收益优于去年同期: {cond_dynamic_4} (当前值: {row['excess_return_vs_yoy']:.4f})")
        print(f"  5. 长期超额收益为正: {cond_dynamic_5} (当前值: {row['excess_return_mean']:.4f}, 行业均值: {industry_excess_mean:.4f})")
        print(f"  -> 综合动态条件结果: {'✅' if is_dynamic else '❌'}")
        
        # --- 调试代码结束 ---

        if is_safe and is_dynamic:
            strategy = "买入机会"
            reasons = []
            if cond_dynamic_1: reasons.append("资金流入加速")
            if cond_dynamic_2: reasons.append("成交分位高位")
            if cond_dynamic_3: reasons.append("超额收益趋势改善")
            if cond_dynamic_4: reasons.append("超额收益优于去年同期")
            if cond_dynamic_5: reasons.append("长期超额为正")
            reason = "、".join(reasons)

        # --- 低买 (Low_Buy) 策略 ---
        low_buy_cond = (
            (pd.notna(row['discount_quantile_1y']) and row['discount_quantile_1y'] < 0.2) and
            (pd.notna(row['change_10d_discount']) and row['change_10d_discount'] < -0.01) and
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
        if strategy == "买入机会":
            valuation_cond = (pd.notna(row['latest_discount_rate']) and row['latest_discount_rate'] > 0.02)
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
