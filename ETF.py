"""
ETF监控模型
--------------------
该脚本用于构建一个ETF监控系统。
主要功能包括：
1. 获取特定ETF的历史数据。
2. 计算和监控关键指标，例如跟踪误差、流动性（成交量/额）和估值。
3. 帮助投资者更好地理解和筛选ETF，作为“小助手”提供理性投资机会的提醒。

作者：[ChiaraVan]
创建日期：[30/08/2025]
"""


import tushare as ts
import pandas as pd
import re
from datetime import datetime, timedelta
import numpy as np
import os

token = os.environ.get('TUSHARE_TOKEN')
if token:
    ts.set_token(token)
    pro = ts.pro_api()
else:
    raise ValueError("TUSHARE_TOKEN is not set in environment variables.")

def run_data_fetcher():
    """
    执行整个数据获取和指标计算流程。
    """
    print("--- 1. 获取并筛选目标行业ETF ---")
    df_etf_list = pro.fund_basic(market='E')
    df_offshore_list = pro.fund_basic(market='O')
    df_all_funds = pd.concat([df_etf_list, df_offshore_list], ignore_index=True)
    print(f"合并后，总共获取到 {len(df_all_funds)} 只基金的信息。")

    # 根据基金名称或基准名称中的关键词，筛选出消费、科技、医疗行业的基金。
    keywords_consumer = ['消费', '食品', '酒', '饮料']
    keywords_tech = ['科技', '信息技术', '芯片', '半导体', '计算机', '通信', '新能源']
    keywords_healthcare = ['医疗', '医药', '生物', '创新药', '健康']

    name_col = 'name'
    benchmark_col = 'benchmark'

    condition_consumer = (df_all_funds[name_col].str.contains('|'.join(keywords_consumer), na=False)) | \
                         (df_all_funds[benchmark_col].str.contains('|'.join(keywords_consumer), na=False))
    condition_tech = (df_all_funds[name_col].str.contains('|'.join(keywords_tech), na=False)) | \
                     (df_all_funds[benchmark_col].str.contains('|'.join(keywords_tech), na=False))
    condition_healthcare = (df_all_funds[name_col].str.contains('|'.join(keywords_healthcare), na=False)) | \
                           (df_all_funds[benchmark_col].str.contains('|'.join(keywords_healthcare), na=False))

    df_selected_funds = df_all_funds[condition_consumer | condition_tech | condition_healthcare].copy()
    df_selected_funds.loc[condition_consumer[condition_consumer.index], 'industry'] = '消费'
    df_selected_funds.loc[condition_tech[condition_tech.index], 'industry'] = '科技'
    df_selected_funds.loc[condition_healthcare[condition_healthcare.index], 'industry'] = '医疗'

    selected_columns = ['ts_code', 'name', 'market', 'list_date', 'm_fee', 'c_fee', 'issue_amount', 'benchmark', 'invest_type', 'industry']
    df_lean = df_selected_funds.loc[:, selected_columns]

    df_lean['list_date'] = pd.to_datetime(df_lean['list_date'], errors='coerce')

    def clean_benchmark_name(name):
        if pd.isna(name):
            return None
        cleaned_name = re.sub(r'\*.*', '', name)
        cleaned_name = cleaned_name.replace('收益率', '')
        return cleaned_name.strip()

    df_lean['cleaned_benchmark_name'] = df_lean['benchmark'].apply(clean_benchmark_name)

    print("\n--- 2. 匹配基金与其基准指数 ---")
    df_csi = pro.index_basic(market='CSI')
    df_sse = pro.index_basic(market='SSE')
    df_szse = pro.index_basic(market='SZSE')
    df_index_basic = pd.concat([df_csi, df_sse, df_szse], ignore_index=True)
    df_index_basic['list_date'] = pd.to_datetime(df_index_basic['list_date'], errors='coerce')

    def robust_map_name_to_code(row, df_index_basic):
        cleaned_fund_name = row['cleaned_benchmark_name']
        fund_list_date = row['list_date']
        
        # 手动映射，解决一些特殊情况
        manual_map = {
            '沪深300指数': '000300.SH',
            '创业板指数': '399006.SZ',
            '中证消费电子主题指数': '931104.CSI',
            '上证科创板芯片指数': '000685.SH',
        }
        if cleaned_fund_name in manual_map:
            return manual_map[cleaned_fund_name]
        
        for index, idx_row in df_index_basic.iterrows():
            if pd.notna(cleaned_fund_name) and pd.notna(idx_row['name']) and idx_row['name'] in cleaned_fund_name and \
               pd.notna(fund_list_date) and pd.notna(idx_row['list_date']) and fund_list_date >= idx_row['list_date']:
                return idx_row['ts_code']
        return None

    df_lean['benchmark_code'] = df_lean.apply(robust_map_name_to_code, axis=1, args=(df_index_basic,))

    print("\n--- 3. 筛选并保存最终的基金列表 ---")
    df_funds = df_lean.copy()
    unmatched_funds = df_funds[df_funds['benchmark_code'].isnull()]

    df_funds.to_csv('etf_funds_list_raw.csv', index=False, encoding='utf-8-sig')

    df_funds.dropna(subset=['benchmark_code'], inplace=True)
    min_issue_amount = 2.0  # Tushare 的 issue_amount 单位是“亿元”
    df_funds = df_funds[df_funds['issue_amount'] >= min_issue_amount]
    df_funds.to_csv('etf_funds_list_final.csv', index=False, encoding='utf-8-sig')
    print(f"最终筛选后，可用于计算的基金数量: {len(df_funds)} 只")
    
    if not unmatched_funds.empty:
        print(f"\n未能自动匹配的基金数量: {len(unmatched_funds)} 只")

    print("\n--- 4. 计算超额收益和流动性指标 ---")
    results_list = []
    today = datetime.now().date()
    one_year_ago = (today - timedelta(days=365)).strftime('%Y%m%d')
    three_years_ago = (today - timedelta(days=3 * 365)).strftime('%Y%m%d')

    for index, row in df_funds.iterrows():
        fund_code = row['ts_code']
        benchmark_code = row['benchmark_code']
        aum = row['issue_amount']

        try:
            # 获取基金日行情数据（fund_daily）和指数日行情数据（index_daily）
            df_fund_daily = pro.fund_daily(ts_code=fund_code, start_date=three_years_ago)
            df_index_daily = pro.index_daily(ts_code=benchmark_code, start_date=three_years_ago)

            if df_fund_daily.empty or df_index_daily.empty:
                print(f"警告: 未能获取 {fund_code} 或其基准 {benchmark_code} 的日线数据，跳过。")
                continue

            # 数据预处理
            df_fund_daily['trade_date'] = pd.to_datetime(df_fund_daily['trade_date'])
            df_fund_daily.set_index('trade_date', inplace=True)
            df_fund_daily.sort_index(inplace=True)

            df_index_daily['trade_date'] = pd.to_datetime(df_index_daily['trade_date'])
            df_index_daily.set_index('trade_date', inplace=True)
            df_index_daily.sort_index(inplace=True)

            # --- 计算超额收益指标（参考陈嘉禾《投资的奥妙》中的基金超额收益排名模型）---
            merged_data = pd.merge(df_fund_daily, df_index_daily, left_index=True, right_index=True, how='inner', suffixes=('_fund', '_index'))
            
            if merged_data.empty or len(merged_data) < 20:
                print(f"警告: {fund_code} 数据不足，跳过。")
                continue

            merged_data['excess_return'] = merged_data['pct_chg_fund'] - merged_data['pct_chg_index']
            
            # 计算3年期的超额收益率均值和跟踪误差
            df_excess_3y = merged_data[merged_data.index >= three_years_ago]
            excess_return_mean = df_excess_3y['excess_return'].mean()
            tracking_error = df_excess_3y['excess_return'].std() * np.sqrt(250) # 年化跟踪误差

            # 计算短期超额收益移动平均值，用于监测近期表现
            ma_5 = merged_data['excess_return'].rolling(window=5).mean().iloc[-1]
            ma_10 = merged_data['excess_return'].rolling(window=10).mean().iloc[-1]
            ma_15 = merged_data['excess_return'].rolling(window=15).mean().iloc[-1]
            ma_20 = merged_data['excess_return'].rolling(window=20).mean().iloc[-1]

            # --- 计算流动性与市场情绪指标（参考《投资的奥妙》中的最近表现模型）---
            df_liquidity_1y = df_fund_daily[df_fund_daily.index >= one_year_ago]
            df_liquidity_3y = df_fund_daily[df_fund_daily.index >= three_years_ago]
            df_liquidity_6m = df_liquidity_1y[df_liquidity_1y.index >= (today - timedelta(days=180)).strftime('%Y%m%d')]
            
            turnover_1y_mean = df_liquidity_1y['amount'].mean()
            turnover_rate = turnover_1y_mean / (aum * 100000) if aum > 0 else np.nan
            turnover_6m_mean = df_liquidity_6m['amount'].mean()
            turnover_3y_mean = df_liquidity_3y['amount'].mean()
            turnover_6m_vs_3y = turnover_6m_mean / turnover_3y_mean if turnover_3y_mean > 0 else np.nan
            turnover_1y_std = df_liquidity_1y['amount'].std()
            low_quantile_turnover = df_liquidity_1y['amount'].quantile(0.05)
            
            # 市场情绪指标
            aum_thousands = aum * 100000
            df_fund_weekly = df_fund_daily.resample('W')['amount'].sum().to_frame('amount')
            df_fund_monthly = df_fund_daily.resample('ME')['amount'].sum().to_frame('amount')
            latest_week_turnover = df_fund_weekly['amount'].iloc[-1] if not df_fund_weekly.empty else 0
            latest_month_turnover = df_fund_monthly['amount'].iloc[-1] if not df_fund_monthly.empty else 0
            turnover_ratio_1w = latest_week_turnover / aum_thousands if aum_thousands > 0 else np.nan
            turnover_ratio_1m = latest_month_turnover / aum_thousands if aum_thousands > 0 else np.nan
            turnover_acceleration = latest_week_turnover / latest_month_turnover if latest_month_turnover > 0 else np.nan
            turnover_quantile = np.nan
            if not df_fund_weekly.empty and len(df_fund_weekly) >= 52:
                df_weekly_12m = df_fund_weekly.iloc[-52:]
                turnover_quantile = df_weekly_12m['amount'].rank(pct=True).iloc[-1]
            is_divergence = False
            price_change_1w = 0
            df_price_1w = df_fund_daily.tail(5)
            if len(df_price_1w) > 1:
                price_change_1w = (df_price_1w['close'].iloc[-1] - df_price_1w['close'].iloc[0]) / df_price_1w['close'].iloc[0]
            turnover_change_1w = 0
            if len(df_fund_weekly) > 1:
                turnover_change_1w = latest_week_turnover - df_fund_weekly['amount'].iloc[-2]
            if np.sign(price_change_1w) != np.sign(turnover_change_1w):
                is_divergence = True

            # 存储所有结果
            results_list.append({
                'ts_code': fund_code,
                'excess_return_mean': excess_return_mean,
                'tracking_error': tracking_error,
                'excess_return_5d_ma': ma_5,
                'excess_return_10d_ma': ma_10,
                'excess_return_15d_ma': ma_15,
                'excess_return_20d_ma': ma_20,
                'turnover_1y_mean': turnover_1y_mean,
                'turnover_rate': turnover_rate,
                'turnover_6m_vs_3y': turnover_6m_vs_3y,
                'turnover_1y_std': turnover_1y_std,
                'low_quantile_turnover': low_quantile_turnover,
                'turnover_ratio_1w': turnover_ratio_1w,
                'turnover_ratio_1m': turnover_ratio_1m,
                'turnover_acceleration': turnover_acceleration,
                'turnover_quantile': turnover_quantile,
                'is_price_turnover_divergence': is_divergence,
            })
            print(f"已成功计算 {fund_code} 的超额收益和流动性指标。")

        except Exception as e:
            print(f"处理 {fund_code} 时发生错误: {e}")

    df_results = pd.DataFrame(results_list)
    df_funds_with_metrics = pd.merge(df_funds, df_results, on='ts_code', how='left') 

    print("\n--- 5. 计算行业内部相对成交额占比 ---")
    df_industry_turnover = df_funds_with_metrics.groupby('industry')['turnover_ratio_1w'].transform('sum')
    df_funds_with_metrics['turnover_pct_in_industry'] = df_funds_with_metrics['turnover_ratio_1w'] / df_industry_turnover

    
    
    # 计算行业内成交额占比的百分位数
    df_funds_with_metrics['turnover_pct_in_industry_quantile'] = df_funds_with_metrics['turnover_pct_in_industry'].rank(pct=True)

    # 剔除未能成功计算指标的基金
    df_funds_with_metrics.dropna(subset=['tracking_error'], inplace=True)

    output_filename = 'etf_metrics_daily_report.csv'
    df_funds_with_metrics.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n最终包含所有指标的基金列表已保存到 {output_filename} 文件中。")
    
    return df_funds_with_metrics

if __name__ == '__main__':
    run_data_fetcher()



