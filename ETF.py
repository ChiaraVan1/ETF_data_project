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

def initialize_tushare():
    """
    初始化Tushare API，从环境变量中读取Token。
    """
    token = os.environ.get('TUSHARE_TOKEN')
    if token:
        ts.set_token(token)
        return ts.pro_api()
    else:
        raise ValueError("TUSHARE_TOKEN is not set in environment variables.")

def fetch_and_filter_funds(pro):
    """
    获取并筛选消费、科技、医疗行业的ETF。
    """
    print("--- 1. 获取并筛选目标行业ETF ---")
    df_etf_list = pro.fund_basic(market='E')
    df_offshore_list = pro.fund_basic(market='O')
    df_all_funds = pd.concat([df_etf_list, df_offshore_list], ignore_index=True)
    print(f"合并后，总共获取到 {len(df_all_funds)} 只基金的信息。")

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
    
    return df_lean

def map_funds_to_indices(pro, df_lean):
    """
    匹配基金与其基准指数的代码。
    """
    print("\n--- 2. 匹配基金与其基准指数 ---")
    df_csi = pro.index_basic(market='CSI')
    df_sse = pro.index_basic(market='SSE')
    df_szse = pro.index_basic(market='SZSE')
    df_index_basic_map = pd.concat([df_csi, df_sse, df_szse], ignore_index=True)
    df_index_basic_map['list_date'] = pd.to_datetime(df_index_basic_map['list_date'], errors='coerce')

    def robust_map_name_to_code(row, df_index_basic):
        cleaned_fund_name = row['cleaned_benchmark_name']
        fund_list_date = row['list_date']
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

    df_lean['benchmark_code'] = df_lean.apply(robust_map_name_to_code, axis=1, args=(df_index_basic_map,))
    
    return df_lean

def calculate_metrics(pro, df_funds):
    """
    计算所有超额收益、流动性、折价率和风险指标。
    """
    print("\n--- 3. 计算所有超额收益、流动性、折价率和风险指标 ---")
    results_list = []
    today = datetime.now().date()
    one_year_ago = (today - timedelta(days=365)).strftime('%Y%m%d')
    three_years_ago = (today - timedelta(days=3 * 365)).strftime('%Y%m%d')

    for index, row in df_funds.iterrows():
        fund_code = row['ts_code']
        benchmark_code = row['benchmark_code']
        aum = row['issue_amount']
        
        data_status = 'ok'
        print(f"\n处理基金 {fund_code} ({row['name']})...")

        # 初始化所有指标为NaN
        metrics = {
            'ts_code': fund_code,
            'excess_return_mean': np.nan, 'tracking_error': np.nan,
            'excess_return_5d_ma': np.nan, 'excess_return_10d_ma': np.nan,
            'excess_return_15d_ma': np.nan, 'excess_return_20d_ma': np.nan,
            'turnover_1y_mean': np.nan, 'turnover_rate': np.nan,
            'turnover_6m_vs_3y': np.nan, 'turnover_1y_std': np.nan,
            'low_quantile_turnover': np.nan, 'turnover_ratio_1w': np.nan,
            'turnover_ratio_1m': np.nan, 'turnover_acceleration': np.nan,
            'turnover_quantile': np.nan, 'is_price_turnover_divergence': np.nan,
            'annualized_volatility': np.nan, 'max_drawdown': np.nan,
            'latest_discount_rate': np.nan, 'discount_quantile_1y': np.nan,
            'discount_quantile_3y': np.nan, 'change_5d_discount': np.nan,
            'change_10d_discount': np.nan,
        }

        try:
            # --- 核心数据获取（健壮性处理） ---
            df_fund_daily = pro.fund_daily(ts_code=fund_code, start_date=three_years_ago)
            if df_fund_daily.empty:
                print(f"  警告: 未能获取 {fund_code} 的日行情数据，将跳过部分指标计算。")
                data_status += ';fund_daily_missing'
            
            df_fund_nav = pro.fund_nav(ts_code=fund_code, start_date=three_years_ago)
            if df_fund_nav.empty:
                print(f"  警告: 未能获取 {fund_code} 的净值数据，将跳过折价率和最大回撤计算。")
                data_status += ';fund_nav_missing'
            
            df_index_daily = pro.index_daily(ts_code=benchmark_code, start_date=three_years_ago)
            if df_index_daily.empty:
                print(f"  警告: 未能获取基准 {benchmark_code} 的日行情数据，将跳过超额收益和跟踪误差计算。")
                data_status += ';index_daily_missing'
            
            # 数据预处理与合并
            merged_data = pd.DataFrame()
            if not df_fund_daily.empty:
                df_fund_daily.rename(columns={'close': 'close_fund', 'pct_chg': 'pct_chg_fund'}, inplace=True)
                df_fund_daily['trade_date'] = pd.to_datetime(df_fund_daily['trade_date'])
                df_fund_daily.set_index('trade_date', inplace=True)
                df_fund_daily.sort_index(inplace=True)
                merged_data = df_fund_daily.copy()
            
            if not df_fund_nav.empty:
                df_fund_nav['nav_date'] = pd.to_datetime(df_fund_nav['nav_date'])
                df_fund_nav.set_index('nav_date', inplace=True)
                df_fund_nav.sort_index(inplace=True)
                if not merged_data.empty:
                    merged_data = pd.merge(merged_data, df_fund_nav[['unit_nav']], left_index=True, right_index=True, how='left')
            
            if not df_index_daily.empty:
                df_index_daily.rename(columns={'close': 'close_index', 'pct_chg': 'pct_chg_index'}, inplace=True)
                df_index_daily['trade_date'] = pd.to_datetime(df_index_daily['trade_date'])
                df_index_daily.set_index('trade_date', inplace=True)
                df_index_daily.sort_index(inplace=True)
                if not merged_data.empty:
                    merged_data = pd.merge(merged_data, df_index_daily[['pct_chg_index']], left_index=True, right_index=True, how='left')
            
            # --- 计算指标（仅在数据可用时）---
            
            # 超额收益和跟踪误差 (依赖 df_fund_daily 和 df_index_daily)
            if 'pct_chg_fund' in merged_data.columns and 'pct_chg_index' in merged_data.columns and len(merged_data.dropna(subset=['pct_chg_fund', 'pct_chg_index'])) > 20:
                merged_data['excess_return'] = merged_data['pct_chg_fund'] - merged_data['pct_chg_index']
                df_excess_3y = merged_data[merged_data.index >= three_years_ago]
                metrics['excess_return_mean'] = df_excess_3y['excess_return'].mean()
                metrics['tracking_error'] = df_excess_3y['excess_return'].std() * np.sqrt(250)
                metrics['excess_return_5d_ma'] = merged_data['excess_return'].rolling(window=5).mean().iloc[-1]
                metrics['excess_return_10d_ma'] = merged_data['excess_return'].rolling(window=10).mean().iloc[-1]
                metrics['excess_return_15d_ma'] = merged_data['excess_return'].rolling(window=15).mean().iloc[-1]
                metrics['excess_return_20d_ma'] = merged_data['excess_return'].rolling(window=20).mean().iloc[-1]

            # 收益波动率 (依赖 df_fund_daily)
            if 'pct_chg_fund' in merged_data.columns:
                metrics['annualized_volatility'] = merged_data['pct_chg_fund'].std() * np.sqrt(250)
            
            # 最大回撤 (依赖 df_fund_daily)
            if 'pct_chg_fund' in merged_data.columns:
                merged_data['cum_close'] = (1 + merged_data['pct_chg_fund'] / 100).cumprod()
                merged_data['max_close'] = merged_data['cum_close'].cummax()
                merged_data['drawdown'] = (merged_data['max_close'] - merged_data['cum_close']) / merged_data['max_close']
                metrics['max_drawdown'] = merged_data['drawdown'].max()
            
            # 折价率及变化 (依赖 df_fund_daily 和 df_fund_nav)
            if 'unit_nav' in merged_data.columns and 'close_fund' in merged_data.columns:
                merged_data['discount_rate'] = (merged_data['unit_nav'] - merged_data['close_fund']) / merged_data['unit_nav']
                metrics['latest_discount_rate'] = merged_data['discount_rate'].iloc[-1] if not merged_data.empty else np.nan
                
                df_discount_1y = merged_data['discount_rate'][merged_data.index >= one_year_ago].dropna()
                metrics['discount_quantile_1y'] = df_discount_1y.rank(pct=True).iloc[-1] if len(df_discount_1y) > 1 else np.nan
                df_discount_3y = merged_data['discount_rate'][merged_data.index >= three_years_ago].dropna()
                metrics['discount_quantile_3y'] = df_discount_3y.rank(pct=True).iloc[-1] if len(df_discount_3y) > 1 else np.nan

                if len(merged_data) > 5:
                    metrics['change_5d_discount'] = merged_data['discount_rate'].iloc[-1] - merged_data['discount_rate'].iloc[-6]
                if len(merged_data) > 10:
                    metrics['change_10d_discount'] = merged_data['discount_rate'].iloc[-1] - merged_data['discount_rate'].iloc[-11]

            # 流动性与情绪指标 (依赖 df_fund_daily)
            if 'amount' in merged_data.columns and 'close_fund' in merged_data.columns:
                df_liquidity_1y = merged_data[merged_data.index >= one_year_ago]
                df_liquidity_3y = merged_data[merged_data.index >= three_years_ago]
                df_liquidity_6m = df_liquidity_1y[df_liquidity_1y.index >= (today - timedelta(days=180)).strftime('%Y%m%d')]
                
                metrics['turnover_1y_mean'] = df_liquidity_1y['amount'].mean()
                metrics['turnover_rate'] = metrics['turnover_1y_mean'] / (aum * 100000) if aum > 0 else np.nan
                metrics['turnover_6m_mean'] = df_liquidity_6m['amount'].mean()
                metrics['turnover_3y_mean'] = df_liquidity_3y['amount'].mean()
                metrics['turnover_6m_vs_3y'] = metrics['turnover_6m_mean'] / metrics['turnover_3y_mean'] if metrics['turnover_3y_mean'] > 0 else np.nan
                metrics['turnover_1y_std'] = df_liquidity_1y['amount'].std()
                metrics['low_quantile_turnover'] = df_liquidity_1y['amount'].quantile(0.05)
                
                aum_thousands = aum * 100000
                df_fund_weekly = merged_data.resample('W')['amount'].sum().to_frame('amount')
                df_fund_monthly = merged_data.resample('ME')['amount'].sum().to_frame('amount')
                latest_week_turnover = df_fund_weekly['amount'].iloc[-1] if not df_fund_weekly.empty else 0
                latest_month_turnover = df_fund_monthly['amount'].iloc[-1] if not df_fund_monthly.empty else 0
                metrics['turnover_ratio_1w'] = latest_week_turnover / aum_thousands if aum_thousands > 0 else np.nan
                metrics['turnover_ratio_1m'] = latest_month_turnover / aum_thousands if aum_thousands > 0 else np.nan
                metrics['turnover_acceleration'] = latest_week_turnover / latest_month_turnover if latest_month_turnover > 0 else np.nan
                metrics['turnover_quantile'] = np.nan
                if not df_fund_weekly.empty and len(df_fund_weekly) >= 52:
                    df_weekly_12m = df_fund_weekly.iloc[-52:]
                    metrics['turnover_quantile'] = df_weekly_12m['amount'].rank(pct=True).iloc[-1]
                
                is_divergence = False
                price_change_1w = 0
                df_price_1w = merged_data.tail(5)
                if len(df_price_1w) > 1:
                    price_change_1w = (df_price_1w['close_fund'].iloc[-1] - df_price_1w['close_fund'].iloc[0]) / df_price_1w['close_fund'].iloc[0]
                turnover_change_1w = 0
                if len(df_fund_weekly) > 1:
                    turnover_change_1w = latest_week_turnover - df_fund_weekly['amount'].iloc[-2]
                if np.sign(price_change_1w) != np.sign(turnover_change_1w):
                    is_divergence = True
                metrics['is_price_turnover_divergence'] = is_divergence

            # 存储结果并添加状态信息
            metrics['data_status'] = data_status.lstrip(';')
            results_list.append(metrics)
            print(f"已成功计算 {fund_code} 的所有可用指标。")

        except Exception as e:
            print(f"处理 {fund_code} 时发生未知错误: {e}")
            results_list.append({'ts_code': fund_code, 'data_status': f'error: {e}'})
            
    return pd.DataFrame(results_list)

def post_process_data(df_funds_with_metrics):
    """
    计算行业内部相对成交额占比并进行最终处理。
    """
    print("\n--- 4. 计算行业内部相对成交额占比 ---")
    df_industry_turnover = df_funds_with_metrics.groupby('industry')['turnover_ratio_1w'].transform('sum')
    df_funds_with_metrics['turnover_pct_in_industry'] = df_funds_with_metrics['turnover_ratio_1w'] / df_industry_turnover
    df_funds_with_metrics['turnover_pct_in_industry_quantile'] = df_funds_with_metrics['turnover_pct_in_industry'].rank(pct=True)

    df_funds_with_metrics.dropna(subset=['excess_return_mean', 'tracking_error', 'annualized_volatility'], how='all', inplace=True)
    
    return df_funds_with_metrics

def main():
    """
    主函数，执行整个数据获取和处理流程。
    """
    pro = initialize_tushare()
    df_lean = fetch_and_filter_funds(pro)
    df_funds = map_funds_to_indices(pro, df_lean)

    print("\n--- 3. 筛选并保存最终的基金列表 ---")
    df_funds.to_csv('etf_funds_list_raw.csv', index=False, encoding='utf-8-sig')
    df_funds.dropna(subset=['benchmark_code'], inplace=True)
    min_issue_amount = 2.0  # Tushare 的 issue_amount 单位是“亿元”
    df_funds = df_funds[df_funds['issue_amount'] >= min_issue_amount]
    df_funds.to_csv('etf_funds_list_final.csv', index=False, encoding='utf-8-sig')
    print(f"最终筛选后，可用于计算的基金数量: {len(df_funds)} 只")
    
    unmatched_funds = df_lean[df_lean['benchmark_code'].isnull()]
    if not unmatched_funds.empty:
        print(f"\n未能自动匹配的基金数量: {len(unmatched_funds)} 只")
        
    df_results = calculate_metrics(pro, df_funds)
    df_funds_with_metrics = pd.merge(df_funds, df_results, on='ts_code', how='left')
    df_final_report = post_process_data(df_funds_with_metrics)

    output_filename = 'etf_metrics_daily_report.csv'
    df_final_report.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n最终包含所有指标的基金列表已保存到 {output_filename} 文件中。")

if __name__ == '__main__':
    main()
