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


def get_price_and_valuation_data(ts_code, benchmark_code, start_date, end_date, pro):
    """
    获取单个ETF的价格、净值、指数估值等数据并计算相关指标
    """
    # 1. 获取ETF日线行情数据
    df_daily = pro.fund_daily(ts_code=ts_code, start_date=start_date, end_date=end_date, fields='ts_code,trade_date,close,pre_close,pct_chg')
    df_daily['trade_date'] = pd.to_datetime(df_daily['trade_date'])
    df_daily.set_index('trade_date', inplace=True)
    
    # 2. 获取ETF单位净值数据
    df_nav = pro.fund_nav(ts_code=ts_code, start_date=start_date, end_date=end_date, fields='ts_code,nav_date,unit_nav,adj_nav')
    df_nav['nav_date'] = pd.to_datetime(df_nav['nav_date'])
    df_nav.set_index('nav_date', inplace=True)
    
    # 3. 合并数据
    df_merged = pd.merge(df_daily, df_nav, left_index=True, right_index=True, how='outer')
    df_merged.sort_index(inplace=True)
    
    # 4. 获取跟踪指数的估值数据 (PE/PB)
    df_index_basic = pro.index_dailybasic(ts_code=benchmark_code, start_date=start_date, end_date=end_date, fields='trade_date,pe,pe_ttm,pb')
    if df_index_basic.empty:
        print(f"警告: 未能获取指数 {benchmark_code} 的估值数据。")
    else:
        df_index_basic['trade_date'] = pd.to_datetime(df_index_basic['trade_date'])
        df_index_basic.set_index('trade_date', inplace=True)
        df_merged = pd.merge(df_merged, df_index_basic, left_index=True, right_index=True, how='left')
    
    # 5. 计算价格类指标
    df_merged['pct_chg_decimal'] = df_merged['pct_chg'] / 100
    df_merged['annual_volatility'] = df_merged['pct_chg_decimal'].rolling(window=252).std() * np.sqrt(252)
    df_merged['premium_rate'] = (df_merged['close'] - df_merged['unit_nav']) / df_merged['unit_nav']
    
    return df_merged


def calculate_percentiles(df, col_name, window=None):
    """
    计算历史分位点
    """
    if col_name not in df.columns or df[col_name].isnull().all():
        return np.nan
    
    series = df[col_name].dropna()
    if window:
        series = series.tail(window)
        if len(series) < window * 0.5: # 数据量不足，返回空值
            return np.nan
    
    current_value = series.iloc[-1]
    # 使用秩次百分比计算分位点
    percentile = (series.rank(pct=True).iloc[-1])
    return percentile

def run_data_fetcher():
    """
    执行整个数据获取和指标计算流程。
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

    condition_consumer = (df_all_funds[name_col].str.contains('|'.join(keywords_consumer), na=False)) |                          (df_all_funds[benchmark_col].str.contains('|'.join(keywords_consumer), na=False))
    condition_tech = (df_all_funds[name_col].str.contains('|'.join(keywords_tech), na=False)) |                      (df_all_funds[benchmark_col].str.contains('|'.join(keywords_tech), na=False))
    condition_healthcare = (df_all_funds[name_col].str.contains('|'.join(keywords_healthcare), na=False)) |                            (df_all_funds[benchmark_col].str.contains('|'.join(keywords_healthcare), na=False))

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

    print("--- 2. 匹配基金与其基准指数 ---")
    df_csi = pro.index_basic(market='CSI')
    df_sse = pro.index_basic(market='SSE')
    df_szse = pro.index_basic(market='SZSE')
    df_index_basic = pd.concat([df_csi, df_sse, df_szse], ignore_index=True)
    df_index_basic['list_date'] = pd.to_datetime(df_index_basic['list_date'], errors='coerce')

    def robust_map_name_to_code(row, df_index_basic):
        cleaned_fund_name = row['cleaned_benchmark_name']
        fund_list_date = row['list_date']
        
        manual_map = {
            '沪深300指数': '000300.SH',
            '创业板指数': '399006.SZ',
            '中证消费电子主题指数': '931104.CSI',
            '上证科创板芯片指数': '000685.SH',
            # 待添加 
        }
        if cleaned_fund_name in manual_map:
            return manual_map[cleaned_fund_name]
        
        for index, idx_row in df_index_basic.iterrows():
            if pd.notna(cleaned_fund_name) and pd.notna(idx_row['name']) and idx_row['name'] in cleaned_fund_name and                pd.notna(fund_list_date) and pd.notna(idx_row['list_date']) and fund_list_date >= idx_row['list_date']:
                return idx_row['ts_code']
        return None

    df_lean['benchmark_code'] = df_lean.apply(robust_map_name_to_code, axis=1, args=(df_index_basic,))

    print("
--- 3. 筛选并保存最终的基金列表 ---")
    df_funds = df_lean.copy()
    unmatched_funds = df_funds[df_funds['benchmark_code'].isnull()]
    df_funds.to_csv('etf_funds_list_raw.csv', index=False, encoding='utf-8-sig')
    df_funds.dropna(subset=['benchmark_code'], inplace=True)
    min_issue_amount = 2.0
    df_funds = df_funds[df_funds['issue_amount'] >= min_issue_amount]
    df_funds.to_csv('etf_funds_list_final.csv', index=False, encoding='utf-8-sig')
    print(f"最终筛选后，可用于计算的基金数量: {len(df_funds)} 只")
    
    if not unmatched_funds.empty:
        print(f"
未能自动匹配的基金数量: {len(unmatched_funds)} 只")

    print("
--- 4. 计算所有指标 ---")
    results_list = []
    today = datetime.now().date()
    one_year_ago = (today - timedelta(days=365)).strftime('%Y%m%d')
    three_years_ago = (today - timedelta(days=3 * 365)).strftime('%Y%m%d')

    for index, row in df_funds.iterrows():
        fund_code = row['ts_code']
        benchmark_code = row['benchmark_code']
        aum = row['issue_amount']

        try:
            # 获取基金和指数日行情数据
            df_fund_daily = pro.fund_daily(ts_code=fund_code, start_date=three_years_ago)
            df_index_daily = pro.index_daily(ts_code=benchmark_code, start_date=three_years_ago)
            if df_fund_daily.empty or df_index_daily.empty:
                print(f"警告: 未能获取 {fund_code} 或其基准 {benchmark_code} 的日线数据，跳过。")
                continue
            
            df_fund_daily['trade_date'] = pd.to_datetime(df_fund_daily['trade_date'])
            df_fund_daily.set_index('trade_date', inplace=True)
            df_fund_daily.sort_index(inplace=True)
            df_index_daily['trade_date'] = pd.to_datetime(df_index_daily['trade_date'])
            df_index_daily.set_index('trade_date', inplace=True)
            df_index_daily.sort_index(inplace=True)

            merged_data = pd.merge(df_fund_daily, df_index_daily, left_index=True, right_index=True, how='inner', suffixes=('_fund', '_index'))
            
            if merged_data.empty or len(merged_data) < 20:
                print(f"警告: {fund_code} 数据不足，跳过。")
                continue

            merged_data['excess_return'] = merged_data['pct_chg_fund'] - merged_data['pct_chg_index']
            
            df_excess_3y = merged_data[merged_data.index >= three_years_ago]
            excess_return_mean = df_excess_3y['excess_return'].mean()
            tracking_error = df_excess_3y['excess_return'].std() * np.sqrt(250)
            
            ma_5 = merged_data['excess_return'].rolling(window=5).mean().iloc[-1]
            ma_10 = merged_data['excess_return'].rolling(window=10).mean().iloc[-1]
            ma_15 = merged_data['excess_return'].rolling(window=15).mean().iloc[-1]
            ma_20 = merged_data['excess_return'].rolling(window=20).mean().iloc[-1]
            
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

            # ----------------- 价格和估值指标计算 -----------------
            df_full_data = get_price_and_valuation_data(fund_code, benchmark_code, three_years_ago, datetime.now().strftime('%Y%m%d'), pro)
            
            max_drawdown = np.nan
            if 'adj_nav' in df_full_data.columns and not df_full_data['adj_nav'].isnull().all():
                df_full_data['peak_nav'] = df_full_data['adj_nav'].cummax()
                df_full_data['drawdown'] = (df_full_data['adj_nav'] - df_full_data['peak_nav']) / df_full_data['peak_nav']
                max_drawdown = df_full_data['drawdown'].min()
                
            pe_percentile = calculate_percentiles(df_full_data, 'pe')
            pb_percentile = calculate_percentiles(df_full_data, 'pb')
            premium_rate_percentile = calculate_percentiles(df_full_data, 'premium_rate')
            
            annual_volatility = np.nan
            if 'annual_volatility' in df_full_data.columns and not df_full_data['annual_volatility'].isnull().all():
                annual_volatility = df_full_data['annual_volatility'].iloc[-1]
            
            # ----------------- 整合所有结果 -----------------
            results_list.append({
                'ts_code': fund_code,
                'name': row['name'],
                'industry': row['industry'],
                'invest_type': row['invest_type'],
                'benchmark': row['benchmark'],
                'benchmark_code': benchmark_code,
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
                'annual_volatility': annual_volatility,
                'max_drawdown': max_drawdown,
                'pe': df_full_data['pe'].iloc[-1] if 'pe' in df_full_data.columns and not df_full_data['pe'].isnull().all() else np.nan,
                'pb': df_full_data['pb'].iloc[-1] if 'pb' in df_full_data.columns and not df_full_data['pb'].isnull().all() else np.nan,
                'premium_rate': df_full_data['premium_rate'].iloc[-1] if 'premium_rate' in df_full_data.columns and not df_full_data['premium_rate'].isnull().all() else np.nan,
                'pe_percentile': pe_percentile,
                'pb_percentile': pb_percentile,
                'premium_rate_percentile': premium_rate_percentile,
            })
            print(f"已成功计算 {fund_code} 的所有指标。")

        except Exception as e:
            print(f"处理 {fund_code} 时发生错误: {e}")

    df_results = pd.DataFrame(results_list)
    
    # 将原始基金列表与计算结果合并
    df_funds_with_metrics = pd.merge(df_funds, df_results, on='ts_code', how='left', suffixes=('_original', '_metrics'))
    
    print("
--- 5. 计算行业内部相对成交额占比 ---")
    df_industry_turnover = df_funds_with_metrics.groupby('industry')['turnover_ratio_1w'].transform('sum')
    df_funds_with_metrics['turnover_pct_in_industry'] = df_funds_with_metrics['turnover_ratio_1w'] / df_industry_turnover
    df_funds_with_metrics['turnover_pct_in_industry_quantile'] = df_funds_with_metrics['turnover_pct_in_industry'].rank(pct=True)

    df_funds_with_metrics.dropna(subset=['tracking_error'], inplace=True)
    df_funds_with_metrics.drop_duplicates(subset=['ts_code'], inplace=True)

    output_filename = 'etf_metrics_daily_report.csv'
    df_funds_with_metrics.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"
最终包含所有指标的基金列表已保存到 {output_filename} 文件中。")
    
    return df_funds_with_metrics

if __name__ == '__main__':
    run_data_fetcher()


