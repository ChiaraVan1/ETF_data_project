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

token = os.environ.get('TUSHARE_TOKEN')
if token:
    ts.set_token(token)
    pro = ts.pro_api()
else:
    raise ValueError("TUSHARE_TOKEN is not set in environment variables.")

# --- 1. 获取并筛选目标行业ETF ---
print("正在获取所有基金列表...")
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

# 统一转换日期格式，以备后续比较
df_lean['list_date'] = pd.to_datetime(df_lean['list_date'], errors='coerce')

# 定义清洗函数
def clean_benchmark_name(name):
    if pd.isna(name):
        return None
    cleaned_name = re.sub(r'\*.*', '', name)
    cleaned_name = cleaned_name.replace('收益率', '')
    return cleaned_name.strip()

# 应用清洗函数
df_lean['cleaned_benchmark_name'] = df_lean['benchmark'].apply(clean_benchmark_name)

# --- 2. 匹配基金与其基准指数 ---
print("\n正在分步获取Tushare指数基本信息...")
df_csi = pro.index_basic(market='CSI')
df_sse = pro.index_basic(market='SSE')
df_szse = pro.index_basic(market='SZSE')
df_index_basic = pd.concat([df_csi, df_sse, df_szse], ignore_index=True)
print("Tushare指数基本信息获取完毕。")
# 将指数的上市日期也转换为日期格式
df_index_basic['list_date'] = pd.to_datetime(df_index_basic['list_date'], errors='coerce')

# 构建更鲁棒的映射函数
def robust_map_name_to_code(row, df_index_basic):
    cleaned_fund_name = row['cleaned_benchmark_name']
    fund_list_date = row['list_date']
    
    # 手动映射，解决一些特殊情况
    manual_map = {
        '沪深300指数': '000300.SH',
        '创业板指数': '399006.SZ',
        '中证消费电子主题指数': '931104.CSI',
        '上证科创板芯片指数': '000685.SH',
        # 你可以在这里添加更多已知映射
    }
    if cleaned_fund_name in manual_map:
        return manual_map[cleaned_fund_name]
    
    # 遍历 df_index_basic，进行模糊匹配
    for index, idx_row in df_index_basic.iterrows():
        # 如果基金的基准名称中包含Tushare的简称，并且基金上市日期不早于指数上市日期，则匹配成功
        if pd.notna(cleaned_fund_name) and pd.notna(idx_row['name']) and idx_row['name'] in cleaned_fund_name and \
           pd.notna(fund_list_date) and pd.notna(idx_row['list_date']) and fund_list_date >= idx_row['list_date']:
            return idx_row['ts_code']
        
    return None

# 应用映射函数
# 使用 apply(axis=1) 来传递整行数据给函数
df_lean['benchmark_code'] = df_lean.apply(robust_map_name_to_code, axis=1, args=(df_index_basic,))

# --- 3. 筛选并保存最终的基金列表 ---
unmatched_funds = df_lean[df_lean['benchmark_code'].isnull()]
print("\n数据清洗和映射后的结果:")
print(df_lean.head())
print(f"\n成功映射的基金数量: {len(df_lean) - len(unmatched_funds)} 只")

if not unmatched_funds.empty:
    print(f"未能匹配的基金数量: {len(unmatched_funds)} 只")
    print("\n未能自动匹配的基金基准名称:")
    print(unmatched_funds[['ts_code', 'name', 'benchmark']])

df_lean.to_csv('cleaned_funds_list.csv', index=False, encoding='utf-8-sig')
print("\n清洗后的基金列表已成功保存到 cleaned_funds_list.csv 文件中。")

df_funds = df_lean.copy()

# 进一步筛选基金，剔除不符合监控标准的基金，以提高数据质量和分析效率。
# 筛选标准：
# 1. 必须有对应的基准指数代码，否则无法计算跟踪误差。
# 2. 基金规模不能过小，以保证流动性和代表性。
df_funds.dropna(subset=['benchmark_code'], inplace=True)
# Tushare 的 issue_amount 单位是“亿元”，这里设定筛选门槛为2亿元
min_issue_amount = 2.0  
df_funds = df_funds[df_funds['issue_amount'] >= min_issue_amount]

print(f"应用规模筛选后，基金数量: {len(df_funds)} 只")
df_funds.to_csv('final_filtered_funds.csv', index=False, encoding='utf-8-sig')
print("最终筛选后的基金列表已保存到 final_filtered_funds.csv 文件中。")

df_filtered_funds = df_funds.copy()

# --- 4. 批量下载历史行情数据 ---
end_date = datetime.now().strftime('%Y%m%d')
start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y%m%d')

# 获取所有需要下载数据的代码列表（基金和指数）
fund_codes = df_filtered_funds['ts_code'].tolist()
benchmark_codes = df_filtered_funds['benchmark_code'].tolist()
all_codes_to_download = list(set(fund_codes + benchmark_codes))

all_daily_data = []

print(f"正在获取 {len(all_codes_to_download)} 个基金和指数的历史行情数据...")

for code in all_codes_to_download:
    try:
        if code in fund_codes:
            # 获取基金历史日线数据
            df_daily = pro.fund_daily(ts_code=code, start_date=start_date, end_date=end_date)
            # 添加'asset_type'列，标记为'fund'
            df_daily['asset_type'] = 'fund'
        elif code in benchmark_codes:
            # 获取指数历史日线数据
            df_daily = pro.index_daily(ts_code=code, start_date=start_date, end_date=end_date)
            # 添加'asset_type'列，标记为'index'
            df_daily['asset_type'] = 'index'
        
        if df_daily is not None and not df_daily.empty:
            all_daily_data.append(df_daily)
            print(f"已成功获取 {code} 的数据。")
        else:
            print(f"警告: 未能获取 {code} 的数据，可能数据不存在或Tushare返回空。")

    except Exception as e:
        print(f"获取 {code} 数据时发生错误: {e}")

# 将所有数据合并到一个 DataFrame
df_all_daily_data = pd.concat(all_daily_data, ignore_index=True)

# 只保留我们需要的列
# df_all_daily_data = df_all_daily_data[['ts_code', 'trade_date', 'close', 'asset_type']]

# 将数据保存到本地
# df_all_daily_data.to_csv('historical_data.csv', index=False, encoding='utf-8-sig')
# print("\n所有历史行情数据已成功保存到 historical_data.csv 文件中。")

