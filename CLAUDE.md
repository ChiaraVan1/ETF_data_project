# ETF_data_project

双职责仓库：
1. **为 equity-risk-premium-monitor 提供每日 ETF 执行质量数据**——`simple_etf_metrics.py` 每日生成 `simple_etf_metrics.csv` 并发布到 GitHub Release `latest` tag，供下游 repo 实时下载。
2. **宽市场 ETF 筛选系统**——`ETF.py` + `ETF_screened.py` 对全市场消费/科技/医疗/宽基 ETF 进行打分筛选，结果上传到阿里云 OSS。

---

## 文件结构

```
ETF_data_project/
├── .github/workflows/
│   ├── test.yaml           # ★ 关键：每日 UTC 10:00 运行 simple_etf_metrics.py，发布 latest Release
│   ├── update_data.yml     # 每日 UTC 9:10 运行 ETF.py + ETF_screened.py + 上传 OSS + 发带日期 Release
│   ├── run_strategy.yml    # 手动/push 触发，从 Release 下载数据跑 ETF_screened.py
│   ├── test_pe.yml         # 手动触发，一次性探索脚本（东方财富 PE 接口测试）
│   └── requirements.txt   # pandas / tushare
│
├── simple_etf_metrics.py  # ★ 关键：为 equity-risk-premium-monitor 生成执行质量数据
├── ETF.py                 # 宽市场 ETF 数据获取 + 指标计算
├── ETF_screened.py        # 基于 ETF.py 输出的 CSV 做策略筛选，打买卖标签
├── upload_to_oss.py       # 将 CSV 报告上传到阿里云 OSS
└── README.md              # 核心指标体系说明（字段文档）
```

---

## 脚本职责详解

### `simple_etf_metrics.py`（最重要）

**用途**：为 equity-risk-premium-monitor 的 `etf_metrics.py` 提供数据源。

计算 18 只固定 ETF（与 equity-risk-premium-monitor 的 `ERP_TO_ETF` 映射一一对应）的执行质量指标：
- 折溢价率（latest_discount_rate）及 1年/3年分位
- 折溢价 5日/10日变化
- 换手率分位（52周周度分布）
- 价格/换手背离信号
- 年化波动率 + 1年分位
- 最大回撤 + 1年分位
- 超额收益均值、跟踪误差（相对基准指数）
- 超额收益 5/10/15/20日 MA + MA趋势斜率

输出：`simple_etf_metrics.csv`（以 `ts_code` 为 index）

**硬编码的 18 只 ETF**（`ETF_LIST`）：

| ERP代码 | ETF代码 | 说明 |
|--------|--------|------|
| 000300 | 510300.SH | 沪深300 |
| 000688 | 588000.SH | 科创50 |
| 000922 | 515180.SH | 中证红利 |
| 399989 | 512170.SH | 中证医疗 |
| 931071 | 515980.SH | 人工智能 |
| HSTECH | 513180.SH | 恒生科技 |
| SPY    | 513500.SH | 标普500 |
| QQQ    | 159696.SZ | 纳斯达克100 |
| EWQ    | 513080.SH | 法国 |
| EWJ    | 513880.SH | 日本 |
| 000069 | 510150.SH | 消费80 |
| 930781 | 516620.SH | 中证影视 |
| 000989 | 159936.SZ | 全指可选 |
| 931139 | 515650.SH | CS消费50 |
| 399967 | 512660.SH | 中证军工 |
| 931066 | 512710.SH | 军工龙头 |
| 930598 | 516150.SH | 稀土产业 |
| 930794 | 009225.OF | 中美互联网 |

> **注意**：新增指数时，equity-risk-premium-monitor 和本文件的 `ETF_LIST` 必须同步更新，否则下游无法获取该 ETF 的执行质量数据。

---

### `ETF.py`（宽市场筛选，第一步）

- 从 Tushare 拉取全部场内（E）+ 场外（O）基金基本信息
- 按关键词筛选消费/科技/医疗/宽基 ETF
- 匹配每只 ETF 对应的基准指数代码（含 manual_map 手动映射）
- 计算 3 年历史数据下的完整指标（超额收益、波动、回撤、折溢价、换手等）
- 输出：`etf_metrics_daily_report.csv`（发布到带日期的 GitHub Release + 上传 OSS）

### `ETF_screened.py`（策略筛选，第二步）

- 加载 `etf_metrics_daily_report.csv`
- 按三种策略打标签：
  - **买入机会**：安全条件（规模/流动性/折溢价/波动/回撤分位全过）+ 动态条件之一（资金加速/分位高/超额趋势改善等）
  - **低买**：折价率历史 P20 以下 + 折价快速加深 + 成交冰点
  - **卖出警示**：原"买入机会"但出现溢价/资金撤退/风险恶化信号
- 输出：`etf_screener_final_report.csv`

> **已知问题**：`ETF.py` 中 `turnover_acceleration` 计算为 `本周/本月`（月初时分母≈本周，导致加速度虚高）。`ETF_screened.py` 的 `safe_acceleration()` 已改为 `本周/(本月/4)`，是正确算法，但 `etf_metrics_daily_report.csv` 中的原始字段仍是旧口径。

### `upload_to_oss.py`

上传 `etf_metrics_daily_report.csv` 和 `etf_screener_final_report.csv` 到阿里云 OSS（bucket/endpoint 均从环境变量读取）。

---

## GitHub Actions Workflows

### `test.yaml` ★ 最关键

| 字段 | 值 |
|-----|----|
| 触发 | 每日 UTC 10:00（北京时间 18:00）；push to main；手动 |
| 运行脚本 | `simple_etf_metrics.py` |
| 输出 | `simple_etf_metrics.csv`，`etf_price.csv` |
| 发布方式 | `softprops/action-gh-release`，tag=`latest`，`overwrite: true`（始终覆盖） |
| 下游依赖 | equity-risk-premium-monitor 在 UTC 10:30 下载此 Release |

### `update_data.yml`

| 字段 | 值 |
|-----|----|
| 触发 | 每日 UTC 9:10（北京时间 17:10）；push 时含 ETF.py/ETF_screened.py/upload_to_oss.py；手动 |
| 运行脚本 | `ETF.py` → `ETF_screened.py` → `upload_to_oss.py` |
| 输出 | 上传 OSS；创建带日期时间戳的 GitHub Release |

### `run_strategy.yml`

手动或 push ETF_screened.py 时触发，从最新 Release 下载数据，跑策略 + 上传 OSS + 跑 `video_data_generator.py`。

> ⚠️ `video_data_generator.py` 在本地目录中不存在，该 workflow 运行时会失败。

### `test_pe.yml`

仅手动触发，内嵌 Python 脚本探索东方财富 PE 接口，不产出任何文件，是一次性实验性脚本。

---

## 跨 Repo 依赖关系

```
本 Repo（ETF_data_project）
  test.yaml 每日 UTC 10:00
  → simple_etf_metrics.py（TUSHARE_TOKEN）
  → 生成 simple_etf_metrics.csv
  → 发布到 GitHub Release tag=latest

下游（equity-risk-premium-monitor）
  daily_trade.yml 每日 UTC 10:30
  → etf_metrics.py
  → 下载 https://github.com/ChiaraVan1/ETF_data_project/releases/download/latest/simple_etf_metrics.csv
  → 补充 ETF 折溢价/换手/波动/超额收益信号到报告
```

时序窗口：ETF_data_project 比下游早 30 分钟完成，保证下游拿到当日最新数据。

---

## 环境变量清单

### GitHub Secrets

| 变量名 | 用途 | 使用脚本 |
|-------|------|---------|
| `TUSHARE_TOKEN` | Tushare Pro API token，拉取 A 股 ETF 行情/净值/指数数据 | `ETF.py`、`simple_etf_metrics.py` |
| `ALIYUN_ACCESS_KEY_ID` | 阿里云 OSS Access Key ID | `upload_to_oss.py` |
| `ALIYUN_ACCESS_KEY_SECRET` | 阿里云 OSS Access Key Secret | `upload_to_oss.py` |
| `OSS_ENDPOINT` | 阿里云 OSS Endpoint（如 `oss-cn-hangzhou.aliyuncs.com`） | `upload_to_oss.py` |
| `OSS_BUCKET_NAME` | 阿里云 OSS Bucket 名称 | `upload_to_oss.py` |
| `GITHUB_TOKEN` | GitHub Actions 内置，用于创建 Release 和上传 Release Asset | 所有 workflow |

---

## 添加新 ETF 的操作步骤

1. 在 `simple_etf_metrics.py` 的 `ETF_LIST` 中追加 `("ERP代码", "ts_code")` 条目。
2. 在 `ETF_TO_BENCHMARK` 中追加对应的基准指数代码（若无 A 股基准则填 `None`）。
3. 同步更新 equity-risk-premium-monitor 的 `etf_metrics.py` 中的 `ERP_TO_ETF` 映射。
4. 触发 `test.yaml`（手动或等次日自动），验证新 ETF 出现在发布的 CSV 中。
