# STATUS.md — 跨 session 运行记忆

> 每次 Claude 完成工作后必须更新此文件。
> 最后更新：2026-06-08

---

## 当前运行状态

| 模块 | 状态 | 备注 |
|-----|------|------|
| `test.yaml`（simple_etf_metrics 每日发布） | 正常 | 每日 UTC 10:00，发布到 Release `latest` tag |
| `update_data.yml`（宽市场 ETF 筛选） | 正常 | 每日 UTC 9:10，上传 OSS + 带日期 Release |
| `run_strategy.yml` | ⚠️ 部分失败 | `video_data_generator.py` 不存在，会在该步骤出错 |
| `test_pe.yml` | 仅手动，不常用 | 一次性探索脚本 |
| 覆盖 ETF 数量 | 18 只 | `simple_etf_metrics.py` 中硬编码，与 equity-risk-premium-monitor 对齐 |
| 阿里云 OSS 上传 | 正常 | `etf_metrics_daily_report.csv` + `etf_screener_final_report.csv` |

---

## 已知问题

1. **`video_data_generator.py` 缺失**：`run_strategy.yml` 的最后一步会调用 `video_data_generator.py`，但该文件不在仓库中，导致该 workflow 每次运行到这步时失败。若不需要视频数据，应从 workflow 中删除该步骤。

2. **`turnover_acceleration` 口径不一致**：
   - `ETF.py` 写入 CSV 的 `turnover_acceleration` = `本周成交额 / 本月成交额`（月初时月成交额≈本周，导致数值虚高）
   - `ETF_screened.py` 的 `safe_acceleration()` 已改为 `本周 / (本月/4)`（正确口径）
   - `simple_etf_metrics.py` 也已用 `过去4周均值` 作分母（正确）
   - 但 `etf_metrics_daily_report.csv` 中的原始列仍是旧口径，若外部直接使用该列会有偏差

3. **ETF 列表可能与 equity-risk-premium-monitor 不同步**：`simple_etf_metrics.py` 的 `ETF_LIST` 和 equity-risk-premium-monitor 的 `ERP_TO_ETF` 需要手动保持同步，目前无自动校验机制。

4. **`requirements.txt` 不完整**：`.github/workflows/requirements.txt` 只有 `pandas` 和 `tushare`。`ETF.py` 需要 `numpy`（通常随 tushare 安装）；`upload_to_oss.py` 需要 `oss2`（在 workflow 中单独 `pip install oss2`）。

5. **两套 Release 发布机制共存**：
   - `test.yaml` 发布 `latest` tag（供 equity-risk-premium-monitor 下载，始终覆盖）
   - `update_data.yml` 每日发布带时间戳的新 Release tag（如 `v202506081030`），长期运行会累积大量 Release

---

## 下一步要做的事

- [ ] 删除或修复 `run_strategy.yml` 中对 `video_data_generator.py` 的调用（该文件不存在）。
- [ ] 评估是否需要清理历史 Release（`update_data.yml` 每日产生一个带时间戳的 Release）。
- [ ] 考虑将 `ETF_LIST` 同步校验逻辑加入 CI，在与 equity-risk-premium-monitor 不一致时打印警告。
- [ ] 修复 `ETF.py` 中的 `turnover_acceleration` 计算，与 `ETF_screened.py` 的 `safe_acceleration()` 对齐。

---

## 变更日志

| 日期 | 变更内容 |
|-----|---------|
| 2026-06-08 | 初始化 CLAUDE.md / STATUS.md，读取并记录全项目结构和已知问题 |
