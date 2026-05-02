# News-Aware PGAN 适配说明

## 1. 目标与论文对照检查

本次扩展遵循你的核心要求：

- 因子模型从 Carhart 4 维扩展为 News-Aware 9 维（4 Carhart + 5 正交新闻因子）。
- GAN 训练与推断主干保持不变。
- 用 `model_type` 控制兼容：`carhart` 与 `news_aware` 都可以跑。

与 Scholl 论文评价框架的对应关系：

- 保持原有评价指标层：组合重构、行为稳定性、frontier、counterfactual 等。
- 因子暴露相关指标在 `news_aware` 下使用 9 维暴露，在 `carhart` 下使用 4 维暴露。
- 这属于在论文框架上的扩展实现，不改变论文原 GAN 训练目标函数。

检查结论：

- `train` 阶段：不依赖 `model_type`，与原实现一致。
- `evaluate` 阶段：`model_type` 已接入并分流，支持 4/9 因子两种路径。
- `carhart` 路径兼容未破坏。
- `news_aware` 路径可构建并读取 9 维因子暴露。

## 2. 代码结构（对照 structure.md）

延续原结构，不改变主目录职责：

- 入口：`run_pipeline.py`
- 配置：`config/*.yaml`
- 训练：`src/training/*`
- 评估：`src/evaluation/*`
- 产物：`artifacts/` 与 `outputs/`

news-aware 增量主要落在评估层：

- `src/evaluation/evaluator.py`
	- 新增 `model_type` 分流（`carhart` / `news_aware`）。
	- 动态设置 `factor_columns`（4 或 9）。
- `src/evaluation/factor_exposures_news_aware.py`
	- 组合 Carhart rolling betas 与月度正交新闻因子，输出 9 维暴露。

训练主干保持原样：

- `src/training/trainer.py`
- `src/training/evaluation_exporter.py`

## 3. 配置改动（如何启用 news_aware）

核心配置在 `config/eval.yaml`：

1. `evaluation.model_type`

- `carhart`：4 因子
- `news_aware`：9 因子

2. `evaluation.inputs.news_factors`

- 指向月度正交新闻因子文件。
- 当前默认：`../../LLM news scoring/scores_monthly/news_factors_orthogonalized_2010_2024.csv`

3. `evaluation.factor_exposure_estimation.output_path_news_aware`

- news-aware 因子暴露缓存路径。

可选命令行覆盖（不改 yaml）：

- `run_pipeline.py --model-type carhart`
- `run_pipeline.py --model-type news_aware`

## 4. 数据加载、训练、validation、test 的文件流

### 4.1 训练输入（由 data.yaml 驱动）

来自 `config/data.yaml` placeholders：

- `raw/holdings.csv.zip`
- `raw/stock_characteristics.parquet`（或可解析等价格式）
- `raw/stock_returns.parquet`
- `raw/carhart_factors.parquet`
- 其他元数据文件

训练阶段主要流程：

1. `sample` 产出样本面板
2. `tensors` 构建模型输入索引
3. `train` 训练 GAN，并周期性做 validation

### 4.2 train 后结果在哪

按 `config/paths.yaml`：

- Checkpoints：`artifacts/checkpoints/*.pt`（含 `latest.pt`、`best.pt`、`epoch_xxx.pt`）
- 训练日志：`outputs/metrics/`
	- `train_steps.jsonl`
	- `val_epochs.jsonl`
	- `heartbeat.jsonl`
	- `metrics.csv`
- TensorBoard：`outputs/logs/tensorboard/`

### 4.3 validation 用哪些文件、结果在哪

validation 在 `train` 阶段内部完成：

- 数据切分来自 `config/data.yaml` 的 `split.val_*`。
- 指标写入：
	- `outputs/metrics/val_epochs.jsonl`
	- `outputs/metrics/metrics.csv`

### 4.4 test/evaluation 用哪些文件、结果在哪

评估输入来自两部分：

1. 训练后导出工件（`src/training/evaluation_exporter.py`）

- `artifacts/evaluation/portfolio_predictions.parquet`
- `artifacts/embeddings/strategy_embeddings.parquet`

2. 原始与辅助因子数据（`config/eval.yaml`）

- `raw/stock_returns.parquet`
- `raw/carhart_factors.parquet`
- `raw/stock_characteristics.parquet`（carhart路径可直接用）
- `news_factors`（news-aware 需要）

评估输出目录：`outputs/evaluation/`

- `portfolio_metrics_by_sample.parquet`
- `portfolio_metrics_summary.csv`
- `representation_metrics.json`
- `representation_per_class.csv`
- `stability_by_fund.parquet`
- `stability_summary.csv`
- `frontier_metrics_by_sample.parquet`
- `frontier_summary.csv`
- `counterfactual_metrics_by_case.parquet`
- `counterfactual_summary.csv`
- `skipped_metrics.json`
- `evaluation_report.md`

## 5. run_pipeline 的推荐运行方式

项目根目录下执行：

### 5.1 原 4 因子（Carhart）

```bash
PYTHONPATH=. python run_pipeline.py --stage train --project-root . --model-type carhart
PYTHONPATH=. python run_pipeline.py --stage evaluate --project-root . --model-type carhart
```

### 5.2 News-Aware 9 因子

```bash
PYTHONPATH=. python run_pipeline.py --stage train --project-root . --model-type news_aware
PYTHONPATH=. python run_pipeline.py --stage evaluate --project-root . --model-type news_aware
```

说明：

- `train` 阶段本身不改变 GAN 结构，`model_type` 主要影响 `evaluate` 阶段的因子暴露与行为指标计算。
- 如果只想切换评估，可直接只跑 `--stage evaluate`。

## 6. instruction.md 对齐说明

对齐点：

- 配置驱动：已满足（`model_type`、路径、阈值均由 config 控制，可 CLI 覆盖）。
- graceful degradation：已保留（缺输入会在 `skipped_metrics.json` 记录而非整体崩溃，除非配置要求）。
- 输出工件：与 instruction 列表一致，继续写入 `outputs/evaluation/`。
- EXACT/CLOSE/PROXY 报告：由 `evaluation_report.md` 保持。

## 7. 当前已知边界与建议

1. `news_factors` 月份必须覆盖评估月份；若缺月，news-aware 暴露构建会报错。
2. 运行环境需同时具备 `torch` 与 `pyyaml`。
3. 论文原始实证是 4 因子框架，9 因子属于扩展实验设定，建议在报告里明确标注为扩展版本。

## 8. 最小核查清单

1. `config/eval.yaml` 中 `model_type` 是否为目标值。
2. `news_factors` 路径是否可读取。
3. `artifacts/evaluation/portfolio_predictions.parquet` 是否存在（若只跑 evaluate）。
4. `outputs/evaluation/evaluation_report.md` 与 `skipped_metrics.json` 是否更新。


# Requirements (Training + LLM Scoring)

This file lists the Python packages needed across both parts of this project:

- Portfolio GAN training/evaluation in `portfolio_gan2`
- LLM news scoring scripts in `LLM news scoring`

## Core Packages

```text
numpy>=1.26
pandas>=2.2
pyyaml>=6.0
python-dotenv>=1.0
pyarrow>=15.0
scikit-learn>=1.4
torch>=2.2
torchvision>=0.17
torchaudio>=2.2
tqdm>=4.66
tensorboard>=2.16
```

## LLM Scoring Packages

```text
openai>=1.0
datasets>=2.18
```

## Testing

```text
pytest>=8.0
```

## Recommended Install Command

Run in your activated environment:

```bash
python -m pip install \
  numpy>=1.26 pandas>=2.2 pyyaml>=6.0 python-dotenv>=1.0 pyarrow>=15.0 \
  scikit-learn>=1.4 torch>=2.2 torchvision>=0.17 torchaudio>=2.2 \
  tqdm>=4.66 tensorboard>=2.16 openai>=1.0 datasets>=2.18 pytest>=8.0
```

## Notes

- The import name is `yaml`, but the package name is `pyyaml`.
- If you already use `environment.yml`, this markdown file is a checklist/reference for dependency completeness.

