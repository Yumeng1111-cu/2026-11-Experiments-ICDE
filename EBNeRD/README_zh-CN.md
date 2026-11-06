# EB-NeRD Small 新闻点击率时间衰减：可复现实验代码

这是一个适合 GitHub 的**仅代码版本**，用于研究新闻发布后点击率（Click-through rate）
随新闻年龄变化的规律，包含逐小时聚合、指数拟合、按文章重采样的 bootstrap 和绘图。

**重要：** 本仓库不包含 EB-NeRD 原始数据、任何文章 ID、逐文章统计表，
也默认不上传真实数据衍生的统计 CSV、JSON 或图像。原数据集并不是可随意
再分发的数据，其许可第 4.6 条限制了研究用途。若将结果用于动态定价研究，
请先向权利方确认是否符合其授权范围。

官方数据入口：https://recsys.eb.dk/

官方许可：https://recsys.eb.dk/assets/pdf/general_license_terms.pdf

## 1. 不需要原始数据即可测试

进入项目根目录，运行：

```powershell
python -m pip install -r requirements.txt
python demo.py
python plot_results.py --input results/demo
python check_reproduction.py --output results/demo
```

`demo.py` 生成的是**纯模拟数据**，不是 EB-NeRD 的统计结果。
图像保存在 `results/demo/`，不会被 Git 提交。

## 2. 取得官方数据后复现

从官网按许可条款获取 **EB-NeRD Small**，将其 ZIP 放到 `data/ebnerd_small.zip`，
或将其解压到本地目录。然后运行：

```powershell
python run.py --zip data/ebnerd_small.zip --bootstrap 500
python plot_results.py --input results
python check_reproduction.py --output results
```

已经解压时：

```powershell
python run.py --data-dir "D:\\Datasets\\ebnerd_small" --bootstrap 500
```

测试速度优先时，可将 `--bootstrap 500` 改为 `--bootstrap 0`，
但不再生成 Bootstrap 置信区间。

必需的原始文件：`articles.parquet`、`train/behaviors.parquet`、
`validation/behaviors.parquet`。

## 3. 代码与输出位置

- `run.py`：读取官方原始数据，按新闻年龄聚合并拟合指数模型。
- `src/reader.py`：按批次读取 Parquet。
- `src/analysis.py`：计算曝光、点击、点击率；极大似然拟合；bootstrap。
- `src/figures.py`：两面板诊断图（0–48 h 与 8–48 h）。
- `plot_results.py`：在不需要原始数据的前提下，用已生成的汇总结果绘制
  **观测曲线 + 8–48 h 指数拟合曲线**，使用 Times New Roman 风格和论文色系。
- `check_reproduction.py`：检查本地 CSV/JSON 一致性；支持比较另一份本地参考结果。
- `results/`：本地输出（默认 `.gitignore` 排除）。
- `data/`：官方数据放置位置（默认 `.gitignore` 排除）。

所有样式参数可在 `plot_results.py` 顶部修改。

## 4. 科学和许可方面的注意事项

- 新闻候选曝光以 `article_ids_inview` 中每篇文章出现一次计一次。
- 同一曝光记录内的 ID 还在 `article_ids_clicked` 出现，则记录点击。
- 使用 0–47 的整小时年龄分箱，按 `clicks/exposures` 计算聚合点击率。
- 8–48 h 衰减期采用二项似然极大化拟合；逐小时曝光数用于拟合和加权 R²。
- 默认做 500 次按文章重采样的 Bootstrap。
- **8 h 起点是查看实际曲线后选择的**，属于样本内拟合，并非独立检验。
- **点击率不等于愿付价格（WTP），无法直接观测动态定价买家敏感度。**
- 真实数据衍生图、CSV 和 JSON 的公开分发应按许可条款审核。

详见 [英文说明](README.md)、[方法细节](docs/METHODOLOGY.md)、
[数据许可说明](docs/DATA_ACCESS_AND_LICENSE.md)。

## 5. 上传到 GitHub

建议创建一个新的公开仓库，将该文件夹内容上传。执行 `git status` 时，
检查确认没有 `data/ebnerd_small.zip`、`*.parquet`、`*.npz`、
`results/*.csv` 或任何未经确认允许公开的衍生结果。

本包没有替你选择代码的开源许可证；如需 MIT 等代码许可证，
可在 GitHub 上自行选择，且代码许可证不适用于 EB-NeRD 数据。
