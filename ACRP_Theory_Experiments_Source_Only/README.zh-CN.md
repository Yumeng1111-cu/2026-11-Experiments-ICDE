# ACRP 理论机制数值验证：纯代码仓库

本项目整理自上传的 `ACRP_7_1_PyCharm_Python.zip`，包含三组理论实验的可运行 Python 源代码、绘图代码、参数设置和检查脚本。


## 1. 安装依赖

在项目文件夹执行：

```bash
python -m pip install -r requirements.txt
```

推荐使用 Python 3.13。若在 PyCharm 中运行，请为当前项目新建虚拟环境并选择相应解释器。

## 2. 一键从零生成数据和三张图

先做快速测试：

```bash
python reproduce.py --quick --reps 2
```

这会从头计算实验 1、2、3，生成 `outputs/data/quick/` 下的 CSV 和 `outputs/figures/` 下的 PNG、PDF、SVG。**快速测试仅检查代码是否能运行，不代表论文正式统计结果。** Numba 首次编译需要额外时间。

正式实验（默认 5000 次测试模拟与 5000 次独立 pilot 模拟）：

```bash
python reproduce.py
```

正式运行计算量较大，参数可以在 `settings.py` 中修改。

## 3. 只计算 / 只画图

在 `settings.py` 中设置 `QUICK_MODE=True`（快速）或 `False`（正式），仅计算数据：

```bash
python generate_all.py
```

在 `plot_settings.py` 中设置 `PLOT_DATA_MODE="quick"` 或 `"formal"`，仅绘图：

```bash
python plot_all.py
```

单独运行 `generate_1.py`、`generate_2.py`、`generate_3.py` 或 `plot_1.py`、`plot_2.py`、`plot_3.py` 亦可。

**注意：先生成 CSV 才能单独绘图。** 绘图程序不会自行补算缺失数据。实验 1 的字号、轴线粗细与其他图分开在 `plot_settings.py` 中控制。当前实验 3 使用暖色曲线（`OPTIMUM_STYLES`）。

## 4. 生成的文件保存在哪里？

```text
outputs/                        # 运行后自动创建，不上传 GitHub
├── data/
│   ├── quick/                 # 快速测试产生的 CSV
│   ├── formal/                # 正式实验产生的 CSV
│   └── sigma*.csv             # Monte Carlo 中间缓存
└── figures/
    ├── Experiment1_Mismatch.png / .pdf / .svg
    ├── Experiment2_MemoryTradeoff.png / .pdf / .svg
    └── Experiment3_ComparativeStatics_4Panels.png / .pdf / .svg
```

`.gitignore` 已忽略整个 `outputs/`，保证仓库只收录源代码，不会意外把先前跑出的两套 CSV 混在一起。

## 5. 数据检查

完成计算后可以验证文件结构：

```bash
python test_pricing.py
python scripts/verify_data.py --mode quick
# 正式实验完成后改为 --mode formal
```

## 6. 模型与范围

本理论验证项目的基础估值满足 `log X ~ N(0,0.5²)`，时间敏感度服从 Gamma 分布，平均敏感度 `MU=0.02`。它不是后来采用截断 Logistic / FCC / EB-NeRD 数据的另一套实验。

本仓库未擅自添加许可证与论文作者信息；正式公开时按作者决定补充。

[English README](README.md)
