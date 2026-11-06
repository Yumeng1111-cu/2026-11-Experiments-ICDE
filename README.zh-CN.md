# 消融实验（论文第 7.2 节）

该目录是动态定价论文的 **7.2 消融实验独立代码包**，适合放入已有 GitHub 仓库，与 7.1 理论验证及其他实验代码并列。每个实验包可独立安装依赖、运行与绘图，不会与别的目录中同名的 `settings.py`、`core.py` 冲突。

**本次保留所有现有数据与图片：** 原包内的正式 CSV、快速测试 CSV、PNG、PDF、SVG 均未删减或重新计算；仅去除了与 GitHub 分享无关的 `.venv`、`.idea` 和编译缓存。

## 三组实验

- **A**：平均时间敏感度 `mu∈{0.01,0.02,0.03}` 与相对异质性 `CV∈{0.2,0.3,0.4,0.5}`，`n=80, t=80`。
- **B**：买家数 `n∈{20,40,60,80,100,120}` 与 `CV∈{0.3,0.4,0.5}`，`mu=0.02, t=80`。
- **C**：目标时期 `t∈{40,50,60,70,80,90,100}` 与 `CV∈{0.3,0.4,0.5}`，`mu=0.02, n=80`。

六种对比方法：**U-FH、C-FH、U-SW-FV、ACRP-FV、Oracle-SW、Oracle-EW**。模型使用 `ln(X) ~ N(0, 0.5²)` 与均值为 `mu`、变异系数为 `CV` 的 Gamma 时间敏感度，前向验证期数 `M=20`。完整实现与其他参数参见 `core.py`、`settings.py`。

## 如何运行

在 PyCharm 中打开 `ablation_studies/` 文件夹，或者在终端进入此文件夹，然后执行：

```bash
python -m pip install -r requirements.txt
python verify_saved_data.py
python plot_all.py
```

这会先检查现有数据，再根据 **正式 CSV 直接绘制 A/B/C 三张图**，不会重新执行模拟。若只需要修改字体、线条、颜色，编辑 `plot_settings.py` 即可。可单独运行 `plot_A.py`、`plot_B.py`、`plot_C.py`。

独立计算测试：

```bash
python test_pricing.py
```

重新计算数值实验：在 `settings.py` 中将 `FORCE_RECOMPUTE = True`，然后运行：

```bash
python run_all.py
```

此命令在正式模式下会执行大量计算（每种配置 5000 次测试、每种 Oracle 5000 次训练）。建议先改 `QUICK_MODE = True` 试运行。默认已将 `FORCE_RECOMPUTE` 改为 `False`，避免误操作覆盖上传的数据。单组模拟也可运行 `experiment_A.py` 等脚本。

**注意：** `outputs/data/quick/` 原包只有 A 和 B 的汇总数据，缺少快速模式 C 的结果；因此 `plot_all.py` 默认读取 `formal` 是正常的。如果要绘制全部快速模式图，需要先运行快速模式 C。

## 保存的数据和图

- `outputs/data/formal/`：全部三组正式实验汇总、配置以及逐场景的 Monte Carlo CSV；一共有 51 个实验配置、45 份去重的逐场景文件。
- `outputs/data/quick/`：上传时已有的快速测试结果，按原样保留。
- `outputs/figures/`：A/B/C 三组图的 PNG、PDF、SVG 共九个图表文件。

注意图 B 的绘图配置目前只展示 `n=40..120`，即使 CSV 中还保留 `n=20`；这是绘图展示设置，不是丢失数据。

## 添加到已有 GitHub 仓库

将此压缩包解压，把**整个 `ablation_studies` 文件夹**作为一个新目录上传到已有仓库。不要把里面的 `core.py`、`settings.py` 直接放到仓库根目录，否则容易与 7.1 的同名文件冲突。你也可以把文件夹放在 `experiments/ablation_studies/` 下；最终论文仍然只需一个 GitHub 仓库链接。

本包不自动添加开源许可证或作者信息。论文若处于匿名审稿阶段，公开仓库前需要检查匿名政策。
