# 菜品识别与热量区间估算

《神经网络与深度学习》课程第02题，中等难度。

使用 **Food-101的10类现成标签**训练MobileNetV3-Small，展示菜品Top-3预测；用户确认类别并输入克数后，根据有出处的每100g营养值估算热量范围。

当前已完成新项目代码、10类营养来源核验和浏览器流程检查。Food-101的10000张图片已准备完成，三组正式训练正在本机顺序运行；最终测试成绩尚待汇总。实时状态以网页和`runs/food10_seed42/status.json`为准。

## 运行

双击 **`start_food_app.command`**，打开 **http://127.0.0.1:8780/**。

终端在项目根目录运行：

```bash
.venv-sitting-posture/bin/python scripts/start_food_app.py --open-browser
```

继续复用已安装的Python3.12.2/PyTorch2.8.0环境，因此虚拟环境目录保留原名称。实际业务代码与命令已改为`foodcal`。未训练时，网页明确提示模型未就绪；已核验营养条目的菜品可以手动选择并计算热量。

## 类别

披萨、汉堡、薯条、热狗、炒饭、煎饺、寿司、冰淇淋、华夫饼、肉酱意大利面。英文类别和固定顺序在 [configs/default.json](configs/default.json)。

## 数据准备

数据源：[Food-101官方页面](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)。原图版权属于各自权利人，科学研究之外的使用按官方说明处理。

```bash
.venv-sitting-posture/bin/python scripts/download_food101.py --prepare
```

命令支持断点续传和单实例锁，先校验官方压缩包MD5，只解压选择的10类。保留官方test；validation从官方train按类别和图片哈希组派生。预计10类共10000张：6000训练、1500验证、2500测试；如果发现训练图片与官方测试图片完全重复，将保留测试而排除对应训练图片并写入审计，最终数量以清单为准。

在新设备上，可先安装[uv](https://docs.astral.sh/uv/)并执行：

```bash
uv venv --python 3.12 .venv-sitting-posture
uv pip install --python .venv-sitting-posture/bin/python -e .
.venv-sitting-posture/bin/python scripts/download_pretrained.py
```

这会安装`pyproject.toml`中的项目依赖并校验官方初始化权重；`requirements-macos-py312.lock.txt`另记录本机完整复用环境，其中含新项目不需要的历史依赖。下载及训练脚本使用POSIX文件锁，当前验证平台为macOS。正式训练前，实验脚本还会用一张真实训练图片完成官方ImageNet模型的推理与资源检查。

进度记录：`runs/setup/food101_download.json`；最终清单：`data/processed/food101_10/manifest.csv`。下载任务已启动时不要再开第二个准备进程。

## 训练与对比

已启动的任务会等待数据准备完成，然后顺序完成A/B/C三种设置，按验证集Macro-F1选模型，最后对固定方案评估官方测试集并部署。单独重跑时可使用：

```bash
.venv-sitting-posture/bin/python scripts/run_food_experiments.py --device mps
```

默认输出 `runs/food10_seed42/`，日志 `runs/setup/food_experiments.log`。同一输出有单实例锁；完整实验可跳过，未完成的训练不会伪装成可续训权重，重试需选择新的`--output`。`--wait-for-data`仅用于等待已启动的下载任务。不要同时运行下面的单组命令；它们用于单独复现实验。

```bash
# A：冻结预训练骨干，仅训练分类头
.venv-sitting-posture/bin/python -m foodcal train \
  --output runs/frozen_seed42 --mode frozen --device mps

# B：在相同数据与设置下，解冻最后一个特征块
.venv-sitting-posture/bin/python -m foodcal train \
  --output runs/last_block_seed42 --mode last_block --device mps

# C：在B的设置上加入颜色增强
.venv-sitting-posture/bin/python -m foodcal train \
  --output runs/last_block_aug_seed42 --mode last_block --augment --device mps
```

训练只按validation选择最佳权重；最终test留到方案固定后评估。每轮输出使用新目录，记录实际曲线、验证指标、模型与配置。对比A/B检验微调范围，对比B/C检验增强。

```bash
.venv-sitting-posture/bin/python -m foodcal evaluate \
  --checkpoint runs/last_block_seed42/checkpoint.pt \
  --output runs/last_block_seed42_test --split test --device mps
```

顺序实验脚本将选定的10类checkpoint部署到`models/food10/best.pt`；正在运行的网页会在下一次状态刷新时自动加载，无需改代码。也可启动时用`--checkpoint`指定其他已训练模型。ImageNet1000类初始化模型不能代替菜品分类器。

## 营养估算

营养条目的具体食物描述、单位和来源保存在`data/nutrition.json`，说明见[营养数据来源](docs/营养数据来源.md)。按`kcal/100g × 用户输入克数/100`计算；参考范围描述已列配方的差异，不是置信区间，也不能覆盖任意油量和做法。模型不从照片测量重量。

## 检查

```bash
.venv-sitting-posture/bin/python -m foodcal check
.venv-sitting-posture/bin/python -m unittest discover -s tests -v
.venv-sitting-posture/bin/python data/nutrition_sources/verify_sources.py
node scripts/check_food_browser.mjs
```

项目计划与课程对应见 [项目计划](docs/项目计划.md)，实际运行记录见[运行记录](docs/运行记录.md)。原始课程要求Word仅保留在本机，仓库提供相关要求摘录。代码仓库：[Yukikazct/food-calorie-estimation](https://github.com/Yukikazct/food-calorie-estimation)，保留原有Git提交历史。原始图片、虚拟环境、模型权重和本机训练输出通过下载与训练脚本在本地重建。
