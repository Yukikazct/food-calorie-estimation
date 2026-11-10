# 项目范围

当前项目为课程第02题“菜品识别与热量区间估算”。仅维护食物图像分类、份量输入与有来源的营养估算，不恢复已删除的旧项目。

# 固定运行环境

继续复用用户指定的虚拟环境 `.venv-sitting-posture/`（仅保留已有目录名）。安装、数据处理、训练、测试和推理均使用 `.venv-sitting-posture/bin/python`，不使用系统 Python 或默认 Conda 替代。

- 环境检查：`.venv-sitting-posture/bin/python -m foodcal check`
- 测试：`.venv-sitting-posture/bin/python -m unittest discover -s tests -v`
- Python3.12.2 / PyTorch2.8.0 / torchvision0.23.0 / Apple MPS。
- 依赖变化后更新 `requirements-macos-py312.lock.txt`。

# 数据与实验

- Food-101选10类；保持官方train/test划分，validation仅从官方train派生，固定种子并记录来源。
- 同源重复图片不得跨训练/验证/测试。最终test不用于选模型或调整超参数。
- 预训练ImageNet权重不等于已训练的食物分类模型。随机初始化、模拟数据及程序检查不得写成真实分类结果。
- 至少两组控制变量对照：冻结骨干/解冻末层，以及不增强/颜色增强。
- 热量由用户选择的菜品、输入克数及有来源的每100g营养值计算；不得声称单张照片能确定重量或精确热量。
- 实验产物放 `runs/`，原始数据放 `data/raw/`，处理清单放 `data/processed/`，结论放 `docs/`。
- 原始课程要求和Git历史保留；未授权不得改写或删除远程仓库。
