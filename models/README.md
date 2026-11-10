# 模型

`pretrained/mobilenet_v3_small-047dcff4.pth`是torchvision官方ImageNet初始化权重，来源与完整SHA-256在旁边的provenance.json中。torchvision源码采用BSD-3-Clause许可。

实际完成10类Food-101训练后，验证选定的模型才可部署为`food10/best.pt`。ImageNet权重不包含本项目训练过的10类分类头。
