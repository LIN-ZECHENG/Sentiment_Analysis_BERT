<div align="center">
  
[English](../README.md)| 简体中文

![pic](https://github.com/LIN-ZECHENG/Sentiment_Analysis_BERT/blob/main/src/20250422_2117_%E7%8E%B0%E4%BB%A3%E6%83%85%E6%84%9F%E5%88%86%E6%9E%90_simple_compose_01jset7vtvf6rv53bkmm2grnf3.png?raw=true)

<h2 id="title">Sentiment_Analysis_BERT</h2>
</div>




### 📌项目简介

本项目基于中文预训练语言模型 `bert-base-chinese` 与 `xlm-roberta-base`，结合 RNN 神经网络结构，构建一个轻量级的中文情感分析系统。模型在 [ChnSentiCorp](https://huggingface.co/datasets/ChnSentiCorp) 数据集上进行训练，支持快速部署与使用。

项目适用于中文短文本情感分类任务，便于扩展到评论分析、用户反馈识别等实际场景。





### 🎯实验结果

仅使用 ChnSentiCorp 数据集中的 **1000 条训练样本**，训练 **3 个周期（Epochs）**，在完整测试集上的准确率如下所示：

| **模型结构**                   | **准确率（ACC）** |
| ------------------------------ | ----------------- |
| BERT (bert-base-chinese)       | **88.42%**        |
| XLM-RoBERTa (xlm-roberta-base) | **87.75%**        |
| BERT + RNN                     | **88.92%**        |
| XLM-RoBERTa + RNN              | **88.67%**        |





### 🚀快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 运行示例（Demo）

```bash
python demo.py
```





### ⚙️模型训练与配置

使用 `main.py` 启动训练任务，可通过 `config.py` 修改训练参数（如模型类型、训练轮数、批次大小等）：

```bash
python main.py --model bert+rnn --num_epoch 5
```

训练后的模型将保存在 `./trained_model` 目录下（已上传训练后的模型，可以直接使用）。





### 📈 模型评估与部署

测试模型在验证集上的表现：

```bash
python test_set.py
```

使用训练好的模型进行情感预测：

```bash
python demo.py
```

你可以输入任意中文句子，模型将自动判断其情感倾向（积极 / 消极）。





### 📂 目录结构

```bash
├── config.py               # 超参数配置
├── main.py                 # 模型训练入口
├── demo.py                 # 单句情感预测
├── test_set.py             # 在测试集上评估模型
├── model.py                # 模型定义（含RNN结构）
├── trained_model/          # 保存训练后的模型
├── requirements.txt        # Python依赖项
```


![Search](https://github.com/LIN-ZECHENG/Sentiment_Analysis_BERT/blob/main/src/Product-of-the-Week-%5Bremix%5D.gif?raw=true)

