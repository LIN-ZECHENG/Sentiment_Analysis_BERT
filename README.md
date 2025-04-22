<div align="center">
  
English | [简体中文](src/README_CN.md)


### 

<h2 id="title">Sentiment_Analysis_BERT</h2>
</div>

### 📌 Project Overview

This project is based on the Chinese pre-trained language models `bert-base-chinese` and `xlm-roberta-base`, combined with the RNN neural network structure, to build a lightweight Chinese sentiment analysis system. The model is trained on the [ChnSentiCorp](https://huggingface.co/datasets/ChnSentiCorp) dataset and supports rapid deployment and use.

The project is suitable for Chinese short-text sentiment classification tasks and can be easily extended to scenarios like comment analysis and user feedback recognition.



### 🎯 Experimental Results

The model was trained with **1000 training samples** from the ChnSentiCorp dataset for **3 epochs**. The accuracy on the full test set is shown below:

| **Model Architecture**         | **Accuracy (ACC)** |
| ------------------------------ | ------------------ |
| BERT (bert-base-chinese)       | **88.42%**         |
| XLM-RoBERTa (xlm-roberta-base) | **87.75%**         |
| BERT + RNN                     | **88.92%**         |
| XLM-RoBERTa + RNN              | **88.67%**         |



### 🚀 Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Run the demo:

```bash
python demo.py
```



### ⚙️ Model Training and Configuration

Start the training task using `main.py`. You can modify training parameters (such as model type, number of epochs, batch size, etc.) in `config.py`:

```bash
python main.py --model bert+rnn --num_epoch 5
```

The trained model will be saved in the `./trained_model` directory (the trained model has already been uploaded and can be used directly).



### 📈 Model Evaluation and Deployment

Test the model performance on the validation set:

```bash
python test_set.py
```

Use the trained model to predict sentiment:

```bash
python demo.py
```

You can input any Chinese sentence, and the model will automatically determine its sentiment (positive/negative).



### 📂 Directory Structure

```bash
├── config.py               # Hyperparameter configuration
├── main.py                 # Model training entry point
├── demo.py                 # Single-sentence sentiment prediction
├── test_set.py             # Evaluate the model on the test set
├── model.py                # Model definition (including RNN structure)
├── trained_model/          # Directory for saving trained models
├── requirements.txt        # Python dependencies
```


![Search](https://github.com/LIN-ZECHENG/Sentiment_Analysis_BERT/blob/main/src/Product-of-the-Week-%5Bremix%5D.gif?raw=true)
