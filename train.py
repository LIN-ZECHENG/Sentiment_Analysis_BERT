import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, Trainer, TrainingArguments
from datasets import load_dataset
from tqdm import tqdm
from model import Rnn_Model 


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = logits.argmax(axis=1)  # 获取每个样本预测的类别
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

def train_model(args):
    #load model
    print(f'load {args.model}')
    if args.model == 'bert':
        tokenizer = AutoTokenizer.from_pretrained('./bert-base-chinese')
        model = AutoModelForSequenceClassification.from_pretrained('./bert-base-chinese', num_labels=args.num_class)
    elif args.model=='roberta':
        tokenizer = AutoTokenizer.from_pretrained('./xlm-roberta-base')
        model = AutoModelForSequenceClassification.from_pretrained('./xlm-roberta-base', num_labels=args.num_class)
    elif args.model=='bert+rnn':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained('./bert-base-chinese')
        base_model =  AutoModel.from_pretrained('./bert-base-chinese')
    elif args.model=='roberta+rnn':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained('./xlm-roberta-base')
        base_model =  AutoModel.from_pretrained('./xlm-roberta-base')

    #load dataset
    def _tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    dataset = load_dataset("lansinuote/ChnSentiCorp")
    encoded_dataset = dataset.map(_tokenize_function, batched=True)
    
    
    #load trainer
    if args.model=='bert' or args.model=='roberta':
        training_args=TrainingArguments(
            output_dir='.training_output',  #必须提供
            num_train_epochs=args.num_epoch,
            per_device_eval_batch_size=16,
            per_device_train_batch_size=16,
            eval_steps=50,
            save_steps=50
        )
        trainer=Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset['train'].select(range(args.training_set_lenth)),
            eval_dataset=encoded_dataset['validation'].select(range(args.validation_set_lenth)),
            compute_metrics=compute_metrics 
        )
        trainer.train()
        results = trainer.evaluate(eval_dataset=encoded_dataset['test'].select(range(args.test_set_lenth)))
        print(results)
        model_path=f'./trained_model/model_{args.model}'
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print("✅ 模型已完整保存")


    elif args.model=='bert+rnn' or args.model=='roberta+rnn':
        def _train(model, loader, optimizer, criterion):
            model.train()
            total_loss = 0
            for batch in tqdm(loader, desc="Training"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            return total_loss / len(loader)
        
        def _evaluate(model, loader):
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for batch in tqdm(loader, desc="Evaluating"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    predictions = torch.argmax(outputs, dim=1)

                    preds.extend(predictions.cpu().numpy())
                    targets.extend(labels.cpu().numpy())
            acc = accuracy_score(targets, preds)
            return acc


        input_size=768
        model = Rnn_Model(base_model, args.num_class,input_size).to(device)

        encoded_dataset = encoded_dataset.remove_columns(['text'])
        encoded_dataset = encoded_dataset.rename_column("label", "labels")
        encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        train_loader = DataLoader(encoded_dataset["train"].select(range(args.training_set_lenth)), batch_size=16, shuffle=True)
        val_loader = DataLoader(encoded_dataset["validation"].select(range(args.validation_set_lenth)), batch_size=16)
        test_loader = DataLoader(encoded_dataset["test"].select(range(args.test_set_lenth)), batch_size=16)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)  

        for epoch in range(args.num_epoch):
            print(f"Epoch {epoch+1}/{args.num_epoch}")
            train_loss = _train(model, train_loader, optimizer, criterion)
            val_acc = _evaluate(model, val_loader)
            print(f"Train Loss: {train_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        val_acc=_evaluate(model, test_loader)
        print(f'Test Accuracy:{val_acc}')

        model_path=f'./trained_model/model_{args.model}'
        tokenizer.save_pretrained(model_path)
        model.base_model.save_pretrained(model_path + "/base_model")
        torch.save(model.state_dict(), model_path + "/model.pt")
        print("✅ 模型已完整保存")

