import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from datasets import load_dataset
from torch.utils.data import DataLoader
from config import get_config
from model import Rnn_Model
from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


args = get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型加载
model_path = f'./trained_model/model_{args.model}'
tokenizer = AutoTokenizer.from_pretrained(model_path)

if args.model == 'bert' or args.model == 'roberta':
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

elif args.model == 'bert+rnn' or args.model == 'roberta+rnn':
    base_model = AutoModel.from_pretrained(f'{model_path}/base_model')
    model = Rnn_Model(base_model, num_classes=args.num_class, input_size=768).to(device)
    model.load_state_dict(torch.load(f'{model_path}/model.pt'))
    
model.eval()

def _tokenize_function(example):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=128)

dataset = load_dataset("lansinuote/ChnSentiCorp")
encoded_dataset = dataset.map(_tokenize_function, batched=True)
encoded_dataset = encoded_dataset.remove_columns(['text'])
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

test_loader = DataLoader(
    encoded_dataset["test"],
    batch_size=16
)

preds, targets = [], []
model.eval()
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        if args.model in ['bert', 'roberta']:
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        elif args.model in ['bert+rnn', 'roberta+rnn']:
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

        pred = torch.argmax(logits, dim=1)
        preds.extend(pred.cpu().numpy())
        targets.extend(labels.cpu().numpy())

test_acc = accuracy_score(targets, preds)
print(f"✅ 测试集准确率: {test_acc:.4f}")
