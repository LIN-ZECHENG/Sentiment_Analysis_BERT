import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from config import get_config
from model import Rnn_Model
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


args=get_config()

if args.model=='bert' or args.model=='roberta':
    model_path = f'./trained_model/model_{args.model}'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

elif args.model=='bert+rnn' or args.model=='roberta+rnn':
    model_path = f'./trained_model/model_{args.model}'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModel.from_pretrained(f'{model_path}/base_model')
    model = Rnn_Model(base_model, num_classes=args.num_class, input_size=768)
    model.load_state_dict(torch.load(f'{model_path}/model.pt'))

model.eval()
text = "这家餐厅的服务态度真的很差，再也不来了！"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

with torch.no_grad():
    if args.model in ['bert', 'roberta']:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    elif args.model in ['bert+rnn', 'roberta+rnn']:
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

    probs = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    prob_values = probs.squeeze().tolist() 
    labels = ["消极", "积极"]
    predicted_label = labels[predicted_class]


print(f"输入文本: {text}")
print(f"预测情感: {predicted_label}")
print(f"各类别概率: 消极={prob_values[0]:.4f}, 积极={prob_values[1]:.4f}")
