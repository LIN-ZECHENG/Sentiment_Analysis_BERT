from config import get_config        
from train import train_model

if __name__ == '__main__':
    args=get_config()
    train_model(args)
    print('Finish Training.')
