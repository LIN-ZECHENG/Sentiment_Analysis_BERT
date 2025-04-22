import argparse

def get_config():
    parser=argparse.ArgumentParser()
    parser.add_argument('--num_class',type=int,default=2)
    parser.add_argument('--model',type=str,default='bert',
                        choices=['bert','roberta','bert+rnn','roberta+rnn'])
    parser.add_argument('--num_epoch',type=int,default=3)
    parser.add_argument('--training_set_lenth',type=int,default=1000)
    parser.add_argument('--validation_set_lenth',type=int,default=50)
    parser.add_argument('--test_set_lenth',type=int,default=50)
    args = parser.parse_args()

    return args

