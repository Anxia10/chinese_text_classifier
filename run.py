import time
import torch
import argparse
import numpy as np
from importlib import import_module
from datetime import timedelta
from train import train

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default='TextRNN', help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='random', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    model_name = args.model
    if args.embedding == 'random':
        embedding = 'random'

    # from data_process import build_dataset, build_iterator, get_time_dif
    from data_process import DatasetIterater,one_hot,read_vocab

    x = import_module('model.' + model_name)
    config = x.Config(dataset,embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    # 先再data_process中创建vocab
    vocab = read_vocab(config.vocab_path)
    train_data = one_hot(config.train_path,vocab,'Chinese',config.pad_size)
    dev_data = one_hot(config.dev_path,vocab,'Chinese',config.pad_size)
    test_data = one_hot(config.test_path,vocab,'Chinese',config.pad_size)
    train_iter = DatasetIterater(train_data,config.batch_size, config.device)
    dev_iter = DatasetIterater(dev_data,config.batch_size, config.device)
    test_iter = DatasetIterater(test_data,config.batch_size, config.device)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    #train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    print(model.parameters)
    train(config,model,train_iter,dev_iter,test_iter)


