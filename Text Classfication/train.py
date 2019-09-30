# coding: UTF-8
import time
import torch
import numpy as np
from importlib import import_module
from utils import build_dataset, build_iterator, get_time_dif

if __name__ == '__main__':
    data_set = 'THUCNews'
    embedding = 'embedding_SougouNews.npz'
    model_name = 'TextCNN'

    model_info = import_module('models.' + model_name)
    Config = model_info.Config(data_set, embedding)
    start_time = time.time()
    print('Loading data')
    # 如果用词，需要提前在语料上分好词
    vocab, train_data, dev_data, test_data = build_dataset(config=Config, ues_word=False)
    train_iter = build_iterator(train_data, Config)
    dev_iter = build_iterator(dev_data, Config)
    test_iter = build_iterator(test_data, Config)
    print("Time: ", get_time_dif(start_time))

    # train

    Config.n_voacb = len(vocab)
    model = model_info.Model(Config).to(Config.device)
    print(model.parameters)

    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
    for epoch in range(Config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, Config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)



