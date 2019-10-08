# coding: UTF-8
import time
import torch
import torch.nn.functional as F
import numpy as np
from importlib import import_module
from sklearn import metrics
from utils import build_dataset, build_iterator, get_time_dif
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


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
    total_batch = 0  # 记录进行到多少batch
    flag = False  # 记录是否很久没有效果提升
    dev_best_loss = float('inf')

    for epoch in range(Config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, Config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(Config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), Config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  ' \
                      'Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
            total_batch += 1
            if total_batch - last_improve > Config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
                break

    test(Config, model, test_iter)
