import numpy as np
import utils
import random
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from logger import Logger
from config import Config
from models.MLP import MLP


conf = Config()

def load_data(path):
    data = utils.load_dataset(path, conf.sr)
    features = []
    labels = []
    for x, y in data:
        feature = np.concatenate(utils.extract_features(x, conf.sr, conf.n_mfcc))
        features.append(feature)
        labels.append(y)

    print("{} data in [{}] loaded".format(len(data), path))
    return np.array(features), np.array(labels)


def batchify(data, label, shuffle=False):
    data_size = len(data)
    order = list(range(data_size))
    if shuffle:
        random.shuffle(order)

    batches = int(np.ceil(1.*data_size/conf.batch_size))
    for i in range(batches):
        start = i * conf.batch_size
        indices = order[start:start+conf.batch_size]
        yield (data[indices], label[indices])


def acc(output, target):
    pred = output.data.max(1)[1] # get the index of the max log-probability
    correct = pred.eq(target.data).cpu().sum()
    return correct


def train(model, data, label, lr):
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=conf.L2)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    total_loss = 0.
    total_acc = 0.
    data_size = len(data)

    for batch, (x, y) in enumerate(batchify(data, label, True)):
        x = Variable(torch.Tensor(x), volatile=False)
        y = Variable(torch.LongTensor(y))
        model.zero_grad()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        opt.step()

        total_loss += loss.data[0]
        total_acc += acc(y_hat, y)

        if (batch + 1) % conf.log_interval == 0:
            size = conf.batch_size * batch + len(x)
            print("[{:5d}/{:5d}] batches\tLoss: {:5.6f}\tAccuracy: {:2.6f}"
                    .format(size, data_size, total_loss / size, total_acc / size))

    return total_loss/data_size, total_acc/data_size


def evaluate(model, data, label):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.
    total_acc = 0.
    data_size = len(data)

    for batch, (x, y) in enumerate(batchify(data, label)):
        x = Variable(torch.Tensor(x), volatile=True)
        y = Variable(torch.LongTensor(y))
        y_hat = model(x)
        loss = loss_fn(y_hat, y)

        total_loss += loss.data[0]
        total_acc += acc(y_hat, y)

    return total_loss/data_size, total_acc/data_size


def main():
    train_data, train_label = load_data(conf.train_path)
    n_features, n_labels = (len(train_data[0]), 11)
    model = MLP([n_features, 100, 50, n_labels], conf.dropout)

    valid_data, valid_label = load_data(conf.valid_path)
    min_loss = float("inf")
    lr = conf.lr

    save_file = os.path.join(conf.save_path, model.name)
    if os.path.exists(save_file):
        try:
            model = torch.load(save_file)
            min_loss, _ = evaluate(model, valid_data, valid_label)
        except RuntimeError, e:
            print("[loading existing model error] {}".format(str(e)))
            model = MLP([n_features, 100, 50, n_labels], conf.dropout)

    logger = Logger(conf.log_path)

    for epoch in range(conf.epochs):
        print("Epoch {}".format(epoch))
        train_loss, train_acc = train(model, train_data, train_label, lr)
        valid_loss, valid_acc = evaluate(model, valid_data, valid_label)

        if valid_loss < min_loss:
            min_loss = valid_loss
            with open(save_file, "wb") as f:
                torch.save(model, f)
        #else:
        #    lr = max(lr*0.9, 1e-5)

        print("Training set\tLoss: {:5.6f}\tAccuracy: {:2.6f}"
                .format(train_loss, train_acc))
        print("Validation set\tLoss: {:5.6f}\tAccuracy: {:2.6f}"
                .format(valid_loss, valid_acc))

        logger.scalar_summary("dev loss", valid_loss, epoch)
        logger.scalar_summary("dev acc", valid_acc, epoch)

    model = torch.load(save_file)
    test_data, test_label = load_data(conf.test_path)
    test_loss, test_acc = evaluate(model, test_data, test_label)
    print("Best validation loss: {}\nTest set\tLoss: {:5.6f}\tAccuracy: {:2.6f}"
            .format(min_loss, test_loss, test_acc))


if __name__ == "__main__":
    main()
