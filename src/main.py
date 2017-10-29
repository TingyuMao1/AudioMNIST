import numpy as np
import os

import torch

from logger import Logger
from config import Config
import utils
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
            min_loss, _ = model.evaluate(valid_data, valid_label)
            print("Initial validation loss: {:5.6f}".format(min_loss))
        except RuntimeError, e:
            print("[loading existing model error] {}".format(str(e)))
            model = MLP([n_features, 100, 50, n_labels], conf.dropout)

    logger = Logger(conf.log_path)

    for epoch in range(conf.epochs):
        print("Epoch {}".format(epoch))
        train_loss, train_acc = model.train_(train_data, train_label, lr, conf)
        valid_loss, valid_acc = model.evaluate(valid_data, valid_label)

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
    test_loss, test_acc = model.evaluate(test_data, test_label)
    print("Best validation loss: {:5.6f}\nTest set\tLoss: {:5.6f}\tAccuracy: {:2.6f}"
            .format(min_loss, test_loss, test_acc))


if __name__ == "__main__":
    main()
