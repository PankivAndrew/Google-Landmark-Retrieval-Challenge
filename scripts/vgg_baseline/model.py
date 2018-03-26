import sys
sys.path.append('../../')

import os
import logging
import pandas as pd
import torch
from torch.autograd import Variable
from torchvision.transforms import transforms

from scripts.dataset import VGGDataset, GLRCDataset
from scripts.vgg_baseline.vgg import Vgg16

def to_vgg_tensors(in_folder, out_folder, use_cuda=False, log_to="../../data/log/to_vgg_tensors.log"):
    batch_size = 100
    logging.basicConfig(filename=log_to, level=logging.INFO)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    vgg = Vgg16()
    if use_cuda:
        vgg = vgg.cuda()
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = GLRCDataset(in_folder, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    for i, (ids, imgs) in enumerate(loader):
        minibatch = len(imgs)
        imgs = Variable(imgs)
        if use_cuda:
            imgs = imgs.cuda()
        features = vgg(imgs).relu3_3.view(minibatch, -1)
        for this_id, feature in zip(ids, features):
            torch.save(feature, os.path.join(out_folder, this_id))
        logging.info("{0}/{1} patch".format(i + 1, len(loader)))


def classify(use_cuda=False, log_to="../../data/log/classify.log"):
    batch_size = 100
    logging.basicConfig(filename=log_to, level=logging.INFO)

    transform = transforms.Compose([transforms.ToTensor()])
    index_dataset = GLRCDataset("../../data/index_resized", transform=transform)
    test_dataset = GLRCDataset("../../data/test_resized", transform=transform)
    loader_index = torch.utils.data.DataLoader(index_dataset, batch_size=batch_size)

    vgg = Vgg16()
    if use_cuda:
        vgg = vgg.cuda()

    series = []
    test_indexes = []
    for i, (test_index, test_img) in enumerate(test_dataset):
        test_imgs = None
        all_indexes = []
        all_losses = []
        for j, (index_indexes, index_imgs) in enumerate(loader_index):
            minibatch = len(index_imgs)
            if test_imgs is None or len(test_imgs) != minibatch:
                test_imgs = test_img.repeat(minibatch, 1, 1, 1)
                test_imgs = Variable(test_imgs)
                if use_cuda:
                    test_imgs = test_imgs.cuda()
                test_features = vgg(test_imgs).relu3_3.view(minibatch, -1)

            index_imgs = Variable(index_imgs)
            if use_cuda:
                index_imgs = index_imgs.cuda()
            index_features = vgg(index_imgs).relu3_3.view(minibatch, -1)

            loss = ((test_features - index_features) ** 2).sum(1)
            all_indexes += index_indexes
            all_losses += [float(loss.cpu().data.numpy()[0])]
            #if j % 10 == 99:
            logging.info("{0}/{1} image, {2}/{3} patch".format(i + 1, len(test_dataset), j + 1, len(loader_index)))
        df = pd.DataFrame([pd.Series(all_losses, all_indexes)])
        df.to_csv("../../data/output/" + test_index + ".csv")

if __name__ == "__main__":
    to_vgg_tensors("../../data/index_resized", "../../data/index_tensors", use_cuda=True)
    to_vgg_tensors("../../data/test_resized", "../../data/test_tensors", use_cuda=True)
