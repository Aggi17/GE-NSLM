import os

import numpy as np
from torch import nn
import torch as t

t.cuda.empty_cache()
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from transformers import BertModel
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import random

from torchvision import datasets, models, transforms
import os
import re
import pandas as pd
from PIL import Image
import jieba
import os.path
import csv
from transformers import BertTokenizer

import re
from math import log

from PIL import Image
import numpy as np

import random
import torchvision

from .InceptionV3 import GoogLeNet

data_path = '/home/jd/code/dyq/ALGM22/EANN/Data/twitter'

original_train_data = os.path.join(data_path, 'train16.txt')
original_test_data = os.path.join(data_path, 'test16.txt')

new_train = os.path.join(data_path, 'new_train.txt')
new_test = os.path.join(data_path, 'new_test.txt')

image_file_list = [os.path.join(data_path, 'image_sum/')]

image_id_path = os.path.join(data_path, 'tid_image.txt')


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


def read_images(file_list):
    image_list = {}
    img_num = 0

    for path in file_list:
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        for filename in os.listdir(path):
            try:
                img = Image.open(path + filename).convert('RGB')
                img = data_transforms(img)

                img_id = filename.split('.')[0]
                image_list[img_id] = img
                img_num += 1
            except:
                print(filename)

    print("图片总数", img_num)
    return image_list


def get_imgid(image_id_path):
    f = open(image_id_path, 'r', encoding='UTF-8')
    imgid_dic = {}

    lines = f.readlines()
    for i, l in enumerate(lines):
        if i == 0:
            continue
        else:
            postId = l.split('\t')[0]
            imageId = l.split('\t')[1].strip()
            imgid_dic[postId] = imageId
    return imgid_dic


def select_image(image_num, image_id_list, image_list):
    for i in range(image_num):

        image_id = image_id_list[i]

        if image_id in image_list:
            return image_id

    return False


def select_data(twitter_original_data, twitter_selected_data):
    f_old = open(twitter_original_data, 'r', encoding='UTF-8')
    f_new = open(twitter_selected_data, 'w', encoding='UTF-8')

    fake_count = 0
    real_count = 0

    lines = f_old.readlines()
    for i, l in enumerate(lines):
        if i == 0:
            continue
        else:
            postId = l.split('\t')[0]
            label = l.split('\t')[-1].strip()

            imageId_list = l.split('\t')[-3]
            postText = l.split('\t')[1]

            clean_postText = re.sub(r"(http|https)((\W+)(\w+)(\W+)(\w*)(\W+)(\w*)|(\W+)(\w+)(\W+)|(\W+))", "", postText)
            clean_postText = ' '.join(
                re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", clean_postText).split())

            if len(clean_postText) < 5:
                continue

            if (label == 'fake'):
                label = '1'
                fake_count = fake_count + 1
            elif (label == 'real'):
                label = '0'
                real_count = real_count + 1
            else:
                pass

            f_new.write(postId + '|' + clean_postText + '|' + imageId_list + '|' + label + '\n')

    f_old.close()
    f_new.close()

    return fake_count, real_count


def get_max_len(file):
    f = open(file, 'r', encoding='UTF-8')

    max_post_len = 0

    lines = f.readlines()
    post_num = len(lines)
    for i in range(post_num):
        post_content = list(lines[i].split('\t')[1].split())
        tmp_len = len("".join(post_content))
        if tmp_len > max_post_len:
            max_post_len = tmp_len

    f.close()
    return max_post_len


def get_data(dataset, image_list, img_dic):
    if dataset == 'train':
        data_file = new_train
    else:
        data_file = new_test

    f = open(data_file, 'r', encoding='UTF-8')
    lines = f.readlines()

    data_post_id = []
    data_post_content = []
    data_image = []
    data_label = []
    data_img_id = []

    data_num = len(lines)
    print(data_num)
    unmatched_num = 0
    sen, me = 0.0, 0.0
    max, min = 0, 5
    a, b, c = 0, 0, 0

    for line in lines:
        post_id = line.split('|')[0]
        post_content = line.split('|')[1]
        label = line.split('|')[-1].strip()
        if dataset == 'train':
            if post_id in img_dic:

                image_id = img_dic[post_id].strip().split(' ')[0]
                if image_id in image_list:
                    image = image_list[image_id]
                    data_post_id.append(int(post_id))
                    data_post_content.append(post_content)
                    data_image.append(image)
                    data_label.append(int(label))
                    data_img_id.append(image_id)
                    l = len(post_content.split())
                    if l < min: min = l
                    if l > max: max = l
                    sen += l
                    a += 1

                else:
                    unmatched_num += 1

            else:
                image_id = line.split('|')[2]
                if image_id in image_list:
                    image = image_list[image_id]
                    data_post_id.append(int(post_id))
                    data_post_content.append(post_content)
                    data_image.append(image)
                    data_label.append(int(label))
                    data_img_id.append(image_id)
                    l = len(post_content.split())
                    if l < min: min = l
                    if l > max: max = l
                    sen += l
                else:
                    unmatched_num += 1
                    continue
        else:
            image_id = line.split('|')[2].strip().split(',')[0]

            if image_id in image_list:
                image = image_list[image_id]
                data_post_id.append(int(post_id))
                data_post_content.append(post_content)
                data_image.append(image)
                data_label.append(int(label))
                data_img_id.append(image_id)
                l = len(post_content.split())
                if l < min: min = l
                if l > max: max = l
                sen += l
            else:
                print(image_id)
                unmatched_num += 1

    me = sen / len(data_label)
    print("mean", me)
    print("min", min)
    print("max", max)

    f.close()
    print('unmatched_num', unmatched_num)

    data_dic = {'post_id': np.array(data_post_id),
                'post_content': data_post_content,
                'image': data_image,
                'label': np.array(data_label)
                }

    randnum = 42
    random.seed(randnum)
    random.shuffle(data_img_id)
    random.seed(randnum)
    random.shuffle(data_post_id)

    random.seed(randnum)
    random.shuffle(data_post_content)
    random.seed(randnum)
    random.shuffle(data_label)

    random.seed(randnum)
    random.shuffle(data_image)

    file1 = open('paired_train_230407.txt', mode='a+', encoding='UTF-8')

    print("选出的数据数量", len(data_label))
    for i in range(len(data_label)):
        file1.writelines(str(data_post_id[i]) + ' ==sep== ' + data_post_content[i] + ' ==sep== ' + data_img_id[
            i] + ' ==sep== ' + str(data_label[i]) + '\n')
    file1.close()

    pair_num = data_num - unmatched_num
    o_image = t.stack(data_image).cuda()
    label = np.array(data_label, dtype=np.int)

    return data_dic, pair_num, o_image, label


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        if self.right is None:
            residual = x
        else:
            residual = self.right(x)
        out += residual
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, numclasses=512):
        super(ResNet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.layer1 = self.make_layer(64, 128, 4)
        self.layer2 = self.make_layer(128, 256, 4, stride=2)
        self.layer3 = self.make_layer(256, 256, 6, stride=2)
        self.layer4 = self.make_layer(256, 512, 3, stride=2)

        self.fc = nn.Linear(512, numclasses)

    def make_layer(self, inchannel, outchannel, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.bert_cross = BertModel.from_pretrained('/home/jd/code/dyq/ALGM22/EANN/model/bert-base-uncased',
                                                    return_dict=False)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 256)

    def forward(self, text_ids, text_mask):
        text_info, pooled_text_info = self.bert_cross(input_ids=text_ids, attention_mask=text_mask)

        return pooled_text_info


class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()

        vgg_19 = torchvision.models.vgg19(pretrained=True)
        self.feature = vgg_19.features
        self.classifier = nn.Sequential(*list(vgg_19.classifier.children())[:-3])
        pretrained_dict = vgg_19.state_dict()
        model_dict = self.classifier.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.classifier.load_state_dict(model_dict)

    def forward(self, img):
        img = self.feature(img)
        img = img.view(img.size(0), -1)
        image = self.classifier(img)

        return image


def save_npz(txt_v, img_v, labels):
    txt_np = txt_v.detach().cpu().numpy()
    img_np = img_v.detach().cpu().numpy()
    labels = np.array(labels)

    np.savez("0407_text_train.npz", data=txt_np, label=labels)
    np.savez("0407_pimg_train.npz", data=img_np, label=labels)


def save_npz1(img_v, labels):
    img_np = img_v.detach().cpu().numpy()
    labels = np.array(labels)

    np.savez("0407_img_train.npz", data=img_np, label=labels)

if __name__ == "__main__":

    img_list = read_images(image_file_list)
    img_dic = get_imgid(image_id_path)
    f_num, r_num = select_data(original_train_data, new_train)

    train, train_num, ordered_image, label = get_data('train', img_list, img_dic)

    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    ordered_image.cuda()

    max_length = 30

    img_dataloader = DataLoader(ordered_image, batch_size=256, shuffle=False, drop_last=False)
    img_emb = []
    with t.no_grad():
        for i, batch in enumerate(img_dataloader):
            batch = batch.to(device)

            img_model = ResNet()

            img_model.to(device)
            img_vec = img_model(batch)
            img_emb.append(img_vec)
            print("iter, img_vec_shape", i, img_vec.shape)
    i_emb = t.cat(img_emb)
    print("total_shape_i_emb", i_emb.shape)

    ti = i_emb.t().cpu()
    print("shape_ti", ti.shape)
    ln = t.nn.LayerNorm(ti.shape[1], eps=1e-5, elementwise_affine=True)
    i = ln(ti)

    i_emb = i.t()
    print("fin_shape_i_emb", i_emb.shape)

    save_npz1(i_emb, label)
