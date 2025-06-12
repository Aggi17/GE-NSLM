# -*- coding: utf-8 -*-

"""
RoBERTa-based fake news detection model
@Author     : Jiangjie Chen
@Time       : 2020/8/18 14:40
@Contact    : jjchen19@fudan.edu.cn
@Description: Neuro-symbolic approach for fake news detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tfn_model import TFN
from transformers import RobertaModel, BertPreTrainedModel, RobertaConfig
from .checker_utils import attention_mask_to_mask, ClassificationHead, soft_logic, soft_logic1, build_pseudo_labels, \
    get_label_embeddings, temperature_annealing

import re
from math import log

from PIL import Image
import numpy as np

import random
import torchvision

from .InceptionV3 import GoogLeNet
from .encoding_img import vgg, ResNet

# Import transformers for Chinese RoBERTa
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class RobertaChecker(BertPreTrainedModel):
    """RoBERTa-based model for fake news detection with neuro-symbolic reasoning"""
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config, flag_z_IM, flag_z_CI, flag_z_IR, flag_z_OT, flag_img_des, flag_wor_kno, hs=128, share_hs=128, logic_lambda=0.0, prior='nli', temperature=1):
        super().__init__(config)
        self.flag_IM = flag_z_IM
        self.flag_CI = flag_z_CI
        self.flag_IR = flag_z_IR
        self.flag_OT = flag_z_OT

        self.flag_img_des = flag_img_des
        self.flag_wor_kno = flag_wor_kno

        self.ynum_labels = config.num_labels  # 3 labels
        self.znum_labels = 2
        self.hidden_size = config.hidden_size  # RoBERTa hidden size (1024)
        self.roberta = RobertaModel(config)

        # Image encoders
        self.resnet = ResNet()
        self.vgg = vgg()
        self.inv3 = GoogLeNet()
        self.myhs = hs
        self.share_hs = share_hs

        # Tensor Fusion Network for multi-modal fusion
        self.tensor_fusion = TFN((self.myhs, self.myhs, 1024), (self.myhs//2, self.myhs//2, 512), self.myhs, (0.3, 0.3, 0.3, 0.3), self.myhs//2)
        self.tensor_fusion_pre = TFN((self.myhs, 512, 1024), (self.myhs//2, 256, 1024//2), self.myhs, (0.3, 0.3, 0.3, 0.3), self.myhs//2)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self._lambda = logic_lambda
        self.prior = prior
        self.temperature = temperature
        self._step = 0

        # Linear transformations
        self.linear_1 = nn.Linear(50265, self.hidden_size)
        self.linear_2 = nn.Linear(50265, self.hidden_size)
        self.linear_3 = nn.Linear(50265, self.hidden_size)

        # Feature projection layers
        self.linear_et = nn.Linear(self.hidden_size, self.myhs)
        self.linear_eo = nn.Linear(self.hidden_size, self.myhs)
        self.linear_ec = nn.Linear(self.hidden_size, self.myhs)

        self.linear_ev = nn.Linear(512, self.myhs)
        self.linear_ep = nn.Linear(1000, self.myhs)
        self.linear_e4 = nn.Linear(self.myhs*3, self.myhs)

        self.linear_ez = nn.Linear(2, self.myhs // 2)
        self.linear_shared1 = nn.Linear(self.myhs, self.myhs//2)
        self.linear_shared2 = nn.Linear(self.myhs, self.myhs//2)

        # Bilinear layers for feature interaction
        self.bilinear1 = nn.Bilinear(self.myhs, self.myhs, self.myhs, bias=True)
        self.bilinear2 = nn.Bilinear(self.myhs, self.myhs, self.myhs, bias=True)
        self.bilinear3 = nn.Bilinear(self.myhs, self.myhs, self.myhs, bias=True)

        self.var_hidden_size = self.myhs

        if self.flag_IM or self.flag_CI or self.flag_IR or self.flag_OT:
            z_hid_size = self.znum_labels * 3
        else:
            z_hid_size = self.znum_labels * 4

        # Variational inference layers
        self.linear_P_theta = nn.Linear(self.myhs * 2 + z_hid_size, self.var_hidden_size)

        y_hid_size = self.var_hidden_size
        self.linear_Q_phi = nn.Linear(self.myhs + y_hid_size, self.var_hidden_size)
        self.linear_Q_phi_1 = nn.Linear(self.myhs + y_hid_size, self.var_hidden_size)
        self.linear_Q_phi_3 = nn.Linear(self.myhs + y_hid_size, self.var_hidden_size)

        self.linear_Q_phi_global_1 = nn.Linear(3 * self.myhs + y_hid_size, self.var_hidden_size)
        self.linear_Q_phi_global_2 = nn.Linear(3 * self.myhs + y_hid_size, self.var_hidden_size)
        self.linear_Q_phi_global_3 = nn.Linear(3 * self.myhs + y_hid_size, self.var_hidden_size)
        self.linear_Q_phi_global_4 = nn.Linear(3 * self.myhs + y_hid_size, self.var_hidden_size)

        # Classification heads
        self.classifier = ClassificationHead(self.var_hidden_size, self.ynum_labels, config.hidden_dropout_prob)
        self.z_clf = ClassificationHead(self.var_hidden_size, self.znum_labels, config.hidden_dropout_prob)
        self.z_clf_1 = ClassificationHead(self.var_hidden_size, self.znum_labels, config.hidden_dropout_prob)
        self.z_clf_2 = ClassificationHead(self.var_hidden_size, self.znum_labels, config.hidden_dropout_prob)
        self.z_clf_3 = ClassificationHead(self.var_hidden_size, self.znum_labels, config.hidden_dropout_prob)
        self.z_clf_4 = ClassificationHead(self.var_hidden_size, self.znum_labels, config.hidden_dropout_prob)

        # Entity memory components
        self.ent_dim_in = hs
        self.ent_dim_out = hs
        self.ent_mem_hops = 1

        if self.ent_mem_hops > 1:
            self.Wqent_hop = nn.Linear(self.ent_dim_out, self.ent_dim_out)
        self.W_ent_c = nn.Linear(self.ent_dim_in, self.ent_dim_out)
        self.W_ent_a = nn.Linear(self.ent_dim_in, self.ent_dim_out)
        self.bn_entMem = nn.BatchNorm1d(num_features=self.ent_dim_out)

        self.init_weights()
        self.soft = nn.Softmax(dim=-1)

    def forward(self, claim_input_ids, claim_attention_mask, claim_token_type_ids,
                old_claim_input_ids, old_claim_attention_mask,
                qa_input_ids_list, qa_attention_mask_list, qa_token_type_ids_list,
                cap_input_ids_list, cap_attention_mask_list, cap_token_type_ids_list,
                img_input, nli_labels=None, labels=None):
        """Forward pass through the model"""
        self._step += 1
        _zero = torch.tensor(0.).to(claim_input_ids.device)

        if self.flag_IR and self.flag_OT:
            # Extract text features
            e_t = self.roberta(claim_input_ids, attention_mask=claim_attention_mask)[1]
            e_t = self.linear_et(e_t)

            # Extract image features
            e_v = self.resnet(img_input)  # Semantic features
            e_v = self.linear_ev(e_v)

            e_p = self.inv3(img_input)  # Pattern features
            e_p = self.linear_ep(e_p)
            
            # Feature interaction
            s1 = self.bilinear1(e_t, e_v)
            e1 = s1

            s2 = self.bilinear2(e_p, e_p)
            e2 = s2

            e1 = torch.nn.functional.normalize(e1, p=2, dim=1)
            e2 = torch.nn.functional.normalize(e2, p=2, dim=1)

        else:
            if self.flag_wor_kno:  # Use old text for ablation
                e_tn = self.roberta(old_claim_input_ids, attention_mask=old_claim_attention_mask)[0]
            else:
                e_tn = self.roberta(claim_input_ids, attention_mask=claim_attention_mask)[0]

            e_t = self.roberta(claim_input_ids, attention_mask=claim_attention_mask)[1]
            e_t = self.linear_et(e_t)

            e_c = self.roberta(cap_input_ids_list, attention_mask=cap_attention_mask_list)[1]
            e_c = self.linear_ec(e_c)
            e_c = self.linear_shared1(e_c)

            e_v = self.resnet(img_input)  # img_v = self.vgg(img_input)   e_v  512 表示从语义角度提取的向量 注意维度后面标好
            e_v = self.linear_ev(e_v)  # 没有img_des的话，直接是resnet 出来的当ev
            if not self.flag_img_des:  # 有img,拼
                e_v = self.linear_shared1(e_v)
                e_v = torch.cat([e_v, e_c], dim=1)

            e_p = self.inv3(img_input)  # e_p从模式角度提取图片特征  b * h_p 1000
            e_p = self.linear_ep(e_p)

            e_o = self.roberta(qa_input_ids_list, attention_mask=qa_attention_mask_list)[1]
            e_o = self.linear_eo(e_o)

            s1 = self.bilinear1(e_t, e_v)  # 现在是256维的去计算   ci
            e1 = s1

            s2 = self.bilinear2(e_p, e_p)
            e2 = s2

            s3 = self.bilinear3(e_t, e_o)
            e3 = s3

            e4 = self.tensor_fusion(e_o, e_v, e_tn) #tfn

            e1 = torch.nn.functional.normalize(e1, p=2, dim=1)
            e2 = torch.nn.functional.normalize(e2, p=2, dim=1)
            e3 = torch.nn.functional.normalize(e3, p=2, dim=1)
            e4 = torch.nn.functional.normalize(e4, p=2, dim=1)

        e_global = torch.cat([e_t, e_v], dim=1)  # b* 2myhs

        neg_elbo, loss, logic_loss = _zero, _zero, _zero

        if labels is not None:
            # Training
            labels_onehot = F.one_hot(labels, num_classes=self.ynum_labels).to(torch.float)
            y_star_emb = get_label_embeddings(labels_onehot,
                                              self.classifier.out_proj.weight)  # 结果是 b x h`（var_size） 这里可能是对label转换，编码成向量的形式,后面一项是权重矩阵 3*h`

            if self.flag_IM: #without e2
                z = self.Q_phi3(e1, e3, e4, e_global, y_star_emb)  # alb
            elif self.flag_CI: # without e1
                z = self.Q_phi3(e2, e3, e4, e_global, y_star_emb)
            elif self.flag_IR and self.flag_OT:  # feall 先跑一个
                z = self.Q_phi2(e1, e2, e_global, y_star_emb)
            elif self.flag_IR: # without e3
                z = self.Q_phi3(e1, e2, e4, e_global, y_star_emb)
            elif self.flag_OT: # without e4
                z = self.Q_phi3(e1, e2, e3, e_global, y_star_emb)

            else:
                z = self.Q_phi4(e1, e2, e3, e4, e_global, y_star_emb) # # b*4*2

            z_softmax = z.softmax(-1)

            z_gumbel = F.gumbel_softmax(z, tau=temperature_annealing(self.temperature, self._step),
                                        dim=-1, hard=True)  # b  *3 *2  对离散随机变量采样的过程 这个是那个重参数化采样，维度不可变
            y = self.P_theta(e_global, z_gumbel)  # b*2  维度会自己变的，z_hid_size 已经定义了
            y_softmax = y.softmax(-1)

            y_z = soft_logic(z_softmax)

            logic_loss = F.kl_div(y.log_softmax(-1), y_z)  # 原来

            elbo_neg_p_log = F.cross_entropy(y.view(-1, self.ynum_labels), labels.view(
                -1))  # y是P_theta的输出，相当于重建的数据  labels 应该就是真实标签啊 .view(-1) 是变成一维张量 onehot
            if self.prior == 'nli':  # 3*2
                prior = nli_labels
            elif self.prior == 'uniform':
                prior = torch.full((y_z.size(0), 4, self.znum_labels), 1 / self.znum_labels).to(y.device)
            elif self.prior == 'random':
                if self.flag_IR and self.flag_OT:
                    prior = torch.rand([y_z.size(0), 2, self.znum_labels]).to(nli_labels)
                elif self.flag_IM or self.flag_CI or self.flag_IR or self.flag_OT:
                    prior = torch.rand([y_z.size(0), 3, self.znum_labels]).to(nli_labels)
                else:
                    prior = torch.rand([y_z.size(0), 4, self.znum_labels]).to(nli_labels)
                prior = prior.softmax(dim=-1)

            else:
                raise NotImplementedError(self.prior)

            elbo_kl = F.kl_div(z_softmax.log(), prior)  # 先验只知道一些采样值也可？是的 就是这样 第一个参数传入的是一个对数概率矩阵，第二个参数传入的是概率矩阵。

            neg_elbo = elbo_kl + elbo_neg_p_log  # （3）式

            loss = (1 - abs(self._lambda)) * neg_elbo + abs(self._lambda) * logic_loss  # （5）式
        else:
            if self.prior == 'nli':
                z = nli_labels
            elif self.prior == 'uniform':
                z = torch.full((nli_labels.size(0), 4, self.znum_labels), 1 / self.znum_labels).to(nli_labels.device)
            elif self.prior == 'random':

                if self.flag_IR and self.flag_OT:
                    z = torch.rand([nli_labels.size(0), 2, self.znum_labels]).to(nli_labels)  # 随
                elif self.flag_IM or self.flag_CI or self.flag_IR or self.flag_OT:
                    z = torch.rand([nli_labels.size(0), 3, self.znum_labels]).to(nli_labels)  # 随
                else:
                    z = torch.rand([nli_labels.size(0), 4, self.znum_labels]).to(nli_labels)
            z_softmax = z.softmax(-1)

            for i in range(3):  # N = 3? N 啥  decoding 迭代3次？
                z = z_softmax.argmax(-1)  # 先从先验里面取样z
                z = F.one_hot(z, num_classes=2).to(torch.float)

                y = self.P_theta(e_global, z)  # 这里y 输出有负数
                y = y.softmax(-1)  #

                y_emb = get_label_embeddings(y, self.classifier.out_proj.weight)

                if self.flag_IM:
                    z = self.Q_phi3(e1, e3, e4, e_global, y_emb) # alb
                elif self.flag_CI:
                    z = self.Q_phi3(e2, e3, e4, e_global, y_emb)
                elif self.flag_IR and self.flag_OT:  # feall 先跑一个
                    z = self.Q_phi2(e1, e2, e_global, y_emb)
                elif self.flag_IR:
                    z = self.Q_phi3(e1, e2, e4, e_global, y_emb)
                elif self.flag_OT:
                    z = self.Q_phi3(e1, e2, e3, e_global, y_emb)
                else:
                    z = self.Q_phi4(e1, e2, e3, e4, e_global, y_emb)  # # b*4*2

                z_softmax = z.softmax(-1)

            y_softmax = y.softmax(-1)

        return (loss, (neg_elbo, logic_loss), y_softmax, z_softmax)  # batch first

    def Q_phi4(self, e1, e2, e3, e4, X_global,  y):
        '''
        :param e1,2,3: b x self.myhs 256
        :param y: b x h' ([8, 256])
        :return: b x 4 (4个角度) x 2 (real,fake)
        '''
        z_hidden_1 = self.linear_Q_phi_global_1(torch.cat([y, e1, X_global], dim=-1))  # 降到var_size
        z_hidden_1 = F.tanh(z_hidden_1)
        z_1 = self.z_clf_1(z_hidden_1)  # b * 2

        z_hidden_2 = self.linear_Q_phi_global_2(torch.cat([y, e2, X_global], dim=-1))  # 降到var_size
        z_hidden_2 = F.tanh(z_hidden_2)
        z_2 = self.z_clf_2(z_hidden_2)  # 2

        z_hidden_3 = self.linear_Q_phi_global_3(torch.cat([y, e3, X_global], dim=-1))  # 降到var_size
        z_hidden_3 = F.tanh(z_hidden_3)
        z_3 = self.z_clf_3(z_hidden_3)  # 2

        z_hidden_4 = self.linear_Q_phi_global_4(torch.cat([y, e4, X_global], dim=-1))  # (8x769 and 1024x256)
        z_hidden_4 = F.tanh(z_hidden_4)
        z_4 = self.z_clf_4(z_hidden_4)  # 2

        z = torch.cat([z_1.unsqueeze(1), z_2.unsqueeze(1), z_3.unsqueeze(1), z_4.unsqueeze(1)], dim=1)  # b*4*2
        return z

    def Q_phi3(self, e1, e2, e3, X_global, y):
        '''
        :param e1,2,3: b x self.myhs 256
        :param y: b x h' ([8, 256])
        :return: b x 3 (3个角度) x 2 (real,fake)
        '''
        z_hidden_1 = self.linear_Q_phi_global_1(torch.cat([y, e1, X_global], dim=-1))  # 降到var_size
        z_hidden_1 = F.tanh(z_hidden_1)
        z_1 = self.z_clf_1(z_hidden_1)  # b * 2

        z_hidden_2 = self.linear_Q_phi_global_2(torch.cat([y, e2, X_global], dim=-1))  # 降到var_size
        z_hidden_2 = F.tanh(z_hidden_2)
        z_2 = self.z_clf_2(z_hidden_2)  # 2

        z_hidden_3 = self.linear_Q_phi_global_3(torch.cat([y, e3, X_global], dim=-1))  # 降到var_size
        z_hidden_3 = F.tanh(z_hidden_3)
        z_3 = self.z_clf_3(z_hidden_3)  # 2

        z = torch.cat([z_1.unsqueeze(1), z_2.unsqueeze(1), z_3.unsqueeze(1)], dim=1)  # b*3*2
        return z

    def Q_phi2(self, e1, e2, X_global, y):
        '''
        :param e1,2,3: b x self.myhs 256
        :param y: b x h' ([8, 256])
        :return: b x 3 (3个角度) x 2 (real,fake)
        '''
        z_hidden_1 = self.linear_Q_phi_global_1(torch.cat([y, e1, X_global], dim=-1))  # 降到var_size
        z_hidden_1 = F.tanh(z_hidden_1)
        z_1 = self.z_clf_1(z_hidden_1)  # b * 2

        z_hidden_2 = self.linear_Q_phi_global_2(torch.cat([y, e2, X_global], dim=-1))  # 降到var_size
        z_hidden_2 = F.tanh(z_hidden_2)
        z_2 = self.z_clf_2(z_hidden_2)  # 2

        z = torch.cat([z_1.unsqueeze(1), z_2.unsqueeze(1)], dim=1)  # b*2*2
        return z

    def Q_phi_1(self, e1, y):
        '''
        :param e1,: b x self.myhs
        :param y: b x h'
        :return: b x 1 (1个角度) x 2 (real,fake)
        '''
        z_hidden = self.linear_Q_phi_1(torch.cat([y, e1], dim=-1))  # 降到var_size
        z_hidden = F.tanh(z_hidden)
        z = self.z_clf_1(z_hidden)
        return z

    def P_theta(self, X_global, z):  # e_global, z_gumbel
        '''
        X, z => y*
        :param X_global: b x 2myhs
        :param z: b x 3 x 2
        :return: b x 2
        '''
        b = z.size(0)
        _logits = torch.cat([X_global, z.reshape(b, -1)], dim=-1)  # 这边z reshape 了一下
        _logits = self.dropout(_logits)
        _logits = self.linear_P_theta(_logits)
        _logits = torch.tanh(_logits)

        y = self.classifier(_logits)
        return y

    def generic_memory(self, query_proj, results, mem_a_weights, mem_c_weights, bn_mem, mem_hops=1,
                       query_hop_weight=None):
        u = query_proj
        for i in range(0, mem_hops):
            u = F.dropout(u, self.pdrop_mem)

            mem_a = F.relu(mem_a_weights(results))
            mem_a = F.dropout(mem_a, self.pdrop_mem)

            P = F.softmax(torch.sum(mem_a * u.unsqueeze(1), 2), dim=1)
            mem_c = F.relu(mem_c_weights(results))
            mem_c = F.dropout(mem_c, self.pdrop_mem)

            mem_out = torch.sum(P.unsqueeze(2).expand_as(mem_c) * mem_c, 1)
            mem_out = mem_out + u
            u = mem_out
        mem_out = bn_mem(mem_out)

        return mem_out







