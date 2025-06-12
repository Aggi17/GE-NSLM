# -*- coding: utf-8 -*-

"""
Utility functions for PLM checkers
@Author     : Jiangjie Chen
@Time       : 2020/10/15 16:10
@Contact    : jjchen19@fudan.edu.cn
@Description: Helper functions for neural-symbolic reasoning
"""

import torch
import random
import torch.nn.functional as F
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks"""

    def __init__(self, hidden_size, num_labels, hidden_dropout_prob=0.2):
        super().__init__()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels, bias=False)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def temperature_annealing(tau, step):
    """Temperature annealing for Gumbel-Softmax"""
    if tau == 0.:
        tau = 10. if step % 5 == 0 else 1.
    return tau


def soft_logic1(y_i, tnorm='product'):
    """Soft logic for single z variable"""
    p0 = y_i[:, 0]  # b x 3
    p1 = y_i[:, 1]  # b x 3
    p2 = torch.zeros_like(p0)

    logical_prob = torch.stack([p0, p1, p2], dim=-1)
    return logical_prob  # b x 3


def get_label_embeddings(labels, label_embedding):
    """Get label embeddings for variational inference"""
    emb = torch.einsum('oi,bo->bi', label_embedding, labels)
    return emb


def soft_logic(y_i, tnorm='product'):
    """Soft logic implementation for multi-variable reasoning"""
    _p00 = y_i[:, :, 0]  # b x 3
    _p11 = y_i[:, :, 1]  # b x 3

    if tnorm == 'product':
        p_0 = torch.exp(torch.log(_p00).sum(1))
        p_2 = torch.exp(torch.log(_p11).sum(1))
    elif tnorm == 'godel':
        p_rel = _rel.min(-1).values
    elif tnorm == 'lukas':
        raise NotImplementedError(tnorm)
    else:
        raise NotImplementedError(tnorm)

    p_1 = 1 - p_0 - p_2
    p_0 = torch.max(p_0, torch.zeros_like(p_0))
    p_1 = torch.max(p_1, torch.zeros_like(p_1))
    p_2 = torch.max(p_2, torch.zeros_like(p_2))

    logical_prob = torch.stack([p_0, p_1, p_2], dim=-1)
    assert torch.lt(logical_prob, 0).to(torch.int).sum().tolist() == 0, \
        (logical_prob, _p00, _p11)
    return logical_prob  # b x 3


def build_pseudo_labels(labels, m_attn):
    """Build pseudo labels for training"""
    mask = torch.gt(m_attn, 1e-16).to(torch.int)
    sup_label = torch.tensor(2).to(labels)
    nei_label = torch.tensor(1).to(labels)
    ref_label = torch.tensor(0).to(labels)
    pseudo_labels = []
    
    for idx, label in enumerate(labels):
        mm = mask[idx].sum(0)
        if label == 2:  # SUPPORTS
            pseudo_label = F.one_hot(sup_label.repeat(mask.size(1)), num_classes=3).to(torch.float)

        elif label == 0:  # REFUTES
            num_samples = magic_proportion(mm)
            ids = torch.topk(m_attn[idx], k=num_samples).indices
            pseudo_label = []
            for i in range(mask.size(1)):
                if i >= mm:
                    _label = torch.tensor([1 / 3, 1 / 3, 1 / 3]).to(labels)
                elif i in ids:
                    _label = F.one_hot(ref_label, num_classes=3).to(torch.float)
                else:
                    if random.random() > 0.5:
                        _label = torch.tensor([0., 0., 1.]).to(labels)
                    else:
                        _label = torch.tensor([0., 1., 0.]).to(labels)
                pseudo_label.append(_label)
            pseudo_label = torch.stack(pseudo_label)

        else:  # NEI
            num_samples = magic_proportion(mm)
            ids = torch.topk(m_attn[idx], k=num_samples).indices
            pseudo_label = sup_label.repeat(mask.size(1))
            pseudo_label[ids] = nei_label
            pseudo_label = F.one_hot(pseudo_label, num_classes=3).to(torch.float)

        pseudo_labels.append(pseudo_label)
    return torch.stack(pseudo_labels)


def magic_proportion(m, magic_n=5):
    """Calculate proportion for pseudo label generation"""
    return m // magic_n + 1


def sequence_mask(lengths, max_len=None):
    """Create boolean mask from sequence lengths"""
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def collapse_w_mask(inputs, mask):
    """Collapse sequence with mask"""
    hidden = inputs.size(-1)
    output = inputs * mask.unsqueeze(-1).repeat((1, 1, hidden))
    output = output.sum(-2)
    output /= (mask.sum(-1) + 1e-6).unsqueeze(-1).repeat((1, hidden))
    return output


def parse_ce_outputs(ce_seq_output, ce_lengths):
    """Parse claim-evidence outputs"""
    if ce_lengths.max() == 0:
        b, L1, h = ce_seq_output.size()
        return torch.zeros([b, h]).cuda(), torch.zeros([b, h]).cuda()
    masks = []
    for mask_id in range(1, ce_lengths.max() + 1):
        _m = torch.ones_like(ce_lengths) * mask_id
        mask = _m.eq(ce_lengths).to(torch.int)
        masks.append(mask)
    c_output = collapse_w_mask(ce_seq_output, masks[0])
    e_output = torch.stack([collapse_w_mask(ce_seq_output, m)
                            for m in masks[1:]]).mean(0)
    return c_output, e_output


def parse_qa_outputs(qa_seq_output, qa_lengths, k):
    """Parse question-answer outputs"""
    b, L2, h = qa_seq_output.size()
    if qa_lengths.max() == 0:
        return torch.zeros([b, h]).cuda(), torch.zeros([b, h]).cuda(), \
               torch.zeros([k, b, h]).cuda()

    masks = []
    for mask_id in range(1, qa_lengths.max() + 1):
        _m = torch.ones_like(qa_lengths) * mask_id
        mask = _m.eq(qa_lengths).to(torch.int)
        masks.append(mask)

    q_output = collapse_w_mask(qa_seq_output, masks[0])
    a_output = collapse_w_mask(qa_seq_output, masks[1])
    k_cand_output = [collapse_w_mask(qa_seq_output, m)
                     for m in masks[2:2 + k]]
    for i in range(k - len(k_cand_output)):
        k_cand_output.append(torch.zeros([b, h]).cuda())
    k_cand_output = torch.stack(k_cand_output, dim=0)

    return q_output, a_output, k_cand_output


def attention_mask_to_mask(attention_mask):
    '''
    :param attention_mask: b x m x L
    :return: b x m
    '''
    mask = torch.gt(attention_mask.sum(-1), 0).to(torch.int).sum(-1)  # (b,)
    mask = sequence_mask(mask, max_len=attention_mask.size(1)).to(torch.int)  # (b, m)
    return mask


if __name__ == "__main__":
    y = torch.tensor([[[0.3, 0.5, 0.2], [0.1, 0.4, 0.5]]])
    mask = torch.tensor([1, 1])
    s = soft_logic(y, mask)
    print(s)