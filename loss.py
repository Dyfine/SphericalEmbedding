import myutils
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

class TripletLoss(Module):
    def __init__(self, instance, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.instance = instance

    def forward(self, inputs, targets, normalized=True):
        norm_temp = inputs.norm(dim=1, p=2, keepdim=True)
        if normalized:
            inputs = inputs.div(norm_temp.expand_as(inputs))

        nB = inputs.size(0)
        idx_ = torch.arange(0, nB, dtype=torch.long)

        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(nB, nB)
        dist = dist + dist.t()
        # use squared
        dist.addmm_(1, -2, inputs, inputs.t()).clamp_(min=1e-12)

        adjacency = targets.expand(nB, nB).eq(targets.expand(nB, nB).t())
        adjacency_not = ~adjacency
        mask_ap = (adjacency.float() - torch.eye(nB).cuda()).long()
        mask_an = adjacency_not.long()

        dist_ap = (dist[mask_ap == 1]).view(-1, 1)
        dist_an = (dist[mask_an == 1]).view(nB, -1)
        dist_an = dist_an.repeat(1, self.instance - 1)
        dist_an = dist_an.view(nB * (self.instance - 1), nB - self.instance)
        num_loss = dist_an.size(0) * dist_an.size(1)

        triplet_loss = torch.sum(
            torch.max(torch.tensor(0, dtype=torch.float).cuda(), self.margin + dist_ap - dist_an)) / num_loss
        final_loss = triplet_loss * 1.0

        with torch.no_grad():
            assert normalized == True
            cos_theta = torch.mm(inputs, inputs.t())
            mask = targets.expand(nB, nB).eq(targets.expand(nB, nB).t())
            avg_ap = cos_theta[(mask.float() - torch.eye(nB).cuda()) == 1].mean()
            avg_an = cos_theta[mask.float() == 0].mean()

        return final_loss, avg_ap, avg_an

class TripletSemihardLoss(Module):
    def __init__(self, margin=0.2):
        super(TripletSemihardLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets, normalized=True):
        norm_temp = inputs.norm(dim=1, p=2, keepdim=True)
        if normalized:
            inputs = inputs.div(norm_temp.expand_as(inputs))

        nB = inputs.size(0)
        idx_ = torch.arange(0, nB, dtype=torch.long)

        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(nB, nB)
        dist = dist + dist.t()
        # use squared
        dist.addmm_(1, -2, inputs, inputs.t()).clamp_(min=1e-12)

        temp_euclidean_score = dist * 1.0

        adjacency = targets.expand(nB, nB).eq(targets.expand(nB, nB).t())
        adjacency_not = ~ adjacency

        dist_tile = dist.repeat(nB, 1)
        mask = (adjacency_not.repeat(nB, 1)) * (dist_tile > (dist.transpose(0, 1).contiguous().view(-1, 1)))
        mask_final = (mask.float().sum(dim=1, keepdim=True) > 0).view(nB, nB).transpose(0, 1)

        # negatives_outside: smallest D_an where D_an > D_ap
        temp1 = (dist_tile - dist_tile.max(dim=1, keepdim=True)[0]) * (mask.float())
        negtives_outside = temp1.min(dim=1, keepdim=True)[0] + dist_tile.max(dim=1, keepdim=True)[0]
        negtives_outside = negtives_outside.view(nB, nB).transpose(0, 1)

        # negatives_inside: largest D_an
        temp2 = (dist - dist.min(dim=1, keepdim=True)[0]) * (adjacency_not.float())
        negtives_inside = temp2.max(dim=1, keepdim=True)[0] + dist.min(dim=1, keepdim=True)[0]
        negtives_inside = negtives_inside.repeat(1, nB)

        semi_hard_negtives = torch.where(mask_final, negtives_outside, negtives_inside)

        loss_mat = self.margin + dist - semi_hard_negtives

        mask_positives = adjacency.float() - torch.eye(nB).cuda()
        mask_positives = mask_positives.detach()
        num_positives = torch.sum(mask_positives)

        triplet_loss = torch.sum(
            torch.max(torch.tensor(0, dtype=torch.float).cuda(), loss_mat * mask_positives)) / num_positives
        final_loss = triplet_loss * 1.0
        
        with torch.no_grad():
            assert normalized == True
            cos_theta = torch.mm(inputs, inputs.t())
            mask = targets.expand(nB, nB).eq(targets.expand(nB, nB).t())
            avg_ap = cos_theta[(mask.float() - torch.eye(nB).cuda()) == 1].mean()
            avg_an = cos_theta[mask.float() == 0].mean()

        return final_loss, avg_ap, avg_an

def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))

class NpairLoss(Module):
    def __init__(self):
        super(NpairLoss, self).__init__()

    def forward(self, inputs, targets, normalized=False):
        nB = inputs.size(0)

        norm_temp = inputs.norm(p=2, dim=1, keepdim=True)

        inputs_n = inputs.div(norm_temp.expand_as(inputs))
        mm_logits = torch.mm(inputs_n, inputs_n.t()).detach()
        mask = targets.expand(nB, nB).eq(targets.expand(nB, nB).t())

        cos_ap = mm_logits[(mask.float() - torch.eye(nB).float().cuda()) == 1].view(nB, -1)
        cos_an = mm_logits[mask != 1].view(nB, -1)

        avg_ap = torch.mean(cos_ap)
        avg_an = torch.mean(cos_an)

        if normalized:
            inputs = inputs.div(norm_temp.expand_as(inputs))
            inputs = inputs * 5.0

        labels = targets.view(-1).cpu().numpy()
        pids = np.unique(labels)

        anchor_idx = []
        positive_idx = []
        for i in pids:
            ap_idx = np.where(labels == i)[0]
            anchor_idx.append(ap_idx[0])
            positive_idx.append(ap_idx[1])

        anchor = inputs[anchor_idx, :]
        positive = inputs[positive_idx, :]

        batch_size = anchor.size(0)

        target = torch.from_numpy(pids).cuda()
        target = target.view(target.size(0), 1)

        target = (target == torch.transpose(target, 0, 1)).float()
        target = target / torch.sum(target, dim=1, keepdim=True).float()

        logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))

        loss_ce = cross_entropy(logit, target)
        loss = loss_ce * 1.0

        return loss, avg_ap, avg_an

class MultiSimilarityLoss(Module):
    def __init__(self):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1
        self.scale_pos = 2.0
        self.scale_neg = 40.0  

    def forward(self, feats, labels):

        norm = feats.norm(dim=1, p=2, keepdim=True)
        feats = feats.div(norm.expand_as(feats))

        labels = labels.view(-1)
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        sim_mat = torch.matmul(feats, torch.t(feats))

        epsilon = 1e-5
        loss = list()

        avg_aps = list()
        avg_ans = list()

        for i in range(batch_size):
            pos_pair_ = sim_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][labels != labels[i]]

            if len(neg_pair_) < 1 or len(pos_pair_) < 1:
                continue

            avg_aps.append(pos_pair_.mean())
            avg_ans.append(neg_pair_.mean())

            neg_pair = neg_pair_[neg_pair_ + self.margin > torch.min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < torch.max(neg_pair_)]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            print('with ms loss = 0 !')
            loss = torch.zeros([], requires_grad=True).cuda()
        else:
            loss = sum(loss) / batch_size
            loss = loss.view(-1)

        avg_ap = sum(avg_aps) / batch_size
        avg_an = sum(avg_ans) / batch_size
        
        return loss, avg_ap, avg_an

