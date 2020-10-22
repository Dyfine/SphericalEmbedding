from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
import logging
import torch
import myutils
import math
from scipy.special import comb

def NMI_F1(X, ground_truth, n_cluster):
    X = [x.cpu().numpy() for x in X]
    # list to numpy
    X = np.array(X)
    
    ground_truth = np.array(ground_truth)

    kmeans = KMeans(n_clusters=n_cluster, n_jobs=4, random_state=0).fit(X)

    logging.info('K-means done')
    nmi, f1 = compute_clutering_metric(np.asarray(kmeans.labels_), ground_truth)

    return nmi, f1

def normalize(x):
    norm = x.norm(dim=1, p=2, keepdim=True)
    x = x.div(norm.expand_as(x))
    return x

def pairwise_similarity(x, y=None):
    if y is None:
        y = x

    y = normalize(y)
    x = normalize(x)

    similarity = torch.mm(x, y.t())
    return similarity


def Recall_at_ks(sim_mat, data_name=None, query_ids=None, gallery_ids=None):
    # start_time = time.time()
    # print(start_time)
    """
    :param sim_mat:
    :param query_ids
    :param gallery_ids
    :param data

    Compute  [R@1, R@2, R@4, R@8]
    """

    ks_dict = dict()
    ks_dict['CUB'] = [1, 2, 4, 8, 16, 32]
    ks_dict['Cars'] = [1, 2, 4, 8, 16, 32]
    ks_dict['SOP'] = [1, 10, 100, 1000]
    ks_dict['Inshop'] = [1, 10, 20, 30, 40, 50]

    assert data_name in ['CUB', 'Cars', 'SOP', 'Inshop']
    k_s = ks_dict[data_name]

    sim_mat = sim_mat.cpu().numpy()
    m, n = sim_mat.shape


    gallery_ids = np.asarray(gallery_ids)
    if query_ids is None:
        query_ids = gallery_ids
    else:
        query_ids = np.asarray(query_ids)


    num_valid = np.zeros(len(k_s))
    neg_nums = np.zeros(m)
    for i in range(m):
        x = sim_mat[i]

        pos_max = np.max(x[gallery_ids == query_ids[i]])

        neg_num = np.sum(x > pos_max)
        neg_nums[i] = neg_num

    for i, k in enumerate(k_s):
        if i == 0:
            temp = np.sum(neg_nums < k)
            num_valid[i] = temp
        else:
            temp = np.sum(neg_nums < k)
            num_valid[i] = temp
    
    return num_valid / float(m)


def compute_clutering_metric(idx, item_ids):

    N = len(idx)

    # cluster centers
    centers = np.unique(idx)
    num_cluster = len(centers)

    # count the number of objects in each cluster
    count_cluster = np.zeros(num_cluster)
    for i in range(num_cluster):
        count_cluster[i] = len(np.where(idx == centers[i])[0])

    # build a mapping from item_id to item index
    keys = np.unique(item_ids)
    num_item = len(keys)
    values = range(num_item)
    item_map = dict()
    for i in range(len(keys)):
        item_map.update([(keys[i], values[i])])

    # count the number of objects of each item
    count_item = np.zeros(num_item)
    for i in range(N):
        index = item_map[item_ids[i]]
        count_item[index] = count_item[index] + 1

    # compute purity
    purity = 0
    for i in range(num_cluster):
        member = np.where(idx == centers[i])[0]
        member_ids = item_ids[member]

        count = np.zeros(num_item)
        for j in range(len(member)):
            index = item_map[member_ids[j]]
            count[index] = count[index] + 1
        purity = purity + max(count)

    # compute Normalized Mutual Information (NMI)
    count_cross = np.zeros((num_cluster, num_item))
    for i in range(N):
        index_cluster = np.where(idx[i] == centers)[0]
        index_item = item_map[item_ids[i]]
        count_cross[index_cluster, index_item] = count_cross[index_cluster, index_item] + 1

    # mutual information
    I = 0
    for k in range(num_cluster):
        for j in range(num_item):
            if count_cross[k, j] > 0:
                s = count_cross[k, j] / N * math.log(N * count_cross[k, j] / (count_cluster[k] * count_item[j]))
                I = I + s

    # entropy
    H_cluster = 0
    for k in range(num_cluster):
        s = -count_cluster[k] / N * math.log(count_cluster[k] / float(N))
        H_cluster = H_cluster + s

    H_item = 0
    for j in range(num_item):
        s = -count_item[j] / N * math.log(count_item[j] / float(N))
        H_item = H_item + s

    NMI = 2 * I / (H_cluster + H_item)

    # compute True Positive (TP) plus False Positive (FP)
    tp_fp = 0
    for k in range(num_cluster):
        if count_cluster[k] > 1:
            tp_fp = tp_fp + comb(count_cluster[k], 2)

    # compute True Positive (TP)
    tp = 0
    for k in range(num_cluster):
        member = np.where(idx == centers[k])[0]
        member_ids = item_ids[member]

        count = np.zeros(num_item)
        for j in range(len(member)):
            index = item_map[member_ids[j]]
            count[index] = count[index] + 1

        for i in range(num_item):
            if count[i] > 1:
                tp = tp + comb(count[i], 2)

    # False Positive (FP)
    fp = tp_fp - tp

    # compute False Negative (FN)
    count = 0
    for j in range(num_item):
        if count_item[j] > 1:
            count = count + comb(count_item[j], 2)

    fn = count - tp

    # compute F measure
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    beta = 1
    F = (beta*beta + 1) * P * R / (beta*beta * P + R)

    return NMI, F

