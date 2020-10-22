import myutils
import os
import torch
import numpy as np
import os.path as osp
from PIL import Image
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import re

class MSBaseDataSet(torch.utils.data.Dataset):
    """
    Basic Dataset read image path from img_source
    img_source: list of img_path and label
    """
    def __init__(self, conf, img_source, transform=None, mode="RGB"):
        self.mode = mode

        self.root = os.path.dirname(img_source)
        assert os.path.exists(img_source), f"{img_source} NOT found."
        self.img_source = img_source

        self.label_list = list()
        self.path_list = list()
        self._load_data()
        self.label_index_dict = self._build_label_index_dict()

        self.num_cls = len(self.label_index_dict.keys())
        self.num_train = len(self.label_list)

        self.transform = transform

    def __len__(self):
        return len(self.label_list)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"| Dataset Info |datasize: {self.__len__()}|num_labels: {len(set(self.label_list))}|"

    def _load_data(self):
        with open(self.img_source, 'r') as f:
            for line in f:
                _path, _label = re.split(r",| ", line.strip())
                self.path_list.append(_path)
                self.label_list.append(_label)

    def _build_label_index_dict(self):
        index_dict = defaultdict(list)
        for i, label in enumerate(self.label_list):
            index_dict[label].append(i)
        return index_dict

    def read_image(self, img_path, mode='RGB'):
        """Keep reading image until succeed.
        This can avoid IOError incurred by heavy IO process."""
        got_img = False
        if not osp.exists(img_path):
            raise IOError(f"{img_path} does not exist")
        while not got_img:
            try:
                img = Image.open(img_path).convert("RGB")
                if mode == "BGR":
                    r, g, b = img.split()
                    img = Image.merge("RGB", (b, g, r))
                got_img = True
            except IOError:
                print(f"IOError incurred when reading '{img_path}'. Will redo.")
                pass
        return img

    def __getitem__(self, index):
        path = self.path_list[index]
        img_path = os.path.join(self.root, path)
        label = self.label_list[index]

        img = self.read_image(img_path, mode=self.mode)

        if self.transform is not None:
            img = self.transform(img)
        return {'image': img, 'label': int(label), 'index': index}


class RandomIdSampler(Sampler):
    def __init__(self, conf, label_index_dict):
        self.label_index_dict = label_index_dict
        self.num_train = 0
        for k in self.label_index_dict.keys():
            self.num_train += len(self.label_index_dict[k])

        self.num_instances = conf.instances
        self.batch_size = conf.batch_size
        assert self.batch_size % self.num_instances == 0
        self.num_pids_per_batch = self.batch_size // self.num_instances

        self.ids = list(self.label_index_dict.keys())

        self.length = self.num_train//self.batch_size * self.batch_size
        self.conf = conf

    def __len__(self):
        return self.length

    def get_batch_ids(self):
        pids = []

        pids = np.random.choice(self.ids,
                                size=self.num_pids_per_batch,
                                replace=False)
        return pids

    def get_batch_idxs(self):
        pids = self.get_batch_ids()

        inds = []
        cnt = 0
        for pid in pids:
            index_list = self.label_index_dict[pid]
            if len(index_list) >= self.num_instances:
                t = np.random.choice(index_list, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(index_list, size=self.num_instances, replace=True)
            t_ = t.astype(int)
            for ind in t:
                yield ind
                cnt += 1
                if cnt == self.batch_size:
                    break
            if cnt == self.batch_size:
                break

    def __iter__(self):
        cnt = 0
        while cnt < len(self):
            for ind in self.get_batch_idxs():
                cnt += 1
                yield ind


