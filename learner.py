import myutils
import os
import torch
from loss import NpairLoss, TripletSemihardLoss, TripletLoss, MultiSimilarityLoss 
import logging
import numpy as np
from models.bninception import BNInception

from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import Sampler
from datetime import datetime
from evaluation import NMI_F1, pairwise_similarity, Recall_at_ks
from data_engine import MSBaseDataSet, RandomIdSampler

def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')

class metric_learner(object):
    def __init__(self, conf, inference=False):

        logging.info(f'metric learner use {conf}')
        self.model = torch.nn.DataParallel(BNInception()).cuda()
        logging.info(f'model generated')

        if not inference:

            if conf.use_dataset == 'CUB':
                self.dataset = MSBaseDataSet(conf, './datasets/CUB_200_2011/cub_train.txt',
                                           transform=conf.transform_dict['rand-crop'], mode='RGB')
            elif conf.use_dataset == 'Cars':
                self.dataset = MSBaseDataSet(conf, './datasets/CARS196/cars_train.txt',
                                             transform=conf.transform_dict['rand-crop'], mode='RGB')
            elif conf.use_dataset == 'SOP':
                self.dataset = MSBaseDataSet(conf, './datasets/SOP/sop_train.txt',
                                             transform=conf.transform_dict['rand-crop'], mode='RGB')
            elif conf.use_dataset == 'Inshop':
                self.dataset = MSBaseDataSet(conf, './datasets/Inshop/inshop_train.txt',
                                             transform=conf.transform_dict['rand-crop'], mode='RGB')

            self.loader = DataLoader(
                self.dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,
                shuffle=False, sampler=RandomIdSampler(conf, self.dataset.label_index_dict), drop_last=True,
                pin_memory=True,
            )

            self.class_num = self.dataset.num_cls
            self.img_num = self.dataset.num_train

            myutils.mkdir_p(conf.log_path, delete=True)
            self.writer = SummaryWriter(str(conf.log_path))
            self.step = 0
            
            self.head_npair = NpairLoss().to(conf.device)
            self.head_semih_triplet = TripletSemihardLoss().to(conf.device)
            self.head_triplet = TripletLoss(instance=conf.instances).to(conf.device)
            self.head_multisimiloss = MultiSimilarityLoss().to(conf.device)
            logging.info('model heads generated')

            backbone_bn_para, backbone_wo_bn_para = [
                [p for k, p in self.model.named_parameters() if
                 ('bn' in k) == is_bn and ('head' in k) == False] for is_bn in [True, False]]

            head_bn_para, head_wo_bn_para = [
                [p for k, p in self.model.module.head.named_parameters() if
                 ('bn' in k) == is_bn] for is_bn in [True, False]]

            self.optimizer = optim.Adam([
                {'params': backbone_bn_para if conf.freeze_bn==False else [], 'lr': conf.lr_p},
                {'params': backbone_wo_bn_para, 'weight_decay': conf.weight_decay, 'lr': conf.lr_p},
                {'params': head_bn_para, 'lr': conf.lr},
                {'params': head_wo_bn_para, 'weight_decay': conf.weight_decay, 'lr': conf.lr},
            ])

            logging.info(f'{self.optimizer}, optimizers generated')

            if conf.use_dataset=='CUB' or conf.use_dataset=='Cars':
                self.board_loss_every = 20  
                self.evaluate_every = 100
                self.save_every = 1000
            elif conf.use_dataset=='Inshop':
                self.board_loss_every = 20  
                self.evaluate_every = 200
                self.save_every = 2000
            else:
                self.board_loss_every = 20  
                self.evaluate_every = 500
                self.save_every = 2000


    def train(self, conf):
        self.model.train()
        self.train_with_fixed_bn(conf)

        myutils.timer.since_last_check('start train')
        data_time = myutils.AverageMeter(20)
        loss_time = myutils.AverageMeter(20)
        loss_meter = myutils.AverageMeter(20)

        self.step = conf.start_step
        
        if self.step == 0 and conf.start_eval:
            nmi, f1, recall_ks = self.test(conf)
            self.writer.add_scalar('{}/test_nmi'.format(conf.use_dataset), nmi, self.step)
            self.writer.add_scalar('{}/test_f1'.format(conf.use_dataset), f1, self.step)
            self.writer.add_scalar('{}/test_recall_at_1'.format(conf.use_dataset), recall_ks[0], self.step)
            logging.info(f'test on {conf.use_dataset}: nmi is {nmi}, f1 is {f1}, recalls are {recall_ks[0]}, {recall_ks[1]}, {recall_ks[2]}, {recall_ks[3:]} ')

            nmi, f1, recall_ks = self.validate(conf)
            self.writer.add_scalar('{}/train_nmi'.format(conf.use_dataset), nmi, self.step)
            self.writer.add_scalar('{}/train_f1'.format(conf.use_dataset), f1, self.step)
            self.writer.add_scalar('{}/train_recall_at_1'.format(conf.use_dataset), recall_ks[0], self.step)
            logging.info(f'val on {conf.use_dataset}: nmi is {nmi}, f1 is {f1}, recall_at_1 is {recall_ks[0]} ')

            self.train_with_fixed_bn(conf)


        while self.step < conf.steps:
            
            loader_enum = enumerate(self.loader)
            while True:
                if self.step > conf.steps:
                    break
                try:
                    ind_data, data = loader_enum.__next__()
                except StopIteration as e:
                    logging.info(f'one epoch finish {e} {ind_data}')
                    break
                data_time.update(myutils.timer.since_last_check(verbose=False))

                if self.step == conf.step_milestones[0]:
                    self.schedule_lr(conf)
                if self.step == conf.step_milestones[1]:
                    self.schedule_lr(conf)
                if self.step == conf.step_milestones[2]:
                    self.schedule_lr(conf)

                imgs = data['image'].to(conf.device)
                labels = data['label'].to(conf.device)
                index = data['index']

                self.optimizer.zero_grad()

                fea = self.model(imgs, normalized=False)

                fea_norm = fea.norm(p=2, dim=1)
                norm_mean = fea_norm.mean()
                norm_var = ((fea_norm - norm_mean) ** 2).mean()


                if self.step==0:
                    self.record_norm_mean = norm_mean.detach()
                else:
                    self.record_norm_mean = (1 - conf.norm_momentum) * self.record_norm_mean + \
                        conf.norm_momentum * norm_mean.detach()
                

                if conf.use_loss == 'triplet':
                    loss, avg_ap, avg_an = self.head_triplet(fea, labels, normalized=True)
                elif conf.use_loss == 'n-npair':
                    loss, avg_ap, avg_an = self.head_npair(fea, labels, normalized=True)
                elif conf.use_loss == 'semihtriplet':
                    loss, avg_ap, avg_an = self.head_semih_triplet(fea, labels, normalized=True)
                elif conf.use_loss == 'ms':
                    loss, avg_ap, avg_an = self.head_multisimiloss(fea, labels)

                
                
                loss_sec = ((fea_norm - self.record_norm_mean) ** 2).mean()
                loss_l2reg = (fea_norm ** 2).mean()

                if conf.sec_wei != 0:
                    loss = loss + conf.sec_wei * loss_sec
                if conf.l2reg_wei != 0:
                    loss = loss + conf.l2reg_wei * loss_l2reg

                loss.backward()

                self.writer.add_scalar('info/norm_var', norm_var.detach().item(), self.step)
                self.writer.add_scalar('info/norm_mean', norm_mean.detach().item(), self.step)
                self.writer.add_scalar('info/loss_sec', loss_sec.item(), self.step)
                self.writer.add_scalar('info/loss_l2reg', loss_l2reg.item(), self.step)
                self.writer.add_scalar('info/avg_ap', avg_ap.item(), self.step)
                self.writer.add_scalar('info/avg_an', avg_an.item(), self.step)
                self.writer.add_scalar('info/record_norm_mean', self.record_norm_mean.item(), self.step)
                self.writer.add_scalar('info/lr', self.optimizer.param_groups[2]['lr'], self.step)

                loss_meter.update(loss.item())

                self.optimizer.step()

                if self.step % self.evaluate_every ==0 and self.step != 0:
                    nmi, f1, recall_ks = self.test(conf)
                    self.writer.add_scalar('{}/test_nmi'.format(conf.use_dataset), nmi, self.step)
                    self.writer.add_scalar('{}/test_f1'.format(conf.use_dataset), f1, self.step)
                    self.writer.add_scalar('{}/test_recall_at_1'.format(conf.use_dataset), recall_ks[0], self.step)
                    logging.info(f'test on {conf.use_dataset}: nmi is {nmi}, f1 is {f1}, recalls are {recall_ks[0]}, {recall_ks[1]}, {recall_ks[2]}, {recall_ks[3:]} ')

                    nmi, f1, recall_ks = self.validate(conf)
                    self.writer.add_scalar('{}/train_nmi'.format(conf.use_dataset), nmi, self.step)
                    self.writer.add_scalar('{}/train_f1'.format(conf.use_dataset), f1, self.step)
                    self.writer.add_scalar('{}/train_recall_at_1'.format(conf.use_dataset), recall_ks[0], self.step)
                    logging.info(f'val on {conf.use_dataset}: nmi is {nmi}, f1 is {f1}, recall_at_1 is {recall_ks[0]} ')

                    self.train_with_fixed_bn(conf)

                if self.step % self.board_loss_every == 0 and self.step != 0:
                    # record lr
                    self.writer.add_scalar('train_loss', loss_meter.avg, self.step)

                    logging.info(f'step {self.step}: ' +
                                 f'loss: {loss_meter.avg:.3f} ' +
                                 f'data time: {data_time.avg:.2f} ' +
                                 f'loss time: {loss_time.avg:.2f} ' +
                                 f'speed: {conf.batch_size/(data_time.avg+loss_time.avg):.2f} imgs/s ' +
                                 f'norm_mean: {norm_mean.item():.2f} ' +
                                 f'norm_var: {norm_var.item():.2f}')

                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf)

                self.step += 1

                loss_time.update(myutils.timer.since_last_check(verbose=False))

        self.save_state(conf, to_save_folder=True)

    def train_with_fixed_bn(self, conf):
        def fix_bn(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        if conf.freeze_bn:
            self.model.apply(fix_bn)
            self.model.module.head.train()
        else:
            pass

    def validate(self, conf):
        logging.info('start eval')
        self.model.eval()

        if conf.use_dataset == 'CUB' or conf.use_dataset == 'Cars' or conf.use_dataset == 'SOP':

            loader = DataLoader(self.dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,
                                    shuffle=False, pin_memory=True, drop_last=False)

            loader_enum = enumerate(loader)
            feas = torch.tensor([])
            labels = np.array([])
            with torch.no_grad():
                while True:
                    try:
                        ind_data, data = loader_enum.__next__()
                    except StopIteration as e:
                        break

                    imgs = data['image']
                    label = data['label']

                    output1 = self.model(imgs, normalized=False)
                    norm = output1.norm(dim=1, p=2, keepdim=True)
                    output1 = output1.div(norm.expand_as(output1))
                    feas = torch.cat((feas, output1.cpu()), 0)
                    labels = np.append(labels, label.cpu().numpy())

            if conf.use_dataset == 'SOP':
                nmi = 0
                f1 = 0
            else:
                pids = np.unique(labels)
                nmi, f1 = NMI_F1(feas, labels, n_cluster=len(pids))

            sim_mat = pairwise_similarity(feas)
            sim_mat = sim_mat - torch.eye(sim_mat.size(0))
            recall_ks = Recall_at_ks(sim_mat, data_name=conf.use_dataset, gallery_ids=labels)

        elif conf.use_dataset=='Inshop':
            nmi = 0
            f1 = 0
            recall_ks = [0.0, 0.0]

        self.model.train()
        logging.info('eval end')
        return nmi, f1, recall_ks

    def test(self, conf):
        logging.info('start test')
        self.model.eval()
       
        if conf.use_dataset=='CUB' or conf.use_dataset=='Cars' or conf.use_dataset=='SOP':

            if conf.use_dataset == 'CUB':
                dataset = MSBaseDataSet(conf, './datasets/CUB_200_2011/cub_test.txt',
                                        transform=conf.transform_dict['center-crop'], mode='RGB')
            elif conf.use_dataset == 'Cars':
                dataset = MSBaseDataSet(conf, './datasets/CARS196/cars_test.txt',
                                        transform=conf.transform_dict['center-crop'], mode='RGB')
            elif conf.use_dataset == 'SOP':
                dataset = MSBaseDataSet(conf, './datasets/SOP/sop_test.txt',
                                        transform=conf.transform_dict['center-crop'], mode='RGB')

            loader = DataLoader(dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,
                                shuffle=False, pin_memory=True, drop_last=False)

            loader_enum = enumerate(loader)
            feas = torch.tensor([])
            labels = np.array([])
            with torch.no_grad():
                while True:
                    try:
                        ind_data, data = loader_enum.__next__()
                    except StopIteration as e:
                        break

                    imgs = data['image']
                    label = data['label']

                    output1 = self.model(imgs, normalized=False)
                    norm = output1.norm(dim=1, p=2, keepdim=True)
                    output1 = output1.div(norm.expand_as(output1))
                    feas = torch.cat((feas, output1.cpu()), 0)
                    labels = np.append(labels, label.cpu().numpy())
 
            if conf.use_dataset == 'SOP':
                nmi = 0
                f1 = 0
            else:
                pids = np.unique(labels)
                nmi, f1 = NMI_F1(feas, labels, n_cluster=len(pids))

            sim_mat = pairwise_similarity(feas)
            sim_mat = sim_mat - torch.eye(sim_mat.size(0))
            recall_ks = Recall_at_ks(sim_mat, data_name=conf.use_dataset, gallery_ids=labels)

        elif conf.use_dataset=='Inshop':
            nmi = 0
            f1 = 0
            # query
            dataset_query = MSBaseDataSet(conf, './datasets/Inshop/inshop_query.txt',
                                          transform=conf.transform_dict['center-crop'], mode='RGB')
            loader_query = DataLoader(dataset_query, batch_size=conf.batch_size, num_workers=conf.num_workers,
                                      shuffle=False, pin_memory=True, drop_last=False)
            loader_query_enum = enumerate(loader_query)
            feas_query = torch.tensor([])
            labels_query = np.array([])
            with torch.no_grad():
                while True:
                    try:
                        ind_data, data = loader_query_enum.__next__()
                    except StopIteration as e:
                        break

                    imgs = data['image']
                    label = data['label']

                    output1 = self.model(imgs, normalized=False)
                    norm = output1.norm(dim=1, p=2, keepdim=True)
                    output1 = output1.div(norm.expand_as(output1))
                    feas_query = torch.cat((feas_query, output1.cpu()), 0)
                    labels_query = np.append(labels_query, label.cpu().numpy())
            # gallery
            dataset_gallery = MSBaseDataSet(conf, './datasets/Inshop/inshop_gallery.txt',
                                            transform=conf.transform_dict['center-crop'], mode='RGB')
            loader_gallery = DataLoader(dataset_gallery, batch_size=conf.batch_size, num_workers=conf.num_workers,
                                        shuffle=False, pin_memory=True, drop_last=False)
            loader_gallery_enum = enumerate(loader_gallery)
            feas_gallery = torch.tensor([])
            labels_gallery = np.array([])
            with torch.no_grad():
                while True:
                    try:
                        ind_data, data = loader_gallery_enum.__next__()
                    except StopIteration as e:
                        break

                    imgs = data['image']
                    label = data['label']

                    output1 = self.model(imgs, normalized=False)
                    norm = output1.norm(dim=1, p=2, keepdim=True)
                    output1 = output1.div(norm.expand_as(output1))
                    feas_gallery = torch.cat((feas_gallery, output1.cpu()), 0)
                    labels_gallery = np.append(labels_gallery, label.cpu().numpy())
            # test
            sim_mat = pairwise_similarity(feas_query, feas_gallery)
            recall_ks = Recall_at_ks(sim_mat, data_name=conf.use_dataset, query_ids=labels_query, gallery_ids=labels_gallery)

        self.model.train()
        logging.info('test end')

        return nmi, f1, recall_ks

    def test_sop_complete(self, conf):
        assert conf.use_dataset == 'SOP'

        logging.info('start complete sop test')
        self.model.eval()

        dataset = MSBaseDataSet(conf, './datasets/SOP/sop_test.txt',
                                transform=conf.transform_dict['center-crop'], mode='RGB')
        loader = DataLoader(dataset, batch_size=conf.batch_size, num_workers=conf.num_workers,
                                shuffle=False, pin_memory=True, drop_last=False)

        loader_enum = enumerate(loader)
        feas = torch.tensor([])
        labels = np.array([])
        with torch.no_grad():
            while True:
                try:
                    ind_data, data = loader_enum.__next__()
                except StopIteration as e:
                    break

                imgs = data['image']
                label = data['label']

                output1 = self.model(imgs, normalized=False)
                norm = output1.norm(dim=1, p=2, keepdim=True)
                output1 = output1.div(norm.expand_as(output1))
                feas = torch.cat((feas, output1.cpu()), 0)
                labels = np.append(labels, label.cpu().numpy())

        pids = np.unique(labels)
        nmi, f1 = NMI_F1(feas, labels, n_cluster=len(pids))

        print(f'nmi: {nmi}, f1: {f1}')

        sim_mat = pairwise_similarity(feas)
        sim_mat = sim_mat - torch.eye(sim_mat.size(0))
        recall_ks = Recall_at_ks(sim_mat, data_name=conf.use_dataset, gallery_ids=labels)

        self.model.train()
        logging.info('test end')

        return nmi, f1, recall_ks

    def load_bninception_pretrained(self, conf):
        model_dict = self.model.state_dict()
        my_dict = {'module.'+k: v for k, v in torch.load(conf.bninception_pretrained_model_path).items() if 'module.'+k in model_dict.keys()}
        print('################################## do not have pretrained:')
        for k in model_dict:
            if k not in my_dict.keys():
                print(k)
        print('##################################')
        model_dict.update(my_dict)
        self.model.load_state_dict(model_dict)

    def schedule_lr(self, conf):
        for params in self.optimizer.param_groups:
            params['lr'] = params['lr'] * conf.lr_gamma
        logging.info(f'{self.optimizer}')

    def save_state(self, conf, to_save_folder=False, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path

        myutils.mkdir_p(save_path, delete=False)
        
        torch.save(
            self.model.state_dict(),
            save_path /
            ('model_{}_step:{}.pth'.format(get_time(), self.step)))
        if not model_only:
            torch.save(
                self.optimizer.state_dict(),
                save_path /
                ('optimizer_{}_step:{}.pth'.format(get_time(), self.step)))

    def load_state(self, conf, resume_path, fixed_str=None, load_optimizer=False):
        from pathlib import Path

        save_path = Path(resume_path)
        modelp = save_path / 'model_{}'.format(fixed_str)
        if not os.path.exists(modelp):
            fixed_strs = [t.name for t in save_path.glob('model*_*.pth')]
            step = [fixed_str.split('_')[-1].split(':')[-1].split('.')[-2] for fixed_str in fixed_strs]
            step = np.asarray(step, dtype=int)
            step_ind = step.argmax()
            fixed_str = fixed_strs[step_ind].replace('model_', '')
            modelp = save_path / 'model_{}'.format(fixed_str)

        print(fixed_str)

        model_dict = self.model.state_dict()
        my_dict = {k: v for k, v in torch.load(modelp).items() if k in model_dict.keys()}
        print('################################## do not have pretrained:')
        for k in model_dict:
            if k not in my_dict.keys():
                print(k)
        print('##################################')
        model_dict.update(my_dict)
        self.model.load_state_dict(model_dict)

        if load_optimizer:
            self.optimizer.load_state_dict(torch.load(save_path / 'optimizer_{}'.format(fixed_str)))
            print(self.optimizer)

