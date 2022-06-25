# encoding:UTF-8
from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture
# import dataloader_cifar as dataloader
from torch.utils.tensorboard import SummaryWriter
import pdb
import io
# from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as tfs
from edl_losses import *
import warnings
from pathlib import Path
from sklearn.preprocessing import normalize

import copy
from models import *
# import tensorflow as tf
from torch.optim.lr_scheduler import StepLR
from scipy import stats, misc
from operator import itemgetter
from kneed import KneeLocator
# from sklearn.ensemble import IsolationForest
import pickle
import joblib
import csv
from dataprocessing import *
from numpy import *
import copy
import scipy.io as sio
import pandas as pd
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from sklearn.manifold import TSNE
from nets.models import DenseNet, UNet
from loguru import logger
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
from matplotlib.collections import PatchCollection
from matplotlib.legend_handler import HandlerLine2D

# import imageio


# from prefetch_generator import BackgroundGenerator
cuda_device = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--method', default='FedMIN', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--noisy_dataset', default='SVHN', type=str)
parser.add_argument('--local_bs', default=32, type=int, help='train batchsize')
parser.add_argument('--blr', '--s_learning_rate', default=0.01, type=float, help='initial learning rate for netS')
parser.add_argument('--clr', '--d_learning_rate', default=0.01, type=float, help='initial learning rate for netD')
parser.add_argument('--r', default=0.6, type=float, help='noise ratio')
parser.add_argument('--on', default=0.5,type=float, help='open noise ratio')
parser.add_argument('--noisy_clnt', default=0.25,type=float, help='the ratio of noisy client')
parser.add_argument('--skip_warmup', default=True)
parser.add_argument('--load_warmupS', default=True,help='load net S or net D')
parser.add_argument('--num_users', default=4, type=int)
parser.add_argument('--warmup_global_epoch', default=5, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--drp', default=False, type=bool)
parser.add_argument('--mode', default='asym', type=str)
parser.add_argument('--load_SL', default=True, type=bool)
parser.add_argument('--autoload', default=False, type=bool)
parser.add_argument('--autodown', default=False, type=bool)
parser.add_argument('--prepare_data', default=False, type=bool)
parser.add_argument('--num_classes', default=10, type=int)
#======================not so common======================================
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--seed', default=123)
parser.add_argument('--inference', action='store_true')
parser.add_argument('--start_epoch', default=1, type=int)
parser.add_argument('--train_epochs_Bnet', default=10)
parser.add_argument('--train_epochs_Cnet', default=10)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--mom', type=int, default=0.9,
                    help="momentem for optimizer")  # 0.9
parser.add_argument('--decay', type=int, default=5e-4,
                    help="momentem for optimizer")
parser.add_argument('--cam_sz', default=32, type=int)

args = parser.parse_args()
def args_groupset():
    global args
    if args.dataset == 'camelyon17':
        args.noisy_dataset = 'Monusac'
        args.local_bs = 32
        args.num_users = 5
        args.num_classes = 2
        args.blr = 0.001
        args.clr = 0.001
        args.drp = True
args_groupset()
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# -------------zbx:设备---------------------------------
writer = SummaryWriter(comment='scalar')

args.title = '{}-{}-{}'.format(args.dataset, args.noisy_dataset,args.mode)
config = 'noise={}_{}_numusr={}'.format(args.r,args.on,args.num_users)
data_dir = os.path.join('data/'+args.title,config)
checkpoint_dir = os.path.join('saveDicts/'+args.method+'/'+args.title,config)
plots_dir = os.path.join('plots/'+args.method+'/'+args.title, config)
# ------------------------------------------------
Path(os.path.join(checkpoint_dir, )).mkdir(parents=True, exist_ok=True)
Path(os.path.join(data_dir, )).mkdir(parents=True, exist_ok=True)
Path(os.path.join(plots_dir, )).mkdir(parents=True, exist_ok=True)
serverPath = os.path.join(data_dir, 'server')
clientPath = os.path.join(data_dir, 'client')
Path(serverPath).mkdir(parents=True, exist_ok=True)
Path(clientPath).mkdir(parents=True, exist_ok=True)

noise_chart_op = [[] for i in range(args.num_users)]
noise_chart_cl = [[] for i in range(args.num_users)]

num_noiy = int(args.noisy_clnt*args.num_users) #脏客户端的个数
pt = int(args.num_users/num_noiy)

noise_idx = [True if i%pt==0 else False for i in range(1,args.num_users+1)]
# noise_idx = [False,False,True,True]
see = np.sum(noise_idx)
noy_clnt = see/args.num_users
print('the number of clients: '+str(args.num_users))
print('the nuber of noisy clients: '+str(see))

logger.add(os.path.join(checkpoint_dir,'runtime.log'))


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))  # 交叉熵损失
        Lu = torch.mean((probs_u - targets_u) ** 2)  # 欧氏距离
        return Lx, Lu, linear_rampup(epoch)

subjective_loss = edl_mse_loss
CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
criterion = SemiLoss()

class Server():
    def __init__(self, model_C, model_B, datapath):
        self.model_C = model_C
        self.model_B = model_B
        # self.model_C = torch.nn.DataParallel(model_C,device_ids=device_ids).cuda()
        # self.model_B = torch.nn.DataParallel(model_B,device_ids=device_ids).cuda()
        self.data_dir = datapath
        self.stop = False
        with open(os.path.join(self.data_dir, 'testdata.pkl'), 'rb') as f:
            test_dataset = joblib.load(f)
        self.test_loader = DataLoader(test_dataset, batch_size=args.local_bs, shuffle=False,num_workers=args.num_workers,drop_last=args.drp)
        with open(os.path.join(self.data_dir, 'benchdata.pkl'), 'rb') as f:
            bench_dataset = joblib.load(f)
        self.bench_loader = DataLoader(bench_dataset, batch_size=args.local_bs, shuffle=True,num_workers=args.num_workers,drop_last=args.drp)
        global noise_chart_cl,noise_chart_op
        with open(os.path.join(serverPath, 'noise_chart_op.pkl'), 'rb') as f:
            noise_chart_op = pickle.load(f)
        with open(os.path.join(serverPath, 'noise_chart_clo.pkl'), 'rb') as f:
            noise_chart_cl = pickle.load(f)


    def test(self, epoch, mod_name=''):
        if mod_name == 'netB':
            model = self.model_B
        else:
            model = self.model_C
        model.eval()
        correct = 0
        total = 0
        LOSS = 0
        criterion = nn.CrossEntropyLoss().to(args.device)
        num_iter = len(self.test_loader)
        with torch.no_grad():
            for batch_ixx, (inputs, targets) in enumerate(self.test_loader):
                sys.stdout.write('\r')
                sys.stdout.write('load_bat %3d'
                                 % (batch_ixx))
                sys.stdout.flush()
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                loss = criterion(outputs, targets)
                LOSS += loss

                total += targets.size(0)
                correct += predicted.eq(targets).cpu().sum().item()
                sys.stdout.write('\r')
                sys.stdout.write('Test  | Global Epoch %d | Iter[%3d/%3d]\t loss: %.4f'
                                 % (epoch, batch_ixx + 1, num_iter,
                                    loss.item()))
                sys.stdout.flush()
                del inputs, targets,loss
                torch.cuda.empty_cache()

        acc = 100. * correct / total
        LOSS = LOSS / len(self.test_loader)
        if mod_name == 'netB':
            accB = acc
            accC = 0
            lossB = LOSS
            lossC = 0
        else:
            accC = acc
            accB = 0
            lossC = LOSS
            lossB = 0
        logger.debug("\n| %s |Global Epoch #%d\t Accuracy: %.2f%%\n" % (mod_name, epoch, acc))
        writer.add_scalars('Test/Accuracy', {'netB': accB, 'netC': accC}, epoch)
        writer.add_scalars('Test/Loss', {'netB': lossB, 'netC': lossC}, epoch)
        del acc, LOSS



    def plt_knee(self,x,y,knee,min,clnt):
        """
        Plot the curve and the knee, if it exists

        :param figsize: Optional[Tuple[int, int]
            The figure size of the plot. Example (12, 8)
        :return: NoReturn
        """
        matplotlib.rcParams.update({'font.size': 80, 'font.family': 'sans-serief',
                                    'font.weight': 'normal', 'grid.linewidth': 3})
        fig = plt.figure(figsize=(22, 20))
        font1 = {'family': 'sans-serief',
                 'weight': 'normal',
                 'size': 60,
                 }

        plt.tick_params()
        plt.title("Knee Point")
        plt.plot(x, y, "b",linewidth=8)
        plt.vlines(
            (knee), 0, 1, linestyles="solid", label='knee point',colors=('r'),linewidth=8
        )
        plt.vlines(
            (min), 0, 1, linestyles="dashed", label='valley point', colors = ('g'),linewidth=8
        )
        plt.xlabel('Index in sorted $\mathregular{Q^n}$',labelpad=45)
        plt.ylabel('KS distance',labelpad=45)
        plt.ylim(0,1)
        plt.legend(loc="upper center",prop=font1)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir,'knee_KS_clnt{}_{}_{}.png'.format(clnt,args.r,args.on)),bbox_inches='tight')
        plt.show()
        plt.close()


    def compute_lamda_gmm_samp(self,agg_gmm,local_gmm,u_idx,n_samples=10**3,plot=False):
        X,_ = agg_gmm.sample(n_samples)
        Y,_ = local_gmm.sample(n_samples)
        user_len = len(Y)
        samp1 = np.squeeze(X)
        samp2 = np.squeeze(Y)
        enu1 = sorted(enumerate(samp1), key=itemgetter(1))
        enu2 = sorted(enumerate(samp2), key=itemgetter(1))

        sp1 = [value for index, value in enu1]
        sp2 = [value for index, value in enu2]
        temp = -1

        D = stats.ks_2samp

        inixis = D(samp1, samp2)
        inixis = inixis[0]  # 0.43225 0.9833

        distance = []
        stp = 100
        STEP = np.arange(user_len / stp, user_len + 1, user_len / stp)
        STEP = STEP.astype(np.int)
        idx = 0
        for t in STEP:
            trunsp2 = sp2[:t]

            dis = D(sp1, trunsp2)
            dis = dis[0]
            distance.append(dis)
            if dis <= inixis:
                inixis = dis
                temp = t - 1
                best_idx = idx
            if log:
                writer.add_scalar('KS_loss', dis, t)
            idx += 1
        kneedle_cov_dec = KneeLocator(STEP[:best_idx], distance[:best_idx], curve='convex',
                                      direction='decreasing',online=True)
        diff_max = np.max(kneedle_cov_dec.y_difference)
        signific = 0.05
        if diff_max < signific:
            tag1 = temp
        else:
            tag1 = kneedle_cov_dec.knee
        tag2 = temp
        t1,t2 = sp2[tag1].item(), sp2[tag2].item()
        if noise_idx[u_idx] and plot:
            self.plt_knee(STEP,distance,tag1,tag2,clnt='noisy')

        return t1,t2

    def pre_train(self, max_epochs=50, savepath='', mod_name=''):
        self.model.train()
        criterion = nn.CrossEntropyLoss().to(args.device)

        loader = self.bench_loader
        num_iter = (len(loader.dataset) // loader.batch_size) + 1
        optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.local_mom,
                              weight_decay=args.local_decay)
        for epoch in range(max_epochs):
            for batch_idx, (inputs, labels) in enumerate(loader):
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                sys.stdout.write('\r')
                sys.stdout.write('Server  | Pre-train | Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.4f'
                                 % (epoch, max_epochs, batch_idx + 1, num_iter,
                                    loss.item()))
                sys.stdout.flush()

            self.test(epoch - max_epochs)
        torch.save(self.model.state_dict(), savepath)

    def FedAvg(self, w, userdatlen='', mod_name=''):
        wt = userdatlen/np.sum(userdatlen)
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(len(w)):
                tensor = torch.mul(w[i][key], wt[i])
                if i == 0:
                    w_avg[key] = tensor.type(w[i][key].dtype)
                else:
                    w_avg[key] += tensor.type(w[i][key].dtype)
        if mod_name == 'netB':
            self.model_B.load_state_dict(w_avg)
        else:
            self.model_C.load_state_dict(w_avg)

    def sendmodel(self):
        torch.save({
            'netB_state_dict': self.model_B.state_dict(),
            'netC_state_dict': self.model_C.state_dict(),
        }, os.path.join(checkpoint_dir, 'servermodel_ep.json'))

    def receivemodel(self):
        torchLoad = torch.load(os.path.join(checkpoint_dir, 'clntmodel.json'))
        self.model_B.load_state_dict(torchLoad['netB_state_dict'])

    def JS_gmm(self,gmm_p, gmm_q, n_samples=10 ** 5):
        X,_ = gmm_p.sample(n_samples)
        log_p_X= gmm_p.score_samples(X)
        log_q_X= gmm_q.score_samples(X)
        log_mix_X = np.logaddexp(log_p_X, log_q_X)
        p_X = np.exp(log_p_X)/np.sum(np.exp(log_p_X))
        q_X = np.exp(log_q_X)/np.sum(np.exp(log_q_X))

        Y,_ = gmm_q.sample(n_samples)
        log_p_Y= gmm_p.score_samples(Y)
        log_q_Y= gmm_q.score_samples(Y)
        log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)
        p_Y = np.exp(log_p_Y)/np.sum(np.exp(log_p_Y))
        q_Y = np.exp(log_q_Y)/np.sum(np.exp(log_q_Y))

        KL1 = log_p_X.mean()-(log_mix_X.mean()-np.log(2))
        KL2 = log_q_Y.mean()-(log_mix_Y.mean()-np.log(2))
        JS = (KL1+KL2)/2

        kl1 = p_X*(log_p_X-log_q_X)
        see = np.sum(p_X)
        kl1 = np.sum(kl1)
        kl2 = q_Y*(log_q_Y-log_p_Y)
        kl2 = np.sum(kl2)
        js = (kl1+kl2)/2
        if JS<0:
            JS=0
        return JS

    def KL_gmm(self,gmm_p, gmm_q, n_samples=10 ** 5):
        X,_ = gmm_p.sample(n_samples)
        log_p_X= gmm_p.score_samples(X)
        p_X = np.exp(log_p_X)/np.sum(np.exp(log_p_X))
        log_q_X= gmm_q.score_samples(X)
        rst = np.sum(p_X*(log_p_X-log_q_X))
        rst2 = log_p_X.mean() - log_q_X.mean()
        if rst<0:
            rst = 0
        return rst


    def gmm_like(self,gmm_like):
        gmm = GaussianMixture(n_components=gmm_like.n_components, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.means_ = gmm_like.means_
        gmm.covariances_ = gmm_like.covariances_
        gmm.weights_ = gmm_like.weights_
        return gmm

    def plt_pdf(self,values_lst,idx):
        matplotlib.rcParams.update({'font.size': 80, 'font.family': 'sans-serief',
                                    'font.weight': 'normal', 'grid.linewidth': 3})
        fig = plt.figure(figsize=(22, 20))
        font1 = {'family': 'sans-serief',
                 'weight': 'normal',
                 'size': 80,
                 }
        if idx=='merge_01_04':
            lgd = ['Mergence']
            clr = ['red']
        else:
            lgd = ['P_clean','P_noise']
            clr = ['green','blue']

        for i in range(len(values_lst)):
            plt.hist(values_lst[i], bins=20, color=clr[i],alpha=0.6)

        plt.grid(axis='y')
        plt.style.use('seaborn-poster')
        plt.ylim(0,4000)
        plt.ylabel('Frequency',labelpad=45)
        plt.xlabel('Loss',labelpad=45)
        plt.legend(lgd,prop=font1)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir,'loss distribution_{}.png'.format(idx)),bbox_inches='tight')
        plt.show()
        plt.close()

    def plt_gmm(self,gmm_lst,n_samples=1000,idx=''):
        matplotlib.rcParams.update({'font.size': 80, 'font.family': 'sans-serief',
                                    'font.weight': 'normal', 'grid.linewidth': 3})
        fig = plt.figure(figsize=(22, 20))
        font1 = {'family': 'sans-serief',
                 'weight': 'normal',
                 'size': 80,
                 }
        if idx == 'global_samp_01_04':
            clr = ['red']
            lgd = ['Aggregation']
        else:
            clr = ['green','blue']
            lgd = ['P_clean','P_noise']
        for g in range(len(gmm_lst)):
            gmm = gmm_lst[g]
            x,_ = gmm.sample(n_samples)
            log_prob = gmm.score_samples(x)
            pdf = np.exp(log_prob)
            prob = gmm.predict_proba(x)
            x_sque = np.squeeze(x)
            x_srt = sorted(enumerate(x_sque), key=lambda x: x[1])
            values = [value for index,value in x_srt]
            key = [index for index,value in x_srt]
            new_pdf = np.zeros_like(pdf)
            for i in range(len(pdf)):
                new_pdf[i] = pdf[key[i]]
            plt.plot(values,new_pdf,color=clr[g],lw=8)
            plt.fill_between(values,new_pdf,0,alpha=0.5,lw=.1,color=clr[g])

        plt.grid(axis='y')
        plt.style.use('seaborn-poster')
        # plt.ylim(0,13)
        plt.ylabel('Probability density',labelpad=45)
        plt.xlabel('Loss',labelpad=45)
        plt.legend(lgd,prop=font1)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir,'gmm from_{}.png'.format(idx)),bbox_inches='tight')
        plt.show()

    def merge_gmm_samp(self,gmm_lst=[],n_samples=10**3,plot=False):
        fake_all = []
        for i in range(len(gmm_lst)):
        # for i in [0,3]:
            X,_ = gmm_lst[i].sample(n_samples)
            each_loss = np.squeeze(X)
            fake_all.extend(each_loss)
        global_gmm = GaussianMixture(n_components=20, max_iter=10, tol=1e-2, reg_covar=5e-4)
        input_loss = np.array(fake_all).reshape(-1,1)
        global_gmm.fit(input_loss)
        if plot:
            self.plt_gmm([global_gmm],idx='global_samp_01_04')
        return global_gmm



class Client:
    def __init__(self, client_id, model_C, model_B, datapath=[]):
        # self.model_C = torch.nn.DataParallel(model_C,device_ids=device_ids).cuda()
        # self.model_B = torch.nn.DataParallel(model_B,device_ids=device_ids).cuda()
        self.model_C = model_C
        self.model_B = model_B
        self.data_dir = datapath
        self.client_id = client_id
        self.sfm_Mat = None
        self.avai_dataset = None
        self.keys = []
        self.clean_labels = []
        self.conf = []
        self.labeled_loader = []
        self.unlabeled_loader = []

        self.open_stack = []
        self.close_stack = []
        self.lof_predict = []
        self.sbj_loss = []
        self.probs = []
        self.preds = []

        with open(os.path.join(self.data_dir, 'dataset' + str(client_id) + '.pkl'), 'rb') as f:
            self.dataset = joblib.load(f)
        if noise_idx[self.client_id]:
            self.dataset.args = args
        self.avai_dataset = copy.deepcopy(self.dataset)
        self.data_loader = DataLoader(self.dataset, shuffle=True, batch_size=args.local_bs,num_workers=args.num_workers,drop_last=args.drp)
        with open(os.path.join(serverPath, 'testdata.pkl'), 'rb') as f:
            test_dataset = joblib.load(f)

        with open(os.path.join(clientPath, 'clean_labels' + str(client_id) + '.pkl'), 'rb') as f:
            self.clean_labels = joblib.load(f)
        self.sudo_labels = self.dataset.labels


    def receivemodel(self):
        torchLoad = torch.load(os.path.join(checkpoint_dir, 'servermodel_ep.json'))
        self.model_B.load_state_dict(torchLoad['netB_state_dict'])
        self.model_C.load_state_dict(torchLoad['netC_state_dict'])

    def sendmodel(self):
        torch.save({
            'netB_state_dict': self.model_B.state_dict(),
            'netC_state_dict': self.model_C.state_dict(),
        }, os.path.join(checkpoint_dir, 'clntmodel.json'))

    def test(self, epoch, mod_name=''):
        if mod_name == 'netB':
            model = self.model_B
        else:
            model = self.model_C
        model.eval()
        correct = 0
        total = 0
        LOSS = 0
        criterion = nn.CrossEntropyLoss().to(args.device)

        with torch.no_grad():
            for batch_ixx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                loss = criterion(outputs, targets)
                LOSS += loss

                total += targets.size(0)
                correct += predicted.eq(targets).cpu().sum().item()
        acc = 100. * correct / total
        LOSS = LOSS / len(self.test_loader)

        # print("\n| %s |Global Epoch #%d\t Accuracy: %.2f%%\n" % (mod_name, epoch, acc))
        logger.debug("\n| Global Epoch #%d|\t Client %s |\t Accuracy: %.2f%%\n" % (epoch,self.client_id, acc))


        return acc,LOSS

    def plt_Histogram(self,data, predictions=None, log=True):
        bin_num = 100
        clean, open_noise, closed_noise = self.get_noise()
        matplotlib.rcParams.update({'font.size': 80, 'font.family': 'sans-serief',
                                    'font.weight': 'normal', 'grid.linewidth': 3})
        fig = plt.figure(figsize=(22, 20))
        font1 = {'family': 'sans-serief',
                 'size': 80,
                 }
        if predictions is not None:
            plt.subplot(121)
        plt.hist(data[clean], bins=bin_num, alpha=0.7, color='green',label='clean samples')
        plt.hist(data[closed_noise], bins=bin_num, alpha=0.7,color='blue',label='closed-set noise')
        plt.hist(data[open_noise], bins=bin_num, alpha=0.7,color='red',label='open-set noise')
        plt.legend(prop=font1,loc='upper right')
        plt.xlabel('Loss',labelpad=45)
        plt.ylabel('Frequency',labelpad=45)
        plt.ylim(0,3200)
        if predictions is not None:
            plt.subplot(122)
            plt.hist(data[predictions[0]], bins=300, alpha=0.5, color='green', label='Predicted clean set')
            plt.hist(data[predictions[2]], bins=300, alpha=0.5, color='blue', label='Predicted closed set')
            plt.hist(data[predictions[1]], bins=300, alpha=0.5, color='red', label='Predicted open set')
            plt.legend(loc='upper right')
        if log:
            plt.grid(axis='y')
            plt.tight_layout()
            print('\nlogging histogram...')
            plt.savefig(os.path.join(checkpoint_dir, 'loss_distribu_clnt{}_{}_{}.png'.format(self.client_id,args.r,args.on)), format='png',bbox_inches='tight')
        plt.show()
        plt.close()

    def get_noise(self):
        data_len = len(self.dataset)
        if noise_idx[self.client_id]:
            open = noise_chart_op[self.client_id]
            close = noise_chart_cl[self.client_id]
            clean = [False] * data_len
            for i in range(data_len):
                if not open[i] and not close[i]:
                    clean[i] = True
        else:
            open = [False] * data_len
            close = [False] * data_len
            clean = [True] * data_len
        return np.array(clean), np.array(open), np.array(close)

    def local_Mixmatch(self, global_ep,epoch):
        optimizer = optim.SGD(self.model_C.parameters(), lr=args.clr, momentum=args.mom,
                               weight_decay=args.decay)
        self.model_C.train()
        unlabeled_train_iter = iter(self.unlabeled_loader)
        num_iter = (len(self.labeled_loader.dataset) // args.local_bs) + 1
        for batch_ixx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(self.labeled_loader):
            try:
                inputs_u, inputs_u2,labels_u,p_u = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter= iter(self.unlabeled_loader)  # 如果有标签数据还没有喂完，但无标签数据已经喂完，那就重复再喂一次
                inputs_u, inputs_u2,labels_u,p_u  = unlabeled_train_iter.next()
            batch_size = inputs_x.size(0)

            # Transform label to one-hot
            labels_x = torch.zeros(batch_size, args.num_classes).scatter_(1, labels_x.view(-1, 1), 1)
            w_x = w_x.view(-1, 1).type(torch.FloatTensor)

            inputs_x, inputs_x2, labels_x, w_x = inputs_x.to(args.device), inputs_x2.to(args.device), labels_x.to(
                args.device), w_x.to(
                args.device)
            inputs_u, inputs_u2 = inputs_u.to(args.device), inputs_u2.to(args.device)

            with torch.no_grad():
                # label guessing of unlabeled samples
                outputs_u1 = self.model_C(inputs_u)
                outputs_u2 = self.model_C(inputs_u2)

                pu = (torch.softmax(outputs_u1, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
                ptu = pu ** (1 / args.T)  # temparature sharpening
                #------------------guess labels-----------------------------
                # targets_u = (1-p_u) * labels_u+p_u * pu
                targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
                targets_u = targets_u.detach()

                # label refinement of labeled samples
                outputs_x = self.model_C(inputs_x)
                outputs_x2 = self.model_C(inputs_x2)

                px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
                px = w_x * labels_x + (1 - w_x) * px
                ptx = px ** (1 / args.T)  # temparature sharpening

                targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
                targets_x = targets_x.detach()

                # mixmatch
            l = np.random.beta(args.alpha, args.alpha)  # beta特别适合用作先验分布
            l = max(l, 1 - l)

            all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

            ixx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[ixx]
            target_a, target_b = all_targets, all_targets[ixx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            logits = self.model_C(mixed_input)
            logits_x = logits[:batch_size * 2]
            logits_u = logits[batch_size * 2:]

            Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size * 2], logits_u,
                                     mixed_target[batch_size * 2:],
                                     epoch + batch_ixx / num_iter)

            # regularization
            prior = torch.ones(args.num_classes) / args.num_classes
            prior = prior.to(args.device)
            pred_mean = torch.softmax(logits, dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))

            loss = Lx + lamb * Lu + penalty
            # loss = Lx + Lu
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            sys.stdout.write('\r')
            sys.stdout.write(
                'User = %d | %s:%.1f-%s | Global Epoch %d | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                % (self.client_id, args.dataset, args.r, args.mode, global_ep, epoch, args.train_epochs_Cnet,
                   batch_ixx + 1,
                   num_iter,
                   Lx.item(), Lu.item()))

            sys.stdout.flush()
            del inputs_u, inputs_u2,labels_u,p_u,inputs_x, inputs_x2, labels_x, w_x
            del outputs_u1,outputs_u2,outputs_x,outputs_x2,logits,logits_u,logits_x
            del prior,penalty,Lx, Lu, lamb,loss,all_inputs,all_targets
            torch.cuda.empty_cache()

        return self.model_C.state_dict(), len(self.labeled_loader.dataset) + len(self.unlabeled_loader.dataset)
        # return self.model_C.state_dict(), len(self.dataset)

    def update_weights(self, global_ep, optimizer, epoch, mod_name):
        if mod_name == 'netB':
            model = self.model_B
            criterion = edl_mse_loss
            epochs = args.train_epochs_Bnet
        else:
            model = self.model_C
            criterion = nn.CrossEntropyLoss().to(args.device)
            epochs = args.train_epochs_Cnet

        model.train()
        num_iter = (len(self.data_loader.dataset) // self.data_loader.batch_size) + 1
        for batch_idx, (inputs, labels) in enumerate(self.data_loader):
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels).mean()

            loss.backward()
            optimizer.step()
            sys.stdout.write('\r')
            sys.stdout.write('User = %d  | Global Epoch %d | Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.4f'
                             % (self.client_id, global_ep, epoch, epochs, batch_idx + 1, num_iter,
                                loss.item()))
            sys.stdout.flush()
            del inputs, labels, outputs, loss
            torch.cuda.empty_cache()


    def local_update(self, epoch, mod_name):

        if mod_name == 'netB':
            iterations = args.train_epochs_Bnet
            model = self.model_B
            optimizer = optim.SGD(model.parameters(), lr=args.blr, momentum=args.mom, weight_decay=args.decay)
        else:
            model = self.model_C
            optimizer = optim.SGD(model.parameters(), lr=args.clr, momentum=args.mom, weight_decay=args.decay)
            iterations = args.train_epochs_Cnet
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

        for iter in range(iterations):
            self.update_weights(epoch, optimizer, iter, mod_name)
            scheduler.step()
         
        return model.state_dict(), len(self.dataset)

    def refine_labels_new(self,epoch,log=True):
        probClosed = self.probs[2]
        self.model_C.eval()
        samples = self.dataset.samples
        self.sudo_labels = copy.deepcopy(self.dataset.labels)
        w_x = probClosed
        w_x = torch.from_numpy(np.expand_dims(w_x, axis=1)).to(args.device)
        eval_loader = DataLoader(data_loader(samples,self.dataset.labels,args=args,mode='ref'), shuffle=False, batch_size=args.local_bs,num_workers=args.num_workers,drop_last=args.drp)
        with torch.no_grad():
            for index, (inputs, labels) in enumerate(eval_loader):
                inputs, labels = inputs.to(args.device), one_hot_embedding(labels,num_classes=args.num_classes).to(args.device)
                outputs = self.model_C(inputs)
                px = torch.softmax(outputs, dim=1).to(args.device)
                start = index*args.local_bs
                end = index*args.local_bs+len(labels)
                px_e = (1-w_x[start:end])* labels.double() + w_x[start:end]* px.double()
                ptx = px_e ** (1 / args.T)  # temparature sharpening
                refined = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
                refined = refined.detach().argmax(dim=1)
                self.sudo_labels[start:end] = refined.cpu().numpy()

        self.data_loader = DataLoader(data_loader(samples, self.sudo_labels,args=args), batch_size=args.local_bs, shuffle=True,num_workers=args.num_workers,drop_last=args.drp)
        lb_chg = self.sudo_labels!=self.dataset.labels[:len(self.sudo_labels)]
        self.preds[0][lb_chg] = np.array([True]*np.sum(lb_chg))
        self.preds[1][lb_chg] = np.array([False]*np.sum(lb_chg))
        self.preds[2][lb_chg] = np.array([False]*np.sum(lb_chg))
        strongClosed = self.sbj_loss.argmax()
        self.preds[2][strongClosed] = True
        self.preds[0][strongClosed] = False
        self.preds[1][strongClosed] = False

    def Subjec_sample(self, ):

        self.model_B.eval()
        dataset = self.dataset
        loader = DataLoader(self.dataset, batch_size=args.local_bs, shuffle=False,num_workers=args.num_workers,drop_last=args.drp,pin_memory=False)
        losses = torch.zeros(len(dataset))
        criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
        print('Client %s | computing loss and feature' % (self.client_id))
        with torch.no_grad():
            for idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                sys.stdout.write('\r')
                sys.stdout.write('load_bat %3d/%3d'
                                 % (idx,len(loader)))
                sys.stdout.flush()

                loss = subjective_loss(self.model_B(inputs), targets)
                losses[args.local_bs * idx:args.local_bs * idx + len(loss)] = loss

        return losses, None

    def fit_gmm_multiple_components(self, input_loss, tag1=0.3, tag2=0.7):

        input_loss = torch.tensor(input_loss).reshape(-1, 1)
        np.random.seed(42)
        gmm = self.gmm
        components_open = []
        components_clean = []
        components_closed = []

        for n in range(gmm.n_components):
            if (gmm.means_[n] > tag1) & (gmm.means_[n] <= tag2):
                components_open.extend([n])
            elif (gmm.means_[n] <= tag1):
                components_clean.extend([n])
            else:
                components_closed.extend([n])
        prob = gmm.predict_proba(input_loss)
        # transform this probability into a 3-component probability
        prob_clean = np.sum(prob[:, components_clean], axis=1)
        prob_closed = np.sum(prob[:, components_closed], axis=1)
        prob_open = np.sum(prob[:, components_open], axis=1)
        x = input_loss.numpy()
        x = x.reshape(len(x))
        return prob_clean, prob_open, prob_closed

    def detect_noise_index(self, prediction, type='open'):
        clean, open, closed = self.get_noise()
        right = 0
        if type == 'open':
            noise = open
        elif type == 'closed':
            noise = closed
        else:
            noise = clean

        for i in range(len(prediction)):
            if prediction[i] and noise[i]:
                right += 1
        if np.sum(noise) == 0:
            recall = -1
            precision = -1
            f1score = -1
        else:
            recall = round(right / np.sum(noise), 2)
            precision = round(right / np.sum(prediction), 2)
            f1score = (recall + precision) / 2

        return recall, precision, round(f1score, 2)

    def performance(self, epoch):
        preds = self.preds
        
        recall_clean, precision_clean, f1score_clean= self.detect_noise_index(preds[0], type='clean')
        recall_open, precision_open, f1score_open= self.detect_noise_index(preds[1], type='open')
        recall_close, precision_close, f1score_close = self.detect_noise_index(preds[2], type='closed')

        writer.add_scalars('Attacker_Index/clean',
                           {'recall': recall_clean, 'precision': precision_clean,
                            'f1score': f1score_clean}, epoch)
        writer.add_scalars('Attacker_Index/open',
                           {'recall': recall_open, 'precision': precision_open,
                            'f1score': f1score_open}, epoch)
        writer.add_scalars('Attacker_Index/close',
                           {'recall': recall_close, 'precision': precision_close,
                            'f1score': f1score_close}, epoch)

    def pred_noise(self, probs):

        probClean, probOpen, probClosed = probs[0], probs[1], probs[2]

        predClean = (probClean > probOpen) & (probClean > probClosed)
        predClosed = (probClosed > probClean) & (probClosed > probOpen)
        predOpen = (probOpen > probClean) & (probOpen > probClosed)

        if len(self.open_stack) != 0:
            for i in range(len(predOpen)):
            
                if self.open_stack[i] and not predClean[i]:
                    if self.lof_predict[i]:
                        predOpen[i] = True
                        predClosed[i] = False
                        predClean[i] = False
                if self.close_stack[i] and not predClean[i]:
                    if not self.lof_predict[i]:
                        predClosed[i] = True
                        predOpen[i] = False
                        predClean[i] = False

        return [predClean, predOpen, predClosed]
       

    def eval_train(self, tag1=0.3, tag2=0.7):
        if len(self.sbj_loss) != 0:
            probClean, probOpen, probClosed = self.fit_gmm_multiple_components(self.sbj_loss, tag1=tag1, tag2=tag2)
            [predClean, predOpen, predClosed] = self.pred_noise(probs=[probClean, probOpen, probClosed])
            self.probs = [probClean, probOpen, probClosed]
            self.preds = [predClean, predOpen, predClosed]
            if args.autodown:
                with open(os.path.join(clientPath, 'probs_%d_t1=%.2f_t2=%.2f.pkl' % (self.client_id, tag1,tag2)),
                          'wb') as f:
                    pickle.dump(self.probs, f)
                with open(os.path.join(clientPath, 'preds_%d_t1=%.2f_t2=%.2f.pkl' % (self.client_id, tag1,tag2)),
                          'wb') as f:
                    pickle.dump(self.preds, f)

        else:
            if not args.load_SL:
                input_loss, lofactor = self.Subjec_sample()
                input_loss = (input_loss - input_loss.min()) / (input_loss.max() - input_loss.min())
            else:
                with open(os.path.join(clientPath, 'sbj_loss_%s.pkl' % (str(self.client_id))), 'rb') as f:
                    input_loss = pickle.load(f)
            self.sbj_loss = np.array(input_loss)
            input_loss = np.array(input_loss).reshape(-1,1)
            np.random.seed(42)
            gmm = GaussianMixture(n_components=20, max_iter=10, tol=1e-2, reg_covar=5e-4)
            gmm.fit(input_loss)
            self.gmm = gmm
            return  gmm

    def prob_add(self, probs1, probs2):
        probs = [0] * 3
        for i in range(3):
            probs[i] = np.sum([probs1[i], probs2[i]], axis=0)
        preds = self.pred_noise(probs)
        return probs, preds

    def labeled_load(self):
        predClean = self.preds[0]
        probClean = self.probs[0]
        samples = self.dataset.samples

        pred_idx = predClean.nonzero()[0]
        labeled_dataset = data_loader(samples[pred_idx], np.array(self.sudo_labels)[pred_idx], mode='labeled', prob=probClean[pred_idx],args=args)

        if len(pred_idx) < args.local_bs:
            labeled_loader = DataLoader(labeled_dataset, batch_size=args.local_bs, shuffle=True,
                                        num_workers=args.num_workers)
        else:
            labeled_loader = DataLoader(labeled_dataset, batch_size=args.local_bs, shuffle=True,num_workers=args.num_workers,drop_last=args.drp)
        self.labeled_loader = labeled_loader

    def unlabeled_load(self):
        predClosed = self.preds[2]
        probClosed = self.probs[2]
        pred_idx = predClosed.nonzero()[0]

        samples = self.dataset.samples
        unlabeled_dataset = data_loader(samples[pred_idx], np.array(self.sudo_labels)[pred_idx], mode='unlabeled',prob=probClosed[pred_idx],args=args)

        if len(pred_idx) < args.local_bs:
            unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.local_bs, shuffle=True,
                                          num_workers=args.num_workers,pin_memory=False)
        else:
            unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.local_bs, shuffle=True,num_workers=args.num_workers,drop_last=args.drp,pin_memory=False)
        self.unlabeled_loader = unlabeled_loader


class FedMIN:
    def __init__(self):
        if args.prepare_data:
            self.prepare_data()
        self.clients = [[] for i in range(args.num_users)]
        self.model_C = None
        self.model_B = None
        self.ini_model()
        for p_id in range(args.num_users):
            self.clients[p_id] = Client(p_id, copy.deepcopy(self.model_C), copy.deepcopy(self.model_B),clientPath)
        self.server = Server(self.model_C, self.model_B, serverPath)
        if not args.skip_warmup:
            self.warmup()
        else:
            if args.load_warmupS:
                torchLoad = torch.load(os.path.join(checkpoint_dir, '10localEp_warmupmodel_5_netB.json'))
                self.server.model_B.load_state_dict(torchLoad['netB_state_dict'])

        args.start_epoch = args.warmup_global_epoch+1

        self.server.sendmodel()
        for p_id in range(args.num_users):
            self.clients[p_id].receivemodel()

    def create_model(self):
        model = ResNet18(num_classes=args.num_classes)
        model = model.to(args.device)
        return model

    def ini_model(self):
        if args.dataset == 'cifar10' :
            self.model_C = self.create_model()
            self.model_B = self.create_model()
        elif args.dataset == 'camelyon17':
            self.model_C = DenseNet(input_shape=[3, args.cam_sz,args.cam_sz]).to(args.device)
            self.model_B = DenseNet(input_shape=[3,args.cam_sz,args.cam_sz]).to(args.device)

    def warmup(self,model_name='netB'):
        Keep_size = [0] * args.num_users
        model_param = [[] for i in range(args.num_users)]

        for iter in range(args.start_epoch,args.warmup_global_epoch+1):

            for ix in range(args.num_users):
                model_param[ix], Keep_size[ix] = self.clients[ix].local_update(iter, model_name)
            self.server.FedAvg(model_param, Keep_size, model_name)
            # self.server.model_B.load_state_dict(model_param_S[4])
            self.server.test(iter, model_name)

            self.server.sendmodel()
            for i in range(args.num_users):
                self.clients[i].receivemodel()
            torch.save({
                'netB_state_dict': self.server.model_B.state_dict(),
                'netC_state_dict': self.server.model_C.state_dict(),
            }, os.path.join(checkpoint_dir,'10localEp_warmupmodel_{}_{}.json'.format(iter,model_name)))

    def make_noise(self, oriset, noise='SVHN',ix=0):

        uni_nm = len(oriset)
        num_all_noise = int(uni_nm * args.r)
        num_open_noise = int(num_all_noise * args.on)
        if noise == 'cifar100':
            noise_data = unpickle('data/cifar-100/cifar-100-python/train')['data']
            noise_data = noise_data.reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))
        elif noise == 'SVHN':
            noise_data = sio.loadmat('data/SVHN/train_32x32.mat')
            noise_data = noise_data['X']
            noise_data = noise_data.transpose((3, 0, 1, 2))

        labels = copy.deepcopy(oriset.labels)
        images = oriset.samples
        idx = list(range(uni_nm))
        target_noise_idx = list(range(uni_nm))
        random.shuffle(target_noise_idx)
        # ========asymmetric noise=========
        if args.mode == 'asym':
            cr = args.r-args.r*args.on
            closed_idx = []
            noise_mapping = [2,9,0,5,7,3,6,4,8,1]
            noise_matrix = np.zeros((args.num_classes,args.num_classes),dtype=float)
            for i in range(args.num_classes):
                if i == noise_mapping[i]:
                    noise_matrix[i][i] = 1
                else:
                    for j in range(args.num_classes):
                        if j == noise_mapping[i]:
                            noise_matrix[i][j] = cr
                        else:
                            if j == i:
                                noise_matrix[i][j] = 1-cr
                            else:
                                noise_matrix[i][j] = 0
            print('noise_matrix:')
            print(noise_matrix)
            for i in range(len(labels)):
                labels[i] = np.random.choice(np.arange(args.num_classes),1,p=noise_matrix[labels[i]])
                if labels[i] != oriset.labels[i]:
                        closed_idx.append(i)
            open_idx = set(idx)-set(closed_idx)
            open_idx = list(open_idx)
            random.shuffle(open_idx)
            open_idx = open_idx[:num_open_noise]
        else:
            # ========symmetric noise=========
            random.shuffle(idx)
            open_idx = idx[:num_open_noise]
            closed_idx = idx[num_open_noise:num_all_noise]
            for i in range(uni_nm):
                if i in closed_idx:
                    rag = list(range(args.num_classes))
                    rag.remove(labels[i])
                    labels[i] = random.choice(rag)
        #
        # self.clients[ix].conf_mat(debug1,debug)
        open_map = list(zip(open_idx, target_noise_idx[:num_open_noise]))
        for cleanIdx, noisyIdx in open_map:
            images[cleanIdx] = noise_data[noisyIdx]

        global noise_chart_op, noise_chart_cl
        noise_chart_op[ix] = [True if i in open_idx else False for i in range(uni_nm)]
        noise_chart_op[ix] = np.array(noise_chart_op[ix])
        noise_chart_cl[ix] = [True if i in closed_idx else False for i in range(uni_nm)]
        noise_chart_cl[ix] = np.array(noise_chart_cl[ix])
        newset = data_loader(images, labels,args)
        return newset

    def bench_left(self, train_dataset):
        bench_idxs = bench_assign(train_dataset)
        X, Y = train_dataset.samples[bench_idxs], train_dataset.labels[bench_idxs]
        idxs_all = list(range(len(train_dataset)))
        idxs_left = list(set(idxs_all) - set(bench_idxs))
        X_left, Y_left = train_dataset.samples[np.array(idxs_left)], train_dataset.labels[np.array(idxs_left)]
        left_dataset = data_loader(X_left, Y_left,args)
        bench_dataset = data_loader(X, Y, args)
        return left_dataset, bench_dataset

    def prepare_data(self):
        if args.dataset == 'cifar10':
            train_dataset, test_dataset = get_dataset(args)

            left_dataset, bench_dataset = self.bench_left(train_dataset)
            Usr_dataset = get_Users_Data(args, left_dataset)

        elif args.dataset == 'camelyon17':
            Usr_dataset,test_dataset = get_Users_Data(args)
            test_dataset,bench_dataset = self.bench_left(test_dataset)
        # new_dataset = self.make_noise(left_dataset)
        # global noise_chart_cl, noise_chart_op
        for p_id in range(args.num_users):
            with open(os.path.join(clientPath, 'clean_labels' + str(p_id) + '.pkl'), 'wb') as f:
                pickle.dump(Usr_dataset[p_id].labels, f)

        for i in range(args.num_users):
            if noise_idx[i]:
                Usr_dataset[i] = self.make_noise(Usr_dataset[i], noise=args.noisy_dataset,ix=i)

        Path(clientPath).mkdir(parents=True, exist_ok=True)
        Path(serverPath).mkdir(parents=True, exist_ok=True)
        for p_id in range(args.num_users):
            with open(os.path.join(clientPath, 'dataset' + str(p_id) + '.pkl'), 'wb') as f:
                pickle.dump(Usr_dataset[p_id], f)
        with open(os.path.join(serverPath, 'testdata.pkl'), 'wb') as f:
            pickle.dump(test_dataset, f)
        with open(os.path.join(serverPath, 'benchdata.pkl'), 'wb') as f:
            pickle.dump(bench_dataset, f)
        with open(os.path.join(serverPath, 'noise_chart_op.pkl'), 'wb') as f:
            pickle.dump(noise_chart_op, f)
        with open(os.path.join(serverPath, 'noise_chart_clo.pkl'), 'wb') as f:
            pickle.dump(noise_chart_cl, f)


    def logDetectIndex(self, epoch):
        right_clean,right_close,right_open = 0,0,0
        num_clean,num_close,num_open = 0,0,0
        pred_clean,pred_close,pred_open = 0,0,0
        for ix in range(args.num_users):
            preds = self.clients[ix].preds
            clean,open,close = self.clients[ix].get_noise()
            num_clean += np.sum(clean)
            num_open += np.sum(open)
            num_close += np.sum(close)
            # open_ratio = num_open/len(clean)
            # close_ratio = num_close/len(clean)
            equal = preds[0]==clean
            right_clean += np.sum(np.array(clean)[equal])
            right_open += np.sum(np.array(open)[preds[1]==open])
            right_close += np.sum(np.array(close)[preds[2]==close])
            pred_clean += np.sum(preds[0])
            pred_open += np.sum(preds[1])
            pred_close += np.sum(preds[2])

        right_noise = right_open+right_close
        num_noise = num_open+num_close
        recall_noise = round(right_noise/num_noise,4)
        precision_noise = right_noise/(pred_open+pred_close)
        f1score_noise = (recall_noise+precision_noise)/2

        recall_clean = round(right_clean/num_clean,4)
        precision_clean = right_clean/pred_clean
        f1score_clean = (recall_clean+precision_clean)/2

        recall_open = right_open/num_open
        precision_open = right_open/pred_open
        f1score_open = (recall_open+precision_open)/2

        recall_close = round(right_close/num_close,4)
        precisions_close = right_close/pred_close
        f1score_close = (recall_close+precisions_close)/2

        writer.add_scalars('Attacker_Index/clean',
                           {'recall': recall_clean, 'precision': precision_clean,
                            'f1score': f1score_clean}, epoch)
        writer.add_scalars('Attacker_Index/open',
                           {'recall': recall_open, 'precision': precision_open,
                            'f1score': f1score_open}, epoch)
        writer.add_scalars('Attacker_Index/close',
                           {'recall': recall_close, 'precision': precisions_close,
                            'f1score': f1score_close}, epoch)

    def logCorrectIndex(self, epoch):
        right = 0
        revision = 0
        num_close = 0
        cRat,oRat,clRat = 0,0,0
        for ix in range(args.num_users):
            Y_raw = self.clients[ix].dataset.labels
            Y_corr = self.clients[ix].sudo_labels
            Y_real = self.clients[ix].clean_labels
            clean,open,close = self.clients[ix].get_noise()
            revision += np.sum(Y_raw!=Y_corr)
            num_close += np.sum(close)
            for i in range(len(Y_raw)):
                if Y_corr[i]!= Y_raw[i]:
                    if clean[i]:
                        cRat += 1
                    elif open[i]:
                        oRat += 1
                    elif close[i]:
                        clRat += 1
                    if Y_corr[i]==Y_real[i]:
                        right += 1

        recall = right/num_close
        precision = right/revision
        f1score = (recall+precision)/2

        writer.add_scalars('Attacker_Index/refine',
                           {'recall': recall, 'precision': precision,
                            'f1score': f1score}, epoch)

        writer.add_scalars('Attacker_Index/refine_compo',
                           {'clean': cRat, 'open': oRat,
                            'closed': clRat}, epoch)


    def runExperiment(self):

        weights_C = [[] for i in range(args.num_users)]
        Keep_size_C = [0] * args.num_users

        tag1_lst, tag2_lst = [0] * args.num_users, [0] * args.num_users
        diff_1, diff_2 = [0] * args.num_users, [0] * args.num_users

        for epoch in range(args.start_epoch, args.num_epochs + 1):

            if epoch==args.start_epoch:
                if not args.autoload:
                    all_gmm = []
                    for ix in range(args.num_users):
                        each_gmm = self.clients[ix].eval_train()
                        all_gmm.append(each_gmm)
                    agg_gmm= self.server.merge_gmm_samp(all_gmm)

                    for ix in range(args.num_users):
                        t1, t2 = self.server.compute_lamda_gmm_samp(agg_gmm, all_gmm[ix],u_idx=ix)
                        d1, d2 = t1 - tag1_lst[ix], t2 - tag2_lst[ix]
                        diff_1[ix], diff_2[ix] = d1,d2
                        tag1_lst[ix], tag2_lst[ix] = t1,t2
                        self.clients[ix].eval_train(tag1=tag1_lst[ix],tag2=tag2_lst[ix])
                        if noise_idx[ix]:
                            self.clients[ix].performance(epoch)

                else:
                    for ix in range(args.num_users):

                        with open(os.path.join(checkpoint_dir,
                                               'probs_%d_t1=%.2f_t2=%.2f.pkl' % (ix, tag1_lst[ix], tag2_lst[ix])),
                                  'rb') as f:
                            self.clients[ix].probs = pickle.load(f)
                        with open(os.path.join(checkpoint_dir,
                                               'preds_%d_t1=%.2f_t2=%.2f.pkl' % (ix, tag1_lst[ix], tag2_lst[ix])),
                                  'rb') as f:
                            self.clients[ix].preds = pickle.load(f)
                        with open(os.path.join(checkpoint_dir, 'sbj_loss_%s.pkl' % (str(ix))), 'rb') as f:
                            self.clients[ix].sbj_loss = pickle.load(f)

            for ix in range(args.num_users):
               
                if len(self.clients[ix].preds[2].nonzero()[0]) == 0 :
                    mixmatch = False
                else:
                    mixmatch = True
                    if args.dataset == 'camelyon17':
                        sbj_arg = np.argsort(self.clients[ix].sbj_loss)
                        self.clients[ix].preds[0][sbj_arg[0]] = True
                        self.clients[ix].preds[0][sbj_arg[1]] = True
                        self.clients[ix].preds[2][sbj_arg[-1]] = True
                        self.clients[ix].preds[2][sbj_arg[-2]] = True

                if mixmatch:
                    self.clients[ix].labeled_load()
                    self.clients[ix].unlabeled_load()
                    for iter in range(10):
                        print('\nTrain netD')
                        weights_C[ix], Keep_size_C[ix] = self.clients[ix].local_Mixmatch(epoch, iter)
                    self.clients[ix].refine_labels_new(epoch=epoch, log=False)
                else:
                    print('\nTrain netD')
                    weights_C[ix], Keep_size_C[ix] = self.clients[ix].local_update(epoch, 'netC')
                if ix == args.num_users - 1:
                    self.logCorrectIndex(epoch)

            print('Global model of epoch={} aggregating...'.format(epoch))
            for ix in range(args.num_users):
                with open(os.path.join(clientPath, 'probs_%d_t1=%.2f_t2=%.2f.pkl' % (ix, tag1_lst[ix], tag2_lst[ix])),
                          'wb') as f:
                    pickle.dump(self.clients[ix].probs, f)
                with open(os.path.join(clientPath, 'preds_%d_t1=%.2f_t2=%.2f.pkl' % (ix, tag1_lst[ix], tag2_lst[ix])),
                          'wb') as f:
                    pickle.dump(self.clients[ix].preds, f)

            self.server.FedAvg(weights_C, Keep_size_C, 'netC')
            self.server.sendmodel()
            self.server.test(epoch, 'netC')

            for i in range(args.num_users):
                self.clients[i].receivemodel()
            if epoch % 5 == 0:
                torch.save({
                    'netB_state_dict': self.server.model_B.state_dict(),
                    'netC_state_dict': self.server.model_C.state_dict(),
                }, os.path.join(checkpoint_dir, 'train_model_%d.json' % (epoch)))
            if self.server.stop:
                break


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def linear_rampup(current, rampup_length=16):
    current = np.clip((current) / rampup_length, 0.0, 1.0)  # 将元素限制在0.0和1.0之间
    return args.lambda_u * float(current)


def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total,used = devices_info[int(cuda_device)].split(',')
    return total,used

def occupy_mem(cuda_device):
    total,used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total*0.9)
    block_mem = 1000
    x = torch.cuda.FloatTensor(256,1024,block_mem)
    print('===========claim GPU mem for '+str(block_mem)+'Mib==========')
    del x

if __name__ == '__main__':
    occupy_mem(cuda_device)

    setup_seed(42)

    flm = FedMIN()
    flm.runExperiment()

