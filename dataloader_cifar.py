from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random
import numpy as np
from PIL import Image
import json
import os
import scipy.io as sio
import torch
from torchnet.meter import AUCMeter
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold as skf
import pdb
import pickle

            
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

def load_user_idx(train_data,train_label,savepath):
    sfolder = skf(n_splits=4, shuffle=False)
    ix = 0
    if not os.path.exists(savepath):
        for train_idx, test_idx in sfolder.split(train_data, train_label):
            savepath_sub = './noise/skfCifar_user_v4={}_idx.pkl'.format(ix)
            with open(savepath_sub, 'wb') as f:
                pickle.dump(test_idx, f)
            ix += 1
    if os.path.exists(savepath):
        with open(savepath, 'rb') as f:
            each_idx = pickle.load(f)
    return each_idx

class cifar_dataset(Dataset): 
    def __init__(self, u_index, dataset, noisy_dataset, noise_mode, r, on, root_dir, noise_data_dir, transform, mode, noise_file='', pred=[], probability=[], log='', targets=None):
        
        self.r = r # total noise ratio
        self.on = on # proportion of open noise
        self.transform = transform
        self.mode = mode  
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
        self.open_noise = None
        self.closed_noise = None
        self.u_index = u_index
        #------------专属于mnist的加载格式---------------
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        data_dir = './data/mnist/'

        if self.mode=='test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['fine_labels']
            elif dataset=='mnist':

                test_dataset = datasets.MNIST(data_dir, train=False, download=False,
                                              transform=apply_transform)
                self.test_data = test_dataset.data.numpy()
                self.test_label = test_dataset.targets.numpy()
                debug = True

       
        elif self.mode=='clean':
            if not os.path.exists(noise_file):
                print('Noise not defined')
                return
            
            if self.open_noise is None or self.closed_noise is not None:
                noise = json.load(open(noise_file,"r"))
                noise_labels = noise['noise_labels']
                self.open_noise = noise['open_noise']
                self.closed_noise = noise['closed_noise']

            train_data=[]
            train_label=[]
            noise_data=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))
            savepath = './noise/skfCifar_user_v4={}_idx.pkl'.format(self.u_index)
            each_idx = load_user_idx(train_data, train_label, savepath)
            train_label = np.array(train_label)[each_idx]
            train_label = train_label.tolist()
            train_data = train_data[each_idx]
            self.user_datlen = len(train_label)

            open_noise = [item[0] for item in self.open_noise]
            # clean_indices = list(set(range(12500)) - set(open_noise) - set(self.closed_noise))
            # self.clean_data = train_data[clean_indices]
            self.clean_label = np.asarray(train_label)

        else:    
            train_data=[]
            train_label=[]
            noise_data=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
            elif dataset=='mnist':
                train_dataset = datasets.MNIST(data_dir, train=True, download=False,
                                               transform=apply_transform)
                train_data = train_dataset.data
                train_label = train_dataset.targets
                train_data = train_data.numpy()
                train_label = train_label.numpy()

            #------------对数据按分层抽样进行划分-------------------------------
            # savepath = './noise/skfCifar_user_v4={}_idx.pkl'.format(self.u_index)
            # each_idx = load_user_idx(train_data,train_label,savepath)


            # train_label = np.array(train_label)
            # train_label = np.array(train_label)[each_idx]
            # train_label = train_label.tolist()
            # train_data = train_data[each_idx]
            # self.user_datlen = len(train_label)

            #---------------原始数据加载方式-------------------------------------------------
            # train_label = train_label[self.u_index*self.user_datlen:(self.u_index+1)*self.user_datlen]
            # train_data = train_data[self.u_index*self.user_datlen:(self.u_index+1)*self.user_datlen]
            from collections import Counter
            see = Counter(train_label)
            train_data = train_data.reshape((-1, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))
            if noisy_dataset == 'imagenet32':
                noise_data = None
            else:
                noise_data = unpickle('%s/train'%noise_data_dir)['data']
            self.user_datlen = len(train_data)
            noise_data = noise_data[self.u_index*self.user_datlen:(self.u_index+1)*self.user_datlen]
            noise_data = noise_data.reshape((self.user_datlen, 3, 32, 32)).transpose((0, 2, 3, 1))

            if os.path.exists(noise_file):
                noise = json.load(open(noise_file,"r"))
                noise_labels = noise['noise_labels']
                self.open_noise = noise['open_noise']
                self.closed_noise = noise['closed_noise']
                for cleanIdx, noisyIdx in noise['open_noise']:
                    if noisy_dataset == 'imagenet32':
                        train_data[cleanIdx] = np.asarray(Image.open('{}/{}.png'.format(noise_data_dir, str(noisyIdx+1).zfill(7)))).reshape((32,32,3))
                    else:
                        train_data[cleanIdx] = noise_data[noisyIdx]
            else:
                #inject noise   
                noise_labels = []                       # all labels (some noisy, some clean)
                idx = list(range(self.user_datlen))                # indices of cifar dataset
                random.shuffle(idx)                 
                num_total_noise = int(self.r*self.user_datlen)     # total amount of noise
                num_open_noise = int(self.on*num_total_noise)     # total amount of noisy/openset images
                if noisy_dataset == 'imagenet32':       # indices of openset source images
                    target_noise_idx = list(range(1281149))
                else:
                    target_noise_idx = list(range(self.user_datlen))
                random.shuffle(target_noise_idx)  
                self.open_noise = list(zip(idx[:num_open_noise], target_noise_idx[:num_open_noise]))  # clean sample -> openset sample mapping
                self.closed_noise = idx[num_open_noise:num_total_noise]      # closed set noise indices
                # populate noise_labels
                for i in range(self.user_datlen):
                    if i in self.closed_noise:
                        if noise_mode=='sym':
                            if dataset=='cifar10': 
                                noiselabel = random.randint(0,9)
                            elif dataset=='cifar100':    
                                noiselabel = random.randint(0,99)
                            noise_labels.append(noiselabel)
                        elif noise_mode=='asym':   
                            noiselabel = self.transition[train_label[i]]
                            noise_labels.append(noiselabel)               
                    else:
                        noise_labels.append(train_label[i])
                # populate openset noise images
                for cleanIdx, noisyIdx in self.open_noise:
                    if noisy_dataset == 'imagenet32':
                        train_data[cleanIdx] = np.asarray(Image.open('{}/{}.png'.format(noise_data_dir, str(noisyIdx+1).zfill(7)))).reshape((32,32,3))
                    else:
                        train_data[cleanIdx] = noise_data[noisyIdx]
                # # write noise to a file, to re-use
                noise = {'noise_labels': noise_labels, 'open_noise': self.open_noise, 'closed_noise': self.closed_noise}
                print("save noise to %s ..."%noise_file)
                json.dump(noise,open(noise_file,"w"))
            
            if self.mode == 'all':
                if len(pred)!=0:
                    pred = [not i for i in pred]
                    self.train_data = train_data[pred]
                    self.noise_labels = targets[pred]
                else:
                    self.train_data = train_data
                    if targets is None:
                        self.noise_labels = noise_labels
                    else:
                        self.noise_labels = targets
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]
                    
                    clean = (np.array(noise_labels)==np.array(train_label))                                                    
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability,clean)
                    # note: If all the labels are clean, the following will return NaN       
                    auc,_,_ = auc_meter.value()                     
                    
                elif self.mode == "unlabeled":
                    pred_idx = pred.nonzero()[0]                                               
                
                self.train_data = train_data[pred_idx]
                self.noise_labels = [noise_labels[i] for i in pred_idx]                          
                print("%s data has a size of %d"%(self.mode,len(self.noise_labels)))            
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_labels[index], self.probability[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2, target, prob            
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            img = Image.fromarray(img)
            img1 = self.transform(img) 
            img2 = self.transform(img) 
            return img1, img2
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_labels[index]
            img = Image.fromarray(img)
            img = self.transform(img)               
            return img, target, index        
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
        elif self.mode=='clean':
            img, target = self.clean_data[index], self.clean_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode=='clean':
            return len(self.clean_data)
        elif self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)

    def get_noise(self):
        return (self.open_noise, self.closed_noise)       
        
        
class cifar_dataloader():  
    def __init__(self, args, u_index,log, noise_file=''):

        self.dataset = args.dataset
        self.r = args.r
        self.on = args.on
        self.noise_mode = args.noise_mode
        self.batch_size = args.batch_size
        self.num_workers = 1
        self.root_dir = args.data_path
        self.noise_data_dir = args.noise_data_dir
        self.log = log
        self.noise_file = noise_file
        self.open_noise = None
        self.closed_noise = None
        self.noisy_dataset = args.noisy_dataset
        self.u_index = u_index
        self.user_datlen = int(50000/args.num_users)
        self.clean_label = []

        if self.dataset=='cifar10':
            # todo: normalise the noise dataset properly
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # 数据增强所用的归一化
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                ])    
        elif self.dataset=='cifar100':    
            self.transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]) 
            self.transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ])
        elif self.dataset=='mnist':
            self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
            ])
            self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
            ])

    def run(self,mode,predClean=[], predClosed=[],predOpen=[], probClean=[], targets=None):
        if mode=='warmup':
            all_dataset = cifar_dataset(u_index=self.u_index, dataset=self.dataset, noisy_dataset=self.noisy_dataset, noise_mode=self.noise_mode, r=self.r, on=self.on, root_dir=self.root_dir, noise_data_dir=self.noise_data_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file)
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
            debug = all_dataset

            img_show = all_dataset.train_data[0]
            plt.imshow(img_show)
            plt.savefig('cifar.png')

            self.open_noise, self.closed_noise = all_dataset.get_noise()
            return trainloader
                                     
        elif mode=='trainD':
            # 传入干净数据的分布以及每个样本的概率
            labeled_dataset = cifar_dataset(u_index=self.u_index,dataset=self.dataset, noisy_dataset=self.noisy_dataset, noise_mode=self.noise_mode, r=self.r, on=self.on, root_dir=self.root_dir, noise_data_dir=self.noise_data_dir, transform=self.transform_train, mode="labeled", noise_file=self.noise_file, pred=predClean, probability=probClean,log=self.log)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = cifar_dataset(u_index=self.u_index, dataset=self.dataset, noisy_dataset=self.noisy_dataset, noise_mode=self.noise_mode, r=self.r, on=self.on, root_dir=self.root_dir, noise_data_dir=self.noise_data_dir, transform=self.transform_train, mode="unlabeled", noise_file=self.noise_file, pred=predClosed)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)    

            self.open_noise, self.closed_noise = labeled_dataset.get_noise()  
            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='trainS':
            all_dataset = cifar_dataset(u_index=self.u_index, dataset=self.dataset, noisy_dataset=self.noisy_dataset, noise_mode=self.noise_mode, r=self.r, on=self.on, root_dir=self.root_dir, noise_data_dir=self.noise_data_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file, targets=targets,pred=predOpen)
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)           
            return trainloader
        
        elif mode=='test':
            test_dataset = cifar_dataset(u_index=self.u_index, dataset=self.dataset, noisy_dataset=self.noisy_dataset, noise_mode=self.noise_mode, r=self.r, on=self.on, root_dir=self.root_dir, noise_data_dir=self.noise_data_dir, transform=self.transform_test, mode='test')
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='clean':
            clean_dataset = cifar_dataset(u_index=self.u_index, dataset=self.dataset, noisy_dataset=self.noisy_dataset, noise_mode=self.noise_mode, r=self.r, on=self.on, root_dir=self.root_dir, noise_data_dir=self.noise_data_dir, transform=self.transform_test, mode='clean', noise_file=self.noise_file)
            clean_loader = DataLoader(
                dataset=clean_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return clean_loader
        
        elif mode=='eval_train':
            eval_dataset = cifar_dataset(u_index=self.u_index, dataset=self.dataset, noisy_dataset=self.noisy_dataset, noise_mode=self.noise_mode, r=self.r, on=self.on, root_dir=self.root_dir, noise_data_dir=self.noise_data_dir, transform=self.transform_test, mode='all', noise_file=self.noise_file)
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
            self.open_noise, self.closed_noise = eval_dataset.get_noise()         
            return eval_loader        
    
    def get_noise(self):
        open_noise = [item[0] for item in self.open_noise]
        clean = list(set(range(self.user_datlen)) - set(open_noise) - set(self.closed_noise))
        return (clean, open_noise, self.closed_noise)
