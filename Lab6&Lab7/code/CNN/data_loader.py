import torch.nn.functional as F
import torch
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import random
import pickle
from PIL import Image 



#数据集的设置*****************************************************************************************************************
root =os.getcwd()+ '/../data/'#调用图像
# root='/home/zyc/Desktop/ai_midterm_project/cnn/data/'

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
#定义读取文件的格式
def default_loader(path):
    return Image.open(path).convert('RGB')

#首先继承上面的dataset类。然后在__init__()方法中得到图像的路径，然后将图像路径组成一个数组，这样在__getitim__()中就可以直接读取：
class trainDataset(Dataset): #创建自己的类：trainDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, transform=None,target_transform=None, loader=default_loader): #初始化一些需要传入的参数
        super(trainDataset,self).__init__()#对继承自父类的属性进行初始化
        data_set =unpickle(root+'cifar-10-python/cifar-10-batches-py/data_batch_1')#按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = data_set[str.encode('data')]
        labels=np.array(data_set[str.encode('labels')])

        data_set =unpickle(root+'cifar-10-python/cifar-10-batches-py/data_batch_2')#按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = np.concatenate((imgs,data_set[str.encode('data')]))
        labels=np.concatenate((labels,np.array(data_set[str.encode('labels')])))

        data_set =unpickle(root+'cifar-10-python/cifar-10-batches-py/data_batch_3')#按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = np.concatenate((imgs,data_set[str.encode('data')]))
        labels=np.concatenate((labels,np.array(data_set[str.encode('labels')])))
        
        data_set =unpickle(root+'cifar-10-python/cifar-10-batches-py/data_batch_4')#按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = np.concatenate((imgs,data_set[str.encode('data')]))
        labels=np.concatenate((labels,np.array(data_set[str.encode('labels')])))

        data_set =unpickle(root+'cifar-10-python/cifar-10-batches-py/data_batch_5')#按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = np.concatenate((imgs,data_set[str.encode('data')]))
#         imgs=imgs/255
        imgs=imgs.astype('uint8')
        labels=np.concatenate((labels,np.array(data_set[str.encode('labels')])))

        # imgs=imgs[0:10]
        # labels=labels[0:10]
#         imgs=Image.fromarray(imgs)
        self.imgs = imgs.reshape(50000,3,32,32).transpose(0,2,3,1)
        self.labels=labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader        
        
    def __getitem__(self, index):#这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        temp_img=self.imgs[index]
        temp_label=self.labels[index]

#         temp_img=temp_img.reshape([32,32,3])
        if self.transform is not None:
            temp_img = self.transform(temp_img) #数据标签转换为Tensor
        return temp_img,temp_label#return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return self.imgs.shape[0]

class validateDataset(Dataset): #创建自己的类：trainDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, transform=None,target_transform=None, loader=default_loader): #初始化一些需要传入的参数
        super(validateDataset,self).__init__()#对继承自父类的属性进行初始化
        data_set =unpickle(root+'cifar-10-python/cifar-10-batches-py/test_batch')#按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = data_set[str.encode('data')]
#         imgs=imgs/255
#         imgs=imgs.astype('float32')
        labels=np.array(data_set[str.encode('labels')])

        # imgs=imgs[0:100]
        # labels=labels[0:100]
        self.imgs = imgs.reshape(10000,3,32,32).transpose(0,2,3,1)
        self.labels=labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader        
        
    def __getitem__(self, index):#这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        temp_img=self.imgs[index]
        temp_label=self.labels[index]
#         temp_img=temp_img.reshape([32,32,3])
        if self.transform is not None:
            temp_img = self.transform(temp_img) #数据标签转换为Tensor
        return temp_img,temp_label#return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return self.imgs.shape[0]

class testDataset(Dataset): #创建自己的类：trainDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, transform=None,target_transform=None, loader=default_loader): #初始化一些需要传入的参数
        super(testDataset,self).__init__()#对继承自父类的属性进行初始化
        data_set =unpickle(root+'cifar-10-python/cifar-10-batches-py/test_batch')#按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = data_set[str.encode('data')]
#         imgs=imgs/255
#         imgs=imgs.astype('float32')
        labels=np.array(data_set[str.encode('labels')])
        

        self.imgs = imgs.reshape(10000,3,32,32).transpose(0,2,3,1)
        self.labels=labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader        
        
    def __getitem__(self, index):#这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        temp_img=self.imgs[index]
        temp_label=self.labels[index]
        temp_img=temp_img.reshape([32,32,3])
        if self.transform is not None:
            temp_img = self.transform(temp_img) #数据标签转换为Tensor
        return temp_img,temp_label#return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return self.imgs.shape[0]