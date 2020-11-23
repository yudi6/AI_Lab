# Code in file autograd/two_layer_net_autograd.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data_loader_no_validate import trainDataset
from data_loader_no_validate import validateDataset
from data_loader_no_validate import testDataset
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from resnet import ResNet18



#***********************************超参数
learning_rate=0.01
max_epoch=150
#***********************************
if torch.cuda.is_available():
    device = torch.device('cuda') # Uncomment this to run on GPU
else:
    device = torch.device('cpu')
class VGG16(nn.Module):
    
    
    def __init__(self):
        super(VGG16, self).__init__()
        
        # 3 * 32 * 32
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=(1, 1)) # 64 * 32 *32
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1)) # 64 * 32* 32
        self.maxpool1 = nn.MaxPool2d((2, 2)) # pooling 64 * 16 * 16
        
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=(1, 1)) # 128 * 16 * 16
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1)) # 128 * 16 * 16
        self.maxpool2 = nn.MaxPool2d((2, 2)) # pooling 128 * 8 * 8
        
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=(1, 1)) # 256 * 8 * 8
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1)) # 256 * 8 * 8
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1)) # 256 * 8 * 8
        self.maxpool3 = nn.MaxPool2d((2, 2)) # pooling 256 * 4 * 4
        
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=(1, 1)) # 512 * 4 * 4
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 4 * 4
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 4 * 4
        self.maxpool4 = nn.MaxPool2d((2, 2)) # pooling 512 * 2 * 2
        
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 2 * 2
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 2 * 2
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 2 * 2
        self.maxpool5 = nn.MaxPool2d((2, 2)) # pooling 512 * 1 * 1
        
        # view
        
        self.fc1 = nn.Linear(512 * 1 * 1, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(256 * 4 * 4, 10)
        self.dropout = nn.Dropout(p=0.5)
        # softmax 1 * 1 * 1000
        
    def forward(self, x):
        
        # x.size(0)即为batch_size
        in_size = x.size(0)
        
        out = self.conv1_1(x) # 222
        out = F.relu(out)
        out = self.conv1_2(out) # 222
        out = F.relu(out)
        out = self.maxpool1(out) # 112
        
        out = self.conv2_1(out) # 110
        out = F.relu(out)
        out = self.conv2_2(out) # 110
        out = F.relu(out)
        out = self.maxpool2(out) # 56
        
        out = self.conv3_1(out) # 54
        out = F.relu(out)
        out = self.conv3_2(out) # 54
        out = F.relu(out)
        out = self.conv3_3(out) # 54
        out = F.relu(out)
        out = self.maxpool3(out) # 28

        
        # 展平
        out = out.view(in_size, -1)
        out=self.dropout(out)
        out = self.fc3(out)
        
        return out 
    def initial_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight,gain=nn.init.calculate_gain('relu'))

class lenet(nn.Module):
    def __init__(self):
        super(lenet,self).__init__()#初始化父类
        self.conv1 = nn.Conv2d(3,6,5)#第一层卷积层
        self.conv2 = nn.Conv2d(6,16,5)#第二层卷积层
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,10)
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))#卷积+池化
        x = F.max_pool2d(F.relu(self.conv2(x)),2)#卷积+池化
        x = x.view(x.size()[0],-1)#展平向量到一维
        x = F.relu(self.fc1(x))#第一层全连接+relu
        x = self.fc2(x)#第二层全连接
        return  x
    def initial_weight(self):#初始化参数
        for m in self.modules():#采用xavier初始化法，初始化每一层参数
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight,gain=nn.init.calculate_gain('relu'))

       
# model=lenet()
# model=VGG16()
model = ResNet18()
# model.initial_weight()
# torch.save(model.state_dict(),'./net1.pt')

# model=VGG19Net()
model=model.to(device)
model.load_state_dict(torch.load('./resnet18.pt'))
# optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,betas=[0.9,0.999])
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
loss_fn = nn.CrossEntropyLoss()    
transform_train = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])
trainDataDealer = trainDataset(transform=transform_train)
validateDataDealer=validateDataset(transform=transform_test)
testDataDealer = testDataset(transform=transform_test)

train_loader = DataLoader(dataset=trainDataDealer,
                          batch_size=64,
                          num_workers=8,
                          shuffle=True)
validate_loader = DataLoader(dataset=validateDataDealer,
                          batch_size=64,
                          num_workers=8,
                          shuffle=True)
test_loader = DataLoader(dataset=testDataDealer,
                          batch_size=64,
                          shuffle=False)


model.eval()
total=0
correct=0
for i, data in enumerate(test_loader):
    img,label=data
    img=img.to(device)
    label=label.to(device)
    img=Variable(img)
    label=Variable(label)
    y_pred=model(img)
    if i==0:
        print(y_pred.data[0])
        print(y_pred.data[1])
    _, predicted = torch.max(y_pred.data, 1)
    total +=label.size(0)
    correct +=(predicted == label).sum().item()
print(correct/total)
