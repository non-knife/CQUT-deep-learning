#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader,Dataset
torch.set_default_tensor_type(torch.DoubleTensor)

iris=load_iris()
X=iris.data
Y=iris.target


train_X,test_X,train_y,test_y=train_test_split(X,Y,test_size=0.3,random_state=2022)

train_X=torch.from_numpy(train_X)
test_X=torch.from_numpy(test_X)
train_y=torch.from_numpy(train_y).long()
test_y=torch.from_numpy(test_y).long()
# test_X=test_X.tensor_type(torch.LongTensor)



class Data(Dataset):
    def __init__(self):
        self.x=train_X
        self.y=train_y
        self.len=self.x.shape[0]
    def __getitem__(self, index):
        return self.x[index],self.y[index]
    def __len__(self):
        return self.len

data_set=Data()
trainloader=DataLoader(dataset=data_set,batch_size=64)
print(data_set.x.shape,data_set.y.shape)

class Net(nn.Module):
    def __init__(self,input_,hidden_,output_):
        super(Net, self).__init__()
        self.linear1=nn.Linear(input_,hidden_)
        self.linear2=nn.Linear(hidden_,10)
        self.linear3=nn.Linear(10,output_)
    def forward(self,x):
        x=torch.sigmoid(self.linear1(x))
        x=torch.sigmoid(self.linear2(x))
        x=self.linear3(x)
        return x

input_dim=4
hidden_dim=30
output_dim=3
# Net=Net.double()
model=Net(input_dim,hidden_dim,output_dim)
# print(model)
loss_fn=nn.CrossEntropyLoss()
# optimizer=torch.optim.SGD(model.parameters(),lr=0.0001,momentum=0.8)
optimizer=torch.optim.Adam(model.parameters(),lr=0.005)

n_epochs=1001
loss_list=[]
i=0
for epoch in range(n_epochs):
    loss_sum=0.0
    for x,y in trainloader:
        optimizer.zero_grad()
        pre=model(x)
        loss=loss_fn(pre,y)
        loss.backward()
        optimizer.step()
        loss_sum+=loss.item()
    loss_list.append(loss_sum/len(trainloader))
    if epoch%100 == 0:
        print('epoch {}, loss {}'.format(epoch, loss_list[epoch]))
        
    
#%%
# 保存模型
torch.save(model.state_dict(),'Model.path')

#%%
# 绘制图形
import matplotlib.pyplot as plt
draw_loss=[]
for i in range(n_epochs):
    if i % 100 == 0:
        draw_loss.append(loss_list[i])
x=[1,100,200,300,400,500,600,700,800,900,1000]
plt.plot(x,draw_loss)
plt.xlabel("Epoch")
plt.ylabel("loss")
plt.show()

#%%
# 加载模型
model=Net(input_dim,hidden_dim,output_dim)
model.load_state_dict(torch.load('Model.path'))
model.eval()

#%%
out = model(test_X) 
prediction = torch.max(out, 1)[1] 
pred_y = prediction.data.numpy()
target_y = test_y.data.numpy()

accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
print("莺尾花预测准确率",accuracy)



        
        
        
        
        
        
        
        
        
        
        
        
