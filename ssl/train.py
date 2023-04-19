import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import nn, optim
from capsule_dataset import MVTEC_Dataset
from conv_auto_encoder import ConvAutoEncoder
from tqdm import tqdm

EPOCH = 10
BATCH_SIZE = 5
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

def main():
    trainset = MVTEC_Dataset('capsule', 
                             'train')
    train_loader = torch.utils.data.DataLoader(trainset, 
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True, 
                                               num_workers=2)
    
    model = ConvAutoEncoder().to(DEVICE)
    critirion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), 
                           lr=0.001)
    
    model.train()
    for epoch in range(EPOCH):
        print('epoch :: ', epoch, ' / ', EPOCH)
        
        for images, _ in tqdm(train_loader):
            inputs = images.to(DEVICE)
            outputs = model(inputs)
            loss = critirion(outputs, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print('loss :: ', loss.item())    
    
    torch.save(model.state_dict(), './model/conv_auto_encoder.pth')
    print('model saved')

if __name__ == '__main__':
    main()