import torch.nn as nn

# https://esther-eun27.tistory.com/4
class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        
        # Encoder
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
            )

        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
            )

        # Decoder
        self.tran_cnn_layer1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size = 2, stride = 2, padding=0),
            nn.ReLU()
            )

        self.tran_cnn_layer2 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size = 2, stride = 2, padding=0),
            nn.Sigmoid()
            )
            
            
    def forward(self, x):
        output = self.cnn_layer1(x)
        output = self.cnn_layer2(output)
        output = self.tran_cnn_layer1(output)
        output = self.tran_cnn_layer2(output)

        return output
    
if __name__ == '__main__':
    model = ConvAutoEncoder()
    print(model)