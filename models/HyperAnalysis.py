import math
import torch.nn as nn
import torch


class Hyper_analysis(nn.Module):
    def __init__(self, num_filters=128):
        super(Hyper_analysis, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.leaky_relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.leaky_relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1)
        self.leaky_relu3 = nn.LeakyReLU()
        self.conv4 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.leaky_relu4 = nn.LeakyReLU()
        self.conv5 = nn.Conv2d(num_filters, num_filters, 3, stride=2, padding=1)

    
    def forward(self, x):
        x = self.leaky_relu1(self.conv1(x))
        x = self.leaky_relu2(self.conv2(x))
        x = self.leaky_relu3(self.conv3(x))
        x = self.leaky_relu4(self.conv4(x))
        x = self.conv5(x)
        return x

if __name__ == "__main__":
    hyper_analysis = Hyper_analysis(num_filters=192)
    input_image = torch.zeros([1,192,16,16])
    result = hyper_analysis(input_image)
    print(result.shape)