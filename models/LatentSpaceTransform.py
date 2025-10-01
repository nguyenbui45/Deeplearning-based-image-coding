from .GDN import GDN
import torch
from torch import nn
    

class Latent_Space_Transform(nn.Module):
    def __init__(self,num_filters=128,out_channel=256):
        super(Latent_Space_Transform,self).__init__()
        # scale factor 2,1,1,1
        # RB0 - block 0
        self.b0_layer0 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b0_layer0_relu = nn.LeakyReLU()
        self.b0_layer1 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b0_layer1_relu = nn.LeakyReLU()
        
        # RB w upsample 0 - block 1, rk = 2   
        self.b1_up_shortcut = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=2,padding=1, output_padding=1)
        self.b1_up_layer0 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=2, padding=1, output_padding=1)
        self.b1_up_layer0_relu = nn.LeakyReLU()
        self.b1_up_layer1 = nn.Conv2d(num_filters, num_filters, 3,stride=1,padding=1)
        self.b1_up_layer0_igdn =  GDN(num_filters, inverse=True)
        
        # RB1 - block  2
        self.b2_layer0 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b2_layer0_relu = nn.LeakyReLU()
        self.b2_layer1 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b2_layer1_relu = nn.LeakyReLU()
        
        # RB w upsample 1 - block 3, rk = 1
        self.b3_up_shortcut = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1,padding=1)
        self.b3_up_layer0 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b3_up_layer0_relu = nn.LeakyReLU()
        self.b3_up_layer1 = nn.Conv2d(num_filters, num_filters, 3,stride=1,padding=1)
        self.b3_up_layer0_igdn =  GDN(num_filters, inverse=True)
        
        # RB2 - block 4
        self.b4_layer0 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b4_layer0_relu = nn.LeakyReLU()
        self.b4_layer1 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b4_layer1_relu = nn.LeakyReLU()
        
        # RB w upsample 2 - block 5, rk = 1
        self.b5_up_shortcut = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1,padding=1)
        self.b5_up_layer0 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b5_up_layer0_relu = nn.LeakyReLU()
        self.b5_up_layer1 = nn.Conv2d(num_filters, num_filters, 3,stride=1,padding=1)
        self.b5_up_layer0_igdn =  GDN(num_filters, inverse=True)
        
        # RB3 - block 6
        self.b6_layer0 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b6_layer0_relu = nn.LeakyReLU()
        self.b6_layer1 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.b6_layer1_relu = nn.LeakyReLU() 
        
        # Conv3 - block 7
        self.b7_layer0 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1)
        
        # activation - block 8
        self.b8_layer0_activation = nn.Conv2d(num_filters,out_channel,1)
        
        
    def forward(self, x):
        # block 0
        b0 = self.b0_layer0_relu(self.b0_layer0(x))
        b0 = self.b0_layer1_relu(self.b0_layer1(b0))
        b0 += x
        
        # block 1
        shortcut0 = self.b1_up_shortcut(b0)
        b1 = self.b1_up_layer0_relu(self.b1_up_layer0(b0))
        b1 = self.b1_up_layer0_igdn(self.b1_up_layer1(b1))
        b1 += shortcut0
        
        # block 2
        b2 = self.b2_layer0_relu(self.b2_layer0(b1))
        b2 = self.b2_layer1_relu(self.b2_layer1(b2))
        b2 += b1        


        # block 3
        shortcut1 = self.b3_up_shortcut(b2)
        b3 = self.b3_up_layer0_relu(self.b3_up_layer0(b2))
        b3 = self.b3_up_layer0_igdn(self.b3_up_layer1(b3))
        b3 += shortcut1
        
        
        # block 4
        b4 = self.b4_layer0_relu(self.b4_layer0(b3))
        b4 = self.b4_layer1_relu(self.b4_layer1(b4))
        b4 += b3 
        
        # block 5
        shortcut2 = self.b5_up_shortcut(b4)
        b5 = self.b5_up_layer0_relu(self.b5_up_layer0(b4))
        b5 = self.b5_up_layer0_igdn(self.b5_up_layer1(b5))
        b5 += shortcut2
        
        # block 6
        b6 = self.b6_layer0_relu(self.b6_layer0(b5))
        b6 = self.b6_layer1_relu(self.b6_layer1(b6))
        b6 += b5
        
        # block 7
        b7 = self.b7_layer0(b6)
        
        # block 8
        b8 = self.b8_layer0_activation(b7)
        
        return b8
    
    
if __name__ == "__main__":
    lst= Latent_Space_Transform(out_channel=256)
    feature = torch.zeros([1,128,16,16])
    output_image = lst(feature)
    print(output_image.shape)
    
    
    
       
        