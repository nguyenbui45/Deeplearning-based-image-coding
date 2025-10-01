'''
Adapt from implementation of 
https://github.com/thekoshkina/learned_image_compression/blob/master/ratedistortionloss.py

'''

import torch
from torch import nn
import numpy as np
from PIL import Image
from torch.distributions import Normal
from torchvision import transforms
import math
    

class RateDistortionLoss(nn.Module):
    '''
    Rate Distortion Loss
    '''
    def __init__(self, type="sigmoid", constant_lambda=True,image_size=256):
        
        super(RateDistortionLoss, self).__init__()
        if type == "normal":
            self.hyper_cumulative = self.simple_cumulative
        elif type == "sigmoid":
            self.hyper_cumulative = self.sigmoid_cumulative
		
        if constant_lambda:
            self.assign_lambda = self.constant_lambda
        else:
            self.assign_lambda = self.lambda_update
            self.epsilon = 1e-2
            
        self.image_size = image_size

        
    def cumulative(self, mu, sigma, x):
        """
        Calculates CDF of Normal distribution with parameters mu and sigma at point x
        """
        half = 0.5
        const = (2 ** 0.5)
        return half * (1 + torch.erf((x - mu) / (const * sigma)))

    def simple_cumulative(self, x):
        """
        Calculates CDF of Normal distribution with mu = 0 and sigma = 1
        """
        half = 0.5
        const = -(2 ** -0.5)
        return half * torch.erf(const * x)

    def sigmoid_cumulative(self, x):
        """
        Calculates sigmoid of the tensor to use as a replacement of CDF
        """
        return torch.sigmoid(x)

    def lambda_update(self, lam, distortion):
        """
        Updates Lagrangian multiplier lambda at each step
        """
        return self.epsilon * distortion + lam

    def constant_lambda(self, lam, distortion):
        """
        Assigns Lambda the same in the case lambda is constant
        """
        return 0.025

    def latent_rate(self, mu, sigma, y):
        """
        Calculate latent rate
        
        Since we assume that each latent is modelled a Gaussian distribution convolved with Unit Uniform distribution we calculate latent rate
        as a difference of the CDF of Gaussian at two different points shifted by -1/2 and 1/2 (limit points of Uniform distribution)
        
        See apeendix 6.2
        J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston,
        “Variational image compression with a scale hyperprior,” 6th Int. Conf. on Learning Representations, 2018. [Online].
        Available: https://openreview.net/forum?id=rkcQFMZRb.
        """
        # upper = self.cumulative(mu, sigma, (y + .5)).dtype(torch.float32)
        # lower = self.cumulative(mu, sigma, (y - .5)).dtype(torch.float32)
        # print(torch.isnan(upper).any())
        # print(torch.isnan(lower).any())
        eps=1e-12
        with torch.amp.autocast(enabled=False,device_type='cuda'):
            u = self.cumulative(mu.float(), sigma.float(), (y.float() + 0.5)).to(torch.float32)
            l = self.cumulative(mu.float(), sigma.float(), (y.float() - 0.5)).to(torch.float32)

            # keep CDF in (0,1) and enforce ordering (don’t use abs)
            u = u.clamp(1e-12, 1 - 1e-12)
            l = l.clamp(1e-12, 1 - 1e-12)
            u, l = torch.maximum(u, l), torch.minimum(u, l)

            pmf = (u - l).clamp_min(eps)          # avoid 0/negative mass
            bits = -(pmf.log() / math.log(2)).sum()  # safe log2
        
        # return -torch.sum(torch.log2(torch.abs(upper - lower)))
        return bits

    def hyperlatent_rate(self, z):
        """
        Calculate hyperlatent rate

        Since we assume that each latent is modelled a Non-parametric convolved with Unit Uniform distribution we calculate latent rate
        as a difference of the CDF of the distribution at two different points shifted by -1/2 and 1/2 (limit points of Uniform distribution)

        See apeendix 6.2
        J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston,
        “Variational image compression with a scale hyperprior,” 6th Int. Conf. on Learning Representations, 2018. [Online].
        Available: https://openreview.net/forum?id=rkcQFMZRb.
        """
        upper = self.hyper_cumulative(z + .5)
        lower = self.hyper_cumulative(z - .5)
        return -torch.sum(torch.log2(torch.abs(upper - lower)))

    def forward(self, x, x_hat,F,F_tilde,gamma, mu, sigma, y_hat, z, lam):
        """
        Calculate Rate-Distortion Loss
        """
        distortion_x = torch.mean((x - x_hat).pow(2)) 
        distortion_F = torch.mean(torch.stack([torch.mean((F[j] - F_tilde[j]).pow(2))*gamma for j in range(len(F))]))
        D = distortion_x + distortion_F
        
        
        latent_rate = torch.mean(self.latent_rate(mu[0], sigma[0], y_hat[0])) + torch.mean(self.latent_rate(mu[1], sigma[1], y_hat[1]))
        hyperlatent_rate = torch.mean(self.hyperlatent_rate(z))
        # print("y1:")
        # print(torch.isnan(mu[0]).any())
        # print(torch.isnan(sigma[0]).any())
        # print(torch.isnan(y_hat[0]).any())
        
        # print("y2:")
        # print(torch.isnan(mu[1]).any())
        # print(torch.isnan(sigma[1]).any())
        # print(torch.isnan(y_hat[1]).any())
        
        R = (latent_rate + hyperlatent_rate) / (self.image_size * self.image_size * x.size(0))
        # lam = self.assign_lambda(lam)
        loss = lam * D + R
        
        # print(D)
        # print(latent_rate)
        # print(hyperlatent_rate)
        
        return loss, R, D
    
    
def latent_rate(feature, mu, sigma):

    gaussian = Normal(mu, sigma)
    pmf = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
    total_bits = torch.sum(torch.clamp(-1.0 * torch.log(pmf + 1e-10) / math.log(2.0), 0, 50))
    
    
    return total_bits

    
    
if __name__ == "__main__":
    original_image = '/home/nguyensolbadguy/Code_Directory/compression/models/yolov3/barbara.bmp'
    compared_image = '/home/nguyensolbadguy/Code_Directory/compression/models/yolov3/compressed_barbara.jpg'
    
    # print(msssim(original_image, compared_image),end=' ')
    # print(psnr(original_image, compared_image),end=' ')

    
    original = np.array(Image.open(original_image).convert('RGB'), dtype=np.float32)
    compared = np.array(Image.open(compared_image).convert('RGB'), dtype=np.float32)
    
    transform =transforms.Compose([transforms.ToTensor()])
    original_tensor = transform(original).unsqueeze(0)
    compared_tensor = transform(compared).unsqueeze(0)
    
    F = torch.ones([1,256,64,64])
    F_tilde = torch.ones([1,256,64,64])*0.1
    
    
    means = torch.zeros([16,128,16,16])
    variances = torch.ones([16,128,16,16])
    feature = torch.randn([16,128,16,16])
    
    z = torch.rand([1,192,4,4])
    
    gamma = [0.006]
    lam = 0.013
    
    RD_loss = RateDistortionLoss('sigmoid')
    #loss = RD_loss(original_tensor,compared_tensor,F,F_tilde,gamma,means,variances,feature,z,lam)
    #print(loss)
    
    bitrate = RD_loss.latent_rate(means,variances,feature)
    print(bitrate)
    
    bitrate2 = latent_rate(feature,means,variances)
    print(bitrate2)
    
    



