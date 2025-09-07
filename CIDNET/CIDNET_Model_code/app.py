import streamlit as st
import os
from datetime import datetime
from PIL import Image
import torch
from torchvision import transforms
import io
import shutil
import base64


import torch

device = "cpu"


import os
from PIL import Image


import torch
import torch.nn as nn

pi = 3.141592653589793

class RGB_HVI(nn.Module):
    def __init__(self):
        super(RGB_HVI, self).__init__()
        self.density_k = torch.nn.Parameter(torch.full([1],0.2)) #define trainable parameter k. Here k is reciprocal to the one mentioned in the paper
        self.this_k = 0 #to track the value of k

    def HVIT(self, img):
        eps = 1e-8 #define hyperparam epsilon as mentioned in paper
        device = img.device
        dtypes = img.dtype

        hue = torch.Tensor(img.shape[0], img.shape[2],img.shape[3]).to(dtypes) #initialize hue

        value = img.max(1)[0].to(dtypes) #value is simply intensity,i.e max value among R,G and B values
        img_min = img.min(1)[0].to(dtypes)

        #computation of hue:
        hue[img[:,2]==value] = 4.0 + ( (img[:,0]-img[:,1]) / (value - img_min + eps)) [img[:,2]==value]
        hue[img[:,1]==value] = 2.0 + ( (img[:,2]-img[:,0]) / (value - img_min + eps)) [img[:,1]==value]
        hue[img[:,0]==value] = (0.0 + ((img[:,1]-img[:,2]) / (value - img_min + eps)) [img[:,0]==value]) % 6

        hue[img.min(1)[0]==value] = 0.0
        hue = hue/6.0 # normalize

        #saturation
        saturation = (value - img_min) / (value + eps)
        saturation[value == 0] = 0

        #define gamma_g and gamma_b

        #define P_gamma ( piecewise function of gamma_g and gamma_b )

        #adding a channel dimension to each
        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)

        k = self.density_k
        self.this_k = k.item()

        #define Ck
        color_sensitive = ((value * 0.5* pi).sin() + eps).pow(k)

        # define h and v by orthogonalizing hue
        # replace hue by P_gamma in cx and cy formula
        cx = (2.0 * pi * hue).cos()
        cy = (2.0 * pi * hue).sin()

        #define T

        #define X,Y,Z axes - add Dt term
        X = color_sensitive * saturation * cx
        Y = color_sensitive * saturation * cy
        Z = value

        xyz = torch.cat([X,Y,Z], dim=1) # Combine the three components into a single tensor with 3 channels
        return xyz

    def PHVIT(self, img):
        eps = 1e-8
        H,V,I = img[:,0,:,:],img[:,1,:,:],img[:,2,:,:] #access each of H,V,I color spaces from 'channel' dimension

        # initialize gate options to False
        self.gated = False
        self.gated2 = False
        self.alpha = 1.0 #initialize hyperparam value to 1


        #clip
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        I = torch.clamp(I,0,1)

        v = I #I in HSV = V in HSV
        k = self.this_k
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)

        #normalize H and V values by Ck
        H = (H) / (color_sensitive + eps)
        V = (V) / (color_sensitive + eps)

        #clip
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)

        #compute hue
        h = torch.atan2(V,H) / (2*pi)
        h = h%1

        #compute saturation
        s = torch.sqrt(H**2 + V**2 + eps)

        #enable gating
        if self.gated:
            magnification = 1.3
            s = s * magnification # artificially increasing the saturation for more vibrant colors.

        #clip
        s = torch.clamp(s,0,1)
        v = torch.clamp(v,0,1)

        #convert HSV to RGB using a standard conversion formula:

        #set up r,g,b variables
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        # scale to an integer value between 0 and 5, representing the six sections of the HSV color wheel.
        hi = torch.floor(h * 6.0)

        # fractional part of h
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))

        hi0 = hi==0
        hi1 = hi==1
        hi2 = hi==2
        hi3 = hi==3
        hi4 = hi==4
        hi5 = hi==5

        # assign values of r, g, and b depending on which sector the hue falls into
        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        #expand r,g,b values along the channel dimension
        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)

        #concatenate to form a 3-channel RGB image
        rgb = torch.cat([r,g,b], dim=1)

        #enable gating
        if self.gated2:
            rgb = rgb * self.alpha

        return rgb


# !pip install einops

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# implementing that version of LayerNorm where the 'vanilla' LayerNorm has an extra weight-bias pair

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        # Permute to [batch_size, height, width, channels] for layer normalization
        x = x.permute(0, 2, 3, 1)
        # Apply layer normalization on the channels dimension (last dimension)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # Permute back to [batch_size, channels, height, width]
        x = x.permute(0, 3, 1, 2)
        return x

#define NormDownsample
class NormDownsample(nn.Module):
    def __init__ (self, in_ch, out_ch, scale=0.5, use_norm=False):
        super(NormDownsample, self).__init__()

        self.use_norm = use_norm

        if self.use_norm:
            self.norm = LayerNorm(out_ch)
        self.selu = nn.SELU()
        self.down = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        nn.UpsamplingBilinear2d(scale_factor=scale))

    def forward(self, x):
#         print(f"NormDownsample input shape: {x.shape}")  # Added
        x = self.down(x)
#         print(f"NormDownsample after downsample: {x.shape}")  # Added
        x = self.selu(x)
        if self.use_norm:
            x = self.norm(x)
#             print(f"NormDownsample after normalization: {x.shape}")  # Added
            return x
        else:
#             print(f"NormDownsample after normalization: {x.shape}")  # Added
            return x

#define NormUpsample
class NormUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2, use_norm=False):
        super(NormUpsample, self).__init__()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = LayerNorm(out_ch)
        self.selu = nn.SELU()
        self.up_scale = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale))
        self.up = nn.Conv2d(out_ch*2, out_ch, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, y):
#         print(f"NormUpsample input x shape: {x.shape}, y shape: {y.shape}")  # Added
        x = self.up_scale(x)
#         print(f"NormUpsample after up_scale: {x.shape}")  # Added
        x = torch.cat([x,y], dim=1)
#         print(f"NormUpsample after concatenation: {x.shape}")  # Added
        x = self.up(x)
        x = self.selu(x)
        if self.use_norm:
            return self.norm(x)
#             print(f"NormUpsample after normalization: {x.shape}")  # Added
        else:
#             print(f"NormUpsample after normalization: {x.shape}")  # Added
            return x


#cross attention block
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        self.num_heads = num_heads

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias) #normal conv
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias) #depth-wise conv

        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias) #ensure dimentional consistency

    def forward(self, x, y):
#         print(f"CAB input x shape: {x.shape}, y shape: {y.shape}")  # Added
        b,c,h,w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1) #split into individual k and v

#         print(f"CAB q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")  # Added
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        #normalize only q and k, and not v
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

#         print(f"CAB after rearrange q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")  # Added
        attn = (q @ k.transpose(-2,-1)) / torch.sqrt(torch.tensor(q.shape[-1], dtype=torch.float32)) #calculating attention scores
        attn = F.softmax(attn, dim=-1) #apply softmax activation function

        out = (attn @ v) #produce a weighted sum of the values for each query.
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

#         print(f"CAB output shape: {out.shape}")  # Added
        out = self.project_out(out)
        return out


class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor = 2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor) #common in feedforward layers of transformers to increase model capacity.

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=False)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias) #depth-wise convolution
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()

    def forward(self, x):
#         print(f"IEL input shape: {x.shape}")  # Added
        x = self.project_in(x)
#         print(f"IEL after project_in: {x.shape}")  # Added
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = x1 + self.Tanh(self.dwconv1(x1))
        x2 = x2 + self.Tanh(self.dwconv2(x2))
        x = x1 * x2
        x = self.project_out(x)
#         print(f"IEL output shape: {x.shape}")  # Added
        return x

class HV_LCA(nn.Module):
    def __init__(self, dim,num_heads, bias=False):
        super(HV_LCA, self).__init__()
        self.gdfn = IEL(dim) # IEL and CDL have same structure
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias)

    def forward(self, x, y):
#         print(f"HV_LCA input x shape: {x.shape}, y shape: {y.shape}")  # Added
        x = x + self.ffn(self.norm(x), self.norm(y))
#         print(f"HV_LCA after ffn: {x.shape}")  # Added
        x = self.gdfn(self.norm(x))
#         print(f"HV_LCA output shape: {x.shape}")  # Added
        return x

class I_LCA(nn.Module):
    def __init__(self, dim,num_heads, bias=False):
        super(I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)

    def forward(self, x, y):
#         print(f"I_LCA input x shape: {x.shape}, y shape: {y.shape}")  # Added
        x = x + self.ffn(self.norm(x), self.norm(y))
#         print(f"I_LCA after ffn: {x.shape}")  # Added
        x = x + self.gdfn(self.norm(x))
#         print(f"I_LCA output shape: {x.shape}")  # Added
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CIDNet(nn.Module):
    def __init__(self,
                channels = [36,36,72,144], #ch1, ch2, ch3, ch4
                heads = [1,2,4,8], #head1, head2, head3, head4
                norm = False):
        super(CIDNet, self).__init__()

        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads

        #HV WAYS

        #from hv map to hv feature
        self.HVE_block0 = nn.Sequential(
        nn.ReplicationPad2d(1),
        nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False)
        )

        #green coloured arrows in the architecture image
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm = norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm = norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm = norm)

        #final 3x3 conv
        self.HVD_block0 = nn.Sequential(
        nn.ReplicationPad2d(1),
        nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False)
        )


        #I WAYS

        #from map to feature
        self.IE_block0 = nn.Sequential(
        nn.ReplicationPad2d(1),
        nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False),
        )

        #red coloured arrows in the architecture image
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm = norm)

        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)

        #final 3x3 conv
        self.ID_block0 = nn.Sequential(
        nn.ReplicationPad2d(1),
        nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False),
        )

        #call LCA blocks
        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)

        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)

        self.trans = RGB_HVI()

    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x) #convert from rgb to hvi
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes) #access intensity dimention

        # Print shapes at every stage
#         print(f"Input shape: {x.shape}")
#         print(f"HVI shape: {hvi.shape}")
#         print(f"Intensity dimension shape: {i.shape}")

        #low

        #from channel to feature
        i_enc0 = self.IE_block0(i)
#         print(f"i_enc0 shape: {i_enc0.shape}")

        '''1st red arrow'''
        i_enc1 = self.IE_block1(i_enc0)
#         print(f"i_enc1 shape: {i_enc1.shape}")

        #from channel to feature
        hv_0 = self.HVE_block0(hvi)
#         print(f"hv_0 shape: {hv_0.shape}")

        #1st green arrow
        hv_1 = self.HVE_block1(hv_0)
#         print(f"hv_1 shape: {hv_1.shape}")

        #1st skip connections
        i_jump0 = i_enc0
        hv_jump0 = hv_0

        #1st LCA block
        i_enc2 = self.I_LCA1(i_enc1, hv_1)
#         print(f"i_enc2 shape (after I_LCA1): {i_enc2.shape}")

        hv_2 = self.HV_LCA1(hv_1, i_enc1)
#         print(f"hv_2 shape (after HV_LCA1): {hv_2.shape}")

        #2nd skip connections
        v_jump1 = i_enc2
        hv_jump1 = hv_2

        '''2nd red arrow'''
        i_enc2 = self.IE_block2(i_enc2)
#         print(f"i_enc2 shape (after IE_block2): {i_enc2.shape}")

        #2nd green arrow
        hv_2 = self.HVE_block2(hv_2)
#         print(f"hv_2 shape (after HVE_block2): {hv_2.shape}")

        #2nd LCA block
        i_enc3 = self.I_LCA2(i_enc2, hv_2)
#         print(f"i_enc3 shape (after I_LCA2): {i_enc3.shape}")

        hv_3 = self.HV_LCA2(hv_2, i_enc2)
#         print(f"hv_3 shape (after HV_LCA2): {hv_3.shape}")

        #3rd skip connections (mostly present between 3rd and 4th LCA Block)
        v_jump2 = i_enc3
        hv_jump2 = hv_3

        '''3rd red arrow'''
        i_enc3 = self.IE_block3(i_enc2)
#         print(f"i_enc3 shape (after IE_block3): {i_enc3.shape}")

        #3rd green arrow
        hv_3 = self.HVE_block3(hv_2)
#         print(f"hv_3 shape (after HVE_block3): {hv_3.shape}")

        #3rd LCA block
        i_enc4 = self.I_LCA3(i_enc3, hv_3)
#         print(f"i_enc4 shape (after I_LCA3): {i_enc4.shape}")

        hv_4 = self.HV_LCA3(hv_3, i_enc3)
#         print(f"hv_4 shape (after HV_LCA3): {hv_4.shape}")

        #4th LCA block
        i_dec4 = self.I_LCA4(i_enc4,hv_4)
#         print(f"i_dec4 shape (after I_LCA4): {i_dec4.shape}")

        hv_4 = self.HV_LCA4(hv_4, i_enc4)
#         print(f"hv_4 shape (after HV_LCA4): {hv_4.shape}")

        #4th green arrow
        hv_3 = self.HVD_block3(hv_4, hv_jump2)
#         print(f"hv_3 shape (after HVD_block3): {hv_3.shape}")

        '''4th red arrow'''
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
#         print(f"i_dec3 shape (after ID_block3): {i_dec3.shape}")


        #5th LCA block
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
#         print(f"i_dec2 shape (after I_LCA5): {i_dec2.shape}")

        hv_2 = self.HV_LCA5(hv_3, i_dec3)
#         print(f"hv_2 shape (after HV_LCA5): {hv_2.shape}")

        #5th green arrow
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
#         print(f"hv_2 shape (after HVD_block2): {hv_2.shape}")

        '''5th red arrow'''
        i_dec2 = self.ID_block2(i_dec3, v_jump1)
#         print(f"i_dec2 shape (after ID_block2): {i_dec2.shape}")


        #6th LCA block
        i_dec1 = self.I_LCA6(i_dec2, hv_2)
#         print(f"i_dec1 shape (after I_LCA6): {i_dec1.shape}")

        hv_1 = self.HV_LCA6(hv_2, i_dec2)
#         print(f"hv_1 shape (after HV_LCA6): {hv_1.shape}")

        '''6th red arrow'''
        i_dec1 = self.ID_block1(i_dec1, i_jump0)
#         print(f"i_dec1 shape (after ID_block1): {i_dec1.shape}")

        #final operation on I dim
        i_dec0 = self.ID_block0(i_dec1)
#         print(f"i_dec0 shape (after ID_block0): {i_dec0.shape}")

        #6th green arrow
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
#         print(f"hv_1 shape (after HVD_block1): {hv_1.shape}")

        #final operation on HV dim
        hv_0 = self.HVD_block0(hv_1)
#         print(f"hv_0 shape (after HVD_block0): {hv_0.shape}")

        #concat outputs from HV and I dimensions
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
#         print(f"output_hvi shape (after concatenation and addition): {output_hvi.shape}")

        #convert this output from HVI to RGB using PHVIT function
        output_rgb = self.trans.PHVIT(output_hvi)
#         print(f"output_rgb shape: {output_rgb.shape}")


        return output_rgb, output_hvi

def l1_loss(pred, target, weight, reduction = 'mean'):
    return torch.mean(torch.abs(pred-target)) if reduction == 'mean' else torch.abs(pred-target)

def sobel_filter(image):
    # Define Sobel kernels for edge detection
    sobel_x = torch.tensor([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    sobel_y = torch.tensor([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Apply Sobel filters
    edges_x = F.conv2d(image, sobel_x, padding=1)
    edges_y = F.conv2d(image, sobel_y, padding=1)

    # Calculate the magnitude of the edges
    edges = torch.sqrt(edges_x**2 + edges_y**2)

    return edges

def edge_loss(original_hr, reconstructed_lr):
    # Generate edge map for the original HR image
    edge_map = sobel_filter(original_hr)

    # Calculate the pixel loss (MAE)
    pixel_loss = F.l1_loss(reconstructed_lr, original_hr)

    # Calculate the edge loss
    loss_edges = (edge_map * torch.abs(original_hr - reconstructed_lr)).mean()

    # Combine losses
    alpha = 0.5  # You can adjust this value
    total_loss = alpha * pixel_loss + (1 - alpha) * loss_edges

    return total_loss

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class EdgeLoss(nn.Module):
    def __init__(self,loss_weight=1.0, reduction='mean'):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        self.mse_loss = torch.nn.MSELoss()
        self.weight = loss_weight

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.mse_loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss*self.weight

# define class 'VGGFeatureExtractor'

import torch
import torch.nn as nn
from torchvision import models

class VGGFeatureExtractor(nn.Module):
    """Simplified VGG network for feature extraction.

    Args:
        layer_name_list (list[str]): Layers from which to extract features.
        vgg_type (str): Type of VGG network. Default: 'vgg19'.
        use_input_norm (bool): If True, normalize input image. Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1]. Default: False.
    """

    def __init__(self, layer_name_list, vgg_type='vgg19', use_input_norm=True, range_norm=False):
        super(VGGFeatureExtractor, self).__init__()
        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        # Load the appropriate VGG model
        vgg_net = getattr(models, vgg_type)(pretrained=True).features

        # Store only the required layers
        self.layers = nn.ModuleDict()
        self.names = {f'layer_{i}': str(i) for i in range(len(vgg_net))}  # Generic layer names
        for i, layer in enumerate(vgg_net):
            layer_name = f'layer_{i}'
            if layer_name in layer_name_list:
                self.layers[layer_name] = layer

        if self.use_input_norm:
            # Register mean and std for input normalization
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        """Forward pass to extract features."""
        if self.range_norm:
            x = (x + 1) / 2  # Normalize from [-1, 1] to [0, 1]

        if self.use_input_norm:
            x = (x - self.mean) / self.std  # Normalize the input

        # Extract features from the specified layers
        output = {}
        for name, layer in self.layers.items():
            x = layer(x)
            output[name] = x.clone()  # Store the feature map from this layer

        return output

class PerceptualLoss(nn.Module):
    """Modified Perceptual loss without style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
        vgg_type (str): The type of vgg network used as feature extractor.
        use_input_norm (bool): If True, normalize the input image in vgg.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
        perceptual_weight (float): Weight for the perceptual loss.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=True,
                 perceptual_weight=1.0,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2Loss()
        elif self.criterion_type == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='mean')
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        return percep_loss

# Hyperparameters for weighted loss
lambda_1 = 1 # L1 loss
lambda_e = 1 # Edge loss
lambda_p = 1 # Perceptual loss

lambda_c = 1 # to balance the loss in different color spaces

# FINAL LOSS
class FinalLoss(nn.Module):
    def __init__(self, lambda_1=1.0, lambda_e=1.0, lambda_p=1.0, lambda_c=1.0):
        super(FinalLoss, self).__init__()
        self.lambda1 = lambda_1
        self.lambdae = lambda_e
        self.lambdap = lambda_p
        self.lambda_c = lambda_c  # Added for the new term
        self.l1_loss = L1Loss()
        self.edge_loss = EdgeLoss()
        self.perceptual_loss = PerceptualLoss(layer_weights={
            'conv1_1': 1.0,
            'conv2_1': 1.0,
            'conv3_1': 1.0,
            'conv4_1': 1.0,
            'conv5_1': 1.0
        })

#     def forward(self, X_hat_HV, X_HV, X_hat, X, Y):
#         # Calculate losses
#         l1 = self.l1_loss(X_hat_HV, X_HV)
#         edge = self.edge_loss(X_hat_HV, X_HV)
#         perceptual = self.perceptual_loss(X_hat, Y)

#         # Calculate total loss
#         total_loss = (self.lambda_c * l1) + (self.lambdae * edge) + (self.lambdap * perceptual)
#         return total_loss

    def forward(self, X_hat_HV, X_HV, X_hat, X):

        # Calculate losses for HVI color space
        l1_HVI = self.l1_loss(X_hat_HV, X_HV)
        edge_HVI = self.edge_loss(X_hat_HV, X_HV)
        perceptual_HVI = self.perceptual_loss(X_hat, X_HV)

        # Calculate losses for RGB color space
        l1_RGB = self.l1_loss(X_hat, X)
        edge_RGB = self.edge_loss(X_hat, X)
        perceptual_RGB = self.perceptual_loss(X_hat, X)

#         print(l1_HVI, edge_HVI, perceptual_HVI)
#         print(l1_RGB, edge_RGB, perceptual_RGB)

        # Calculate total loss
        total_loss = (self.lambda_c*(l1_HVI + edge_HVI + perceptual_HVI)) + (l1_RGB + edge_RGB + perceptual_RGB)
        return total_loss

from torch import optim
model = CIDNet()
def process(low_images):
    
    low_images_rgb = low_images
    model.eval()
    model.load_state_dict(torch.load("cidnet_model_final.pth", map_location=torch.device('cpu')))

#       print(type(high_images_rgb))

    #converting input ground truth images from RGB to HVI space
    rgb_hvi_transform = RGB_HVI()  # Instantiate the transformer
    # Make sure the transformation is done on the GPU
    # rify that high_images_hvi is on the correct device
    #set parameter gradient values to 0
    outputs_rgb, outputs_hvi = model(low_images_rgb)
    
    return outputs_rgb

# print("Model  successfully!")
# x=torch.rand(1, 3, 256, 256)
# y=process(x)
# print(y.shape)


def set_light_theme():
    """Set light theme and custom styles"""
    st.markdown("""
        <style>
        /* Main app styles */
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
            background-color: #ffffff;
        }
        
        /* Upload box styles */
        .upload-box {
            border: 2px dashed #e0e0e0;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            background-color: #fafafa;
            color: #666666;
        }
        
        /* File path styles */
        .file-path {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            margin: 10px 0;
            border: 1px solid #e0e0e0;
            color: #444444;
        }
        
        /* Logo container styles */
        .logo-container {
            position: fixed;
            top: 16px;
            right: 16px;
            z-index: 1000;
            background-color: #ffffff;
            padding: 5px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Logo image styles */
        .logo-img {
            max-height: 50px;
            width: auto;
        }
        
        /* Main title styles */
        .main-title {
            padding-right: 70px;
            color: #2c3e50;
            margin-bottom: 1rem;
        }
        
        /* Subtitle styles */
        .subtitle {
            color: #34495e;
            font-size: 1.2rem;
        }
        
        /* Card styles */
        .stCard {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        /* Button styles */
        .stButton > button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 0.5rem 1rem;
            transition: background-color 0.3s;
        }
        
        .stButton > button:hover {
            background-color: #2980b9;
        }
        
        /* Spinner styles */
        .stSpinner > div {
            border-color: #3498db !important;
        }
        
        /* Success message styles */
        .success-message {
            background-color: #d4edda;
            color: #155724;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        
        /* Error message styles */
        .error-message {
            background-color: #f8d7da;
            color: #721c24;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        
        /* Code block styles */
        .code-block {
            background-color: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            padding: 10px;
            font-family: monospace;
            color: #444444;
        }
        
        /* Directory contents styles */
        .directory-contents {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            padding: 15px;
            margin-top: 10px;
        }
        
        /* Image caption styles */
        .image-caption {
            color: #666666;
            font-size: 0.9rem;
            text-align: center;
            margin-top: 5px;
        }
        </style>
    """, unsafe_allow_html=True)
def add_logo():
    """Add logo to the top right corner"""
    st.markdown(
        """
        <style>
        .logo-container {
            position: relative;
            padding: 5px;
            
        }
        .logo-img {
            max-height: 80px;
            width: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Replace 'logo.png' with your logo file
    # Make sure to place your logo file in the same directory as the script
    if os.path.exists('135524263.jpg'):
        with open('135524263.jpg', 'rb') as f:
            logo_bytes = f.read()
            logo_b64 = base64.b64encode(logo_bytes).decode()
            st.markdown(
                f"""
                <div class="logo-container">
                    <img src="data:image/png;base64,{logo_b64}" class="logo-img">
                </div>
                """,
                unsafe_allow_html=True
            )

def create_directories():
    """Create necessary directories if they don't exist"""
    dirs = ['uploads', 'processed']
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to local storage and return the file path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.name}"
    filepath = os.path.join('uploads', filename)
    
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    return filepath

def process_image(input_path):
    """
    Process the image using your model
    Replace this with your actual model processing logic
    """
    try:
        # Load image
        image = Image.open(input_path)
        
        # Example preprocessing
        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        # REPLACE THIS SECTION WITH YOUR MODEL
        # This is just a placeholder example that inverts the image
        output = process(input_batch)
        print(type(output))
        # Convert output tensor to image
        output_image = transforms.ToPILImage()(output.squeeze(0))
        
        # Save processed image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"processed_{timestamp}.png"
        output_path = os.path.join('processed', output_filename)
        output_image.save(output_path)
        
        return output_path, None
        
    except Exception as e:
        return None, str(e)

def main():
    # Set page config
    st.set_page_config(
        page_title="Local Image Processor",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    

    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .upload-box {
            border: 2px dashed #cccccc;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .file-path {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            margin: 10px 0;
        }
        /* Add padding to title to avoid logo overlap */
        .main-title {
            padding-right: 70px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title with padding to avoid logo overlap
    # Add logo
    add_logo()
    st.markdown('<h1 class="main-title">Local Image Processor</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Rest of your code remains the same...
    # [Previous code for sidebar, main content, etc.]
    
    # Sidebar information
    with st.sidebar:
        st.title("System Info")
        st.markdown("### Storage Locations")
        st.code(f"Upload Directory: {os.path.abspath('uploads')}")
        st.code(f"Processed Directory: {os.path.abspath('processed')}")
        
        if st.button("Clear All Storage"):
            try:
                shutil.rmtree('uploads')
                shutil.rmtree('processed')
                create_directories()
                st.success("Storage cleared successfully!")
            except Exception as e:
                st.error(f"Error clearing storage: {str(e)}")

    # Main content
    col1, col2 = st.columns(2)

    # Input Column
    with col1:
        st.subheader("Input Image")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image file (PNG, JPG, JPEG)"
        )

        if uploaded_file is not None:
            # Save and display input image
            input_path = save_uploaded_file(uploaded_file)
            st.image(input_path, use_column_width=True, caption="Input Image")
            
            # Show file path
            st.markdown("##### Saved Input Path:")
            st.markdown(f"<div class='file-path'>{input_path}</div>", unsafe_allow_html=True)

            # Process button
            if st.button("Process Image", type="primary"):
                with st.spinner("Processing image..."):
                    output_path, error = process_image(input_path)
                    
                    if error:
                        st.error(f"Error processing image: {error}")
                    else:
                        # Display output image in second column
                        with col2:
                            st.subheader("Output Image")
                            st.image(output_path, use_column_width=True, caption="Processed Image")
                            
                            # Show file path
                            st.markdown("##### Saved Output Path:")
                            st.markdown(f"<div class='file-path'>{output_path}</div>", 
                                      unsafe_allow_html=True)
                            
                            # Download button
                            with open(output_path, "rb") as file:
                                st.download_button(
                                    label="Download Processed Image",
                                    data=file,
                                    file_name=os.path.basename(output_path),
                                    mime="image/png"
                                )
        else:
            st.markdown("""
                <div class="upload-box">
                    <p>👆 Upload an image to get started</p>
                </div>
                """, unsafe_allow_html=True)

    # Output Column (initial state)
    with col2:
        if uploaded_file is None:
            st.subheader("Output Image")
            st.markdown("""
                <div class="upload-box">
                    <p>Processed image will appear here</p>
                </div>
                """, unsafe_allow_html=True)

    # Display storage statistics
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Upload Directory Contents")
        uploads = os.listdir('uploads')
        if uploads:
            for file in uploads:
                st.code(file)
        else:
            st.info("No files in uploads directory")

    with col2:
        st.markdown("### Processed Directory Contents")
        processed = os.listdir('processed')
        if processed:
            for file in processed:
                st.code(file)
        else:
            st.info("No files in processed directory")

if __name__ == "__main__":
    main()