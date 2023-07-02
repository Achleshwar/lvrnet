import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
    
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    """
    Please refer Figure 3 in the paper. 
    """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
    
class LAN(nn.Module):
    """
    This is implementation of Level Attention Module. 
    Please refer Figure 7 in the paper. 
    """
    def __init__(self, input_shape):
        super().__init__()
        self.N, self.D, self.C, self.H, self.W = input_shape
        self.conv = nn.Conv2d(self.D*self.C, self.C, kernel_size=1)
        self.softmax = nn.Softmax2d()
    def forward(self, x_og):
        x = x_og.view(self.N, self.D, -1)
        x_T = x.permute(0, 2, 1)
        l_corr = self.softmax(torch.matmul(x, x_T).unsqueeze(0))
        o1 = torch.matmul(l_corr.squeeze(0), x).view(self.N, self.D, self.C, self.H, self.W) + x_og
        o2 = o1.view(self.N, self.D * self.C, self.H, self.W)
        out = self.conv(o2)
        return out
    
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class Group(nn.Module):
    def __init__(self,conv=default_conv, dim=64, kernel_size=3, blocks=2): ##blocks should be 19 here but less RAM -> change later
        super(Group, self).__init__()
        modules = [NAFBlock(dim)  for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)
        
    def forward(self, x):
        res = self.gp(x)
        res += x
        return res
    
class LVRNet(nn.Module):
    """"
    This is complete implementation of our entire network. 
    Please refer Figure 2 in the paper. 
    """
    def __init__(self,gps,blocks,conv=default_conv, bs=1):
        super(LVRNet, self).__init__()
        self.gps=gps
        self.dim=32
        kernel_size=3
        self.batch_size = bs 
        self.size = (256,456) #TODO remove hard coded size
        pre_process = [conv(3, self.dim, kernel_size)]
#         assert self.gps==3
        self.g1= Group(conv, self.dim, kernel_size,blocks=blocks)
        self.g2= Group(conv, self.dim, kernel_size,blocks=blocks)
        self.g3= Group(conv, self.dim, kernel_size,blocks=blocks)
#         self.selective_kernel = nn.ModuleList([SKFF(128**i, 3) for i in range(3)])
        self.lan = LAN((self.batch_size, self.gps, self.dim, self.size[0], self.size[1]))
        post_precess = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]
        # self.lan = 
        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_precess)

    def forward(self, x1):
        x = self.pre(x1)
        res1=self.g1(x)
        res2=self.g2(res1)
        res3=self.g3(res2)
        out = torch.stack([res1, res2, res3], dim=1)
        out_shape = out.shape
        lan_out = self.lan(out)
        # out = self.selective_kernel[0]([res1, res2, res3])
        x=self.post(lan_out)
        return x + x1
