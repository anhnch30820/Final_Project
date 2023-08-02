import torch
import torch.nn as nn

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			  nn.ReLU(inplace=True)
        )


        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.up(x)
        return x


class ATT(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(ATT,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.model = torch.hub.load('PingoLH/Pytorch-HarDNet', 'hardnet68', pretrained=True)
        children = list(self.model.children())
        self.conv1 = nn.Sequential(*children[0][0:2])
        self.conv2 = nn.Sequential(*children[0][2:5])
        self.conv3 = nn.Sequential(*children[0][5:10])
        self.conv4 = nn.Sequential(*children[0][10:13])
        self.conv5 = nn.Sequential(*children[0][13:16])


        self.Up5 = up_conv(ch_in=1024,ch_out=640)
        self.Att5 = ATT(F_g=640,F_l=640,F_int=320)
        self.Up_conv5 = conv_block(ch_in=1280, ch_out=640)

        self.Up4 = up_conv(ch_in=640,ch_out=320)
        self.Att4 = ATT(F_g=320,F_l=320,F_int=128)
        self.Up_conv4 = conv_block(ch_in=640, ch_out=320)

        self.Up3 = up_conv(ch_in=320,ch_out=128)
        self.Att3 = ATT(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = ATT(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Up1 = up_conv(ch_in=64,ch_out=32)
        self.Att1 = ATT(F_g=32,F_l=3,F_int=3)
        self.Up_conv1 = conv_block(ch_in=35, ch_out=16)

        self.Conv_1x1 = nn.Conv2d(16,3,kernel_size=1,stride=1,padding=0)


    def forward(self, im_data):
        x1 = self.conv1(im_data)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)


        d5 = self.Up5(x5)
        x4 = self.Att5(d5, x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(d4,x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(d3,x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(d2,x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        x = self.Att1(d1,im_data)
        d1 = torch.cat((x,d1),dim=1)
        d1 = self.Up_conv1(d1)

        # d1 = self.Up1(d2)
        # d = torch.cat((im_data, d1),dim=1)
        # d = self.Up_conv1(d)
        # d = d+d1
        # d = self.Up_conv(d)

        ouput = (self.Conv_1x1(d1))

        return ouput
    

from copy import deepcopy

class ModelEmaV2(nn.Module):
    def __init__(self, model, decay=0.5, device="cuda"):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model, epoch):  
        if epoch >=10:
           self.decay = 0.995

        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)