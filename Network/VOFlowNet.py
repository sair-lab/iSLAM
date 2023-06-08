import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def conv(in_planes, out_planes, kernel_size=3, stride=2, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
        nn.ReLU(inplace=True)
    )

def linear(in_planes, out_planes):
    return nn.Sequential(
        nn.Linear(in_planes, out_planes), 
        # nn.Dropout(p=0.5),
        nn.ReLU(inplace=True)
    )

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = conv(inplanes, planes, 3, stride, pad, dilation)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        out += x

        return F.relu(out, inplace=True)

class VOFlowRes(nn.Module):
    def __init__(self, intrinsic=True, down_scale=True, config=1, stereo=0, fix_parts=(),
                    extrinsic_encoder_layers=2, trans_head_layers=3):
        super(VOFlowRes, self).__init__()

        self.intrinsic = intrinsic
        self.down_scale = down_scale
        self.config = config
        self.stereo = stereo
        self.fix_parts = fix_parts
        self.extrinsic_encoder_layers = extrinsic_encoder_layers
        self.trans_head_layers = trans_head_layers

        self.feat_net, feat_dim = self.__feature_embedding()
        if stereo==2.2:
            self.feat_net2, _ = self.__feature_embedding()

        if stereo==2.1 or stereo==2.2:
            if extrinsic_encoder_layers >= 1:
                layers = [linear(6, 128)]
                for i in range(extrinsic_encoder_layers-1):
                    layers.append(linear(128, 128))
                self.extrinsic_encoder = nn.Sequential(*layers)
                extrinsic_encoder_dim = 128
            else:   # use sin/cos encoder
                self.extrinsic_encoder = self.__encode_pose
                extrinsic_encoder_dim = 120

            feat_dim_trans = feat_dim*2 + extrinsic_encoder_dim
            # fc1_trans = linear(feat_dim_trans, 256)
            # fc2_trans = linear(256, 32)
            # fc3_trans = nn.Linear(32, 3)
            self.fcAB_trans = linear(feat_dim, 128)
            self.fcAC_trans = linear(feat_dim, 128)
            
            layers = []
            layers.append(linear(128*2 + extrinsic_encoder_dim, 128))
            for i in range(trans_head_layers-3):
                layers.append(linear(128, 128))
            layers.append(linear(128, 32))
            layers.append(nn.Linear(32, 3))
            self.voflow_trans = nn.Sequential(*layers)
       
        else:
            fc1_trans = linear(feat_dim, 128)
            fc2_trans = linear(128, 32)
            fc3_trans = nn.Linear(32, 3)
            self.voflow_trans = nn.Sequential(fc1_trans, fc2_trans, fc3_trans)

        fc1_rot = linear(feat_dim, 128)
        fc2_rot = linear(128, 32)
        fc3_rot = nn.Linear(32, 3)
        self.voflow_rot = nn.Sequential(fc1_rot, fc2_rot, fc3_rot)

        if "feat" in fix_parts:
            self.fix_param(self.feat_net)
        if "feat2" in fix_parts and stereo==2.2:
            self.fix_param(self.feat_net2)
        if "rot" in fix_parts:
            self.fix_param(self.voflow_rot)
        if "trans" in fix_parts:
            self.fix_param(self.voflow_trans)


    def fix_param(self, model):
        for param in model.parameters():
            param.requires_grad = False


    def __feature_embedding(self):
        if self.intrinsic:
            inputnum = 4
        else:
            inputnum = 2
        if self.stereo==1:
            inputnum += 1

        if self.config==0:
            blocknums = [2,2,3,3,3,3,3]
            outputnums = [32,64,64,64,128,128,128]
        elif self.config==1:
            blocknums = [2,2,3,4,6,7,3]
            outputnums = [32,64,64,128,128,256,256]
        elif self.config==2:
            blocknums = [2,2,3,4,6,7,3]
            outputnums = [32,64,64,128,128,256,256]
        elif self.config==3:
            blocknums = [3,4,7,9,9,5,3]
            outputnums = [32,64,128,128,256,256,512]

        layers = []
        layers.append(conv(inputnum, 32, 3, 2, 1, 1))
        layers.append(conv(      32, 32, 3, 1, 1, 1))
        layers.append(conv(      32, 32, 3, 1, 1, 1))

        self.inplanes = 32
        if not self.down_scale:
            layers.append(self.__make_layer(BasicBlock, outputnums[0], blocknums[0], 2, 1, 1)) # (160 x 112)
            layers.append(self.__make_layer(BasicBlock, outputnums[1], blocknums[1], 2, 1, 1)) # (80 x 56)

        layers.append(self.__make_layer(BasicBlock, outputnums[2], blocknums[2], 2, 1, 1)) # 28 x 40
        layers.append(self.__make_layer(BasicBlock, outputnums[3], blocknums[3], 2, 1, 1)) # 14 x 20
        layers.append(self.__make_layer(BasicBlock, outputnums[4], blocknums[4], 2, 1, 1)) # 7 x 10
        layers.append(self.__make_layer(BasicBlock, outputnums[5], blocknums[5], 2, 1, 1)) # 4 x 5
        layers.append(self.__make_layer(BasicBlock, outputnums[6], blocknums[6], 2, 1, 1)) # 2 x 3

        if self.config==2:
            layers.append(conv(outputnums[6], outputnums[6]*2, kernel_size=(2, 3), stride=1, padding=0)) # 1 x 1

        if self.config==2:
            embedding_dim = outputnums[6]*2
        elif self.config==3:
            embedding_dim = outputnums[6]
        else:
            embedding_dim = outputnums[6]*6

        return nn.Sequential(*layers), embedding_dim

    def __make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def __encode_pose(self, x, L=10):
        c = (torch.pow(2, torch.arange(L)) * torch.pi).to(x.device)
        y = c.view(1, -1, 1) * x.unsqueeze(1)
        z = torch.cat([torch.sin(y), torch.cos(y)], dim=1).view(x.shape[0], -1)
        return z

    def forward(self, x, extrinsic=None):
        if self.stereo==2.1 or self.stereo==2.2:
            return self.forward_multicam(x, extrinsic)
        else:
            return self.forward_(x)

    def forward_(self, x, scale_disp=1.0):
        x = self.feat_net(x)
        if self.config==3:
            x = F.avg_pool2d(x, kernel_size = x.shape[-2:])
        
        x = x.view(x.shape[0], -1)
        x_trans = self.voflow_trans(x)
        x_rot = self.voflow_rot(x)

        return torch.cat((x_trans, x_rot), dim=1)

    def forward_multicam(self, x, extrinsic):
        x_AB = x[:, (0,1, 4,5), ...]
        x_AC = x[:, (2,3, 4,5), ...]

        if self.stereo==2.2:
            x_AB = self.feat_net2(x_AB)
        else:
            x_AB = self.feat_net(x_AB)
        x_AC = self.feat_net(x_AC)

        x_AB = x_AB.view(x_AB.shape[0], -1)
        x_AC = x_AC.view(x_AC.shape[0], -1)

        x_ex = self.extrinsic_encoder(extrinsic)
        x_AB_128 = self.fcAB_trans(x_AB)
        x_AC_128 = self.fcAC_trans(x_AC)
        x_trans = torch.cat((x_AC_128, x_AB_128, x_ex), dim=1)
        x_trans = self.voflow_trans(x_trans)
        # assert torch.any(x_trans[0] != x_trans[1]) or torch.any(x_trans[1] != x_trans[2])

        x_rot = self.voflow_rot(x_AC)

        return torch.cat((x_trans, x_rot), dim=1)