
import torch 
import torch.nn as nn
import torch.nn.functional as F

class VONet(nn.Module):
    def __init__(self, fix_parts=('flow', 'stereo')):
        super(VONet, self).__init__()

        from .PWC import PWCDCNet as FlowNet
        self.flowNet = FlowNet(uncertainty=False)

        from .StereoNet7 import StereoNet7 as StereoNet
        self.stereoNet = StereoNet()

        from .VOFlowNet import VOFlowRes as FlowPoseNet
        self.flowPoseNet = FlowPoseNet(intrinsic=True, down_scale=True, stereo=0, fix_parts=fix_parts)
        # from .orig_VOFlowNet import VOFlowRes as FlowPoseNet
        # self.flowPoseNet = FlowPoseNet(intrinsic=True, down_scale=True, config=1, stereo=0)
        
        if "flow" in fix_parts:
            for param in self.flowNet.parameters():
                param.requires_grad = False

        if "stereo" in fix_parts:
            for param in self.stereoNet.parameters():
                param.requires_grad = False

    def forward(self, img0, img1, img0_norm, img0_r_norm, intrinsic):
        # import ipdb;ipdb.set_trace()
        flow, _ = self.flowNet(torch.cat([img0, img1], dim=1))
        flow = flow[0]

        disp, _ = self.stereoNet(torch.cat((img0_norm, img0_r_norm),dim=1))
        disp = F.interpolate(disp, scale_factor=0.25, mode='nearest')
        
        x = torch.cat([flow, intrinsic], dim=1)
        pose = self.flowPoseNet(x)

        return flow, disp, pose


class MultiCamVONet(nn.Module):
    def __init__(self, flowNormFactor=1.0, fix_parts=("flow"), use_stereo=2.1,
                    extrinsic_encoder_layers=2, trans_head_layers=3):
        super(MultiCamVONet, self).__init__()

        from .PWC import PWCDCNet as FlowNet
        self.flowNet = FlowNet(uncertainty=False)

        from .VOFlowNet import VOFlowRes as FlowPoseNet
        self.flowPoseNet = FlowPoseNet(intrinsic=True, down_scale=True, config=1, stereo=use_stereo, fix_parts=fix_parts,
                                        extrinsic_encoder_layers=extrinsic_encoder_layers, trans_head_layers=trans_head_layers)

        self.flowNormFactor = flowNormFactor

        if "flow" in fix_parts:
            for param in self.flowNet.parameters():
                param.requires_grad = False

    def forward(self, imgA, imgB, imgC, intrinsic, extrinsic):
        # import ipdb;ipdb.set_trace()
        flowAB, _ = self.flowNet(torch.cat([imgA, imgB], dim=1))
        flowAC, _ = self.flowNet(torch.cat([imgA, imgC], dim=1))
                
        flowAB = flowAB[0] * self.flowNormFactor
        flowAC = flowAC[0] * self.flowNormFactor

        x = torch.cat([flowAB, flowAC, intrinsic], dim=1)
        pose = self.flowPoseNet(x, extrinsic=extrinsic)

        return flowAB, flowAC, pose

    # def get_flow_loss(self, output, target, criterion, mask=None, unc=None, lamb=1.0):
    #     '''
    #     Note: criterion is not used when uncertainty is included
    #     '''
    #     if mask is not None: 
    #         output_ = output[mask]
    #         target_ = target[mask]
    #         if unc is not None:
    #             unc = unc[mask]
    #     else:
    #         output_ = output
    #         target_ = target

    #     if unc is None:
    #         return criterion(output_, target_), criterion(output_, target_)
    #     else: # if using uncertainty, then no mask 
    #         diff = torch.abs( output_ - target_) # hard code L1 loss
    #         loss_unc = torch.mean(torch.exp(-unc) * diff + unc * lamb)
    #         loss = torch.mean(diff)
    #         return  loss_unc/(1.0+lamb), loss


# if __name__ == '__main__':
#     voflownet = VONet(network=0, intrinsic=True, flowNormFactor=1.0, down_scale=True, config=1, fixflow=True) # 
#     voflownet.cuda()
#     voflownet.eval()
#     print (voflownet)
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import time

#     x, y = np.ogrid[:448, :640]
#     # print (x, y, (x+y))
#     img = np.repeat((x + y)[..., np.newaxis], 3, 2) / float(512 + 384)
#     img = img.astype(np.float32)
#     print (img.dtype)
#     imgInput = img[np.newaxis,...].transpose(0, 3, 1, 2)
#     intrin = imgInput[:,:2,:112,:160].copy()

#     imgTensor = torch.from_numpy(imgInput)
#     intrinTensor = torch.from_numpy(intrin)
#     print (imgTensor.shape)
#     stime = time.time()
#     for k in range(100):
#         flow, pose = voflownet((imgTensor.cuda(), imgTensor.cuda(), intrinTensor.cuda()))
#         print (flow.data.shape, pose.data.shape)
#         print (pose.data.cpu().numpy())
#         print (time.time()-stime)
#     print (time.time()-stime)/100
