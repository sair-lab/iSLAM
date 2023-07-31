
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
