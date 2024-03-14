import cv2
import torch
import numbers
import numpy as np


# ===== general functions =====

KEY2DIM = {
    'img0':3, 'img1':3, 'img0_norm':3, 'img1_norm':3,
    'intrinsic':3, 'flow':3, 'fmask':2,
    'disp0':2, 'disp1':2, 'depth0':2, 'depth1':2,
    'flow_unc':2, 'depth0_unc':2,
    'img0_r':3, 'img1_r':3, 'img0_r_norm':3, 'img1_r_norm':3
}


class Compose(object):
    """
    Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def get_sample_dimention(sample):
    for kk in sample.keys():
        if kk in KEY2DIM: # for sequencial data
            h, w = sample[kk][0].shape[0], sample[kk][0].shape[1]
            return h, w
    assert False,"No image type in {}".format(sample.keys())


class CropCenter(object):
    """
    Crops the a sample of data (tuple) at center
    If the image size is not large enough, it will be first resized with fixed ratio.
    If fix_ratio is False, w and h are resized separatedly.
    If scale_w is given, w will be resized accordingly.
    """

    def __init__(self, size, fix_ratio=True, scale_w=1.0, scale_disp=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.fix_ratio = fix_ratio
        self.scale_w = scale_w
        self.scale_disp = scale_disp

    def __call__(self, sample):
        th, tw = self.size
        hh, ww = get_sample_dimention(sample)
        if ww == tw and hh == th:
            return sample
        # resize the image if the image size is smaller than the target size
        scale_h = max(1, float(th)/hh)
        scale_w = max(1, float(tw)/ww)
        if scale_h>1 or scale_w>1:
            if self.fix_ratio:
                scale_h = max(scale_h, scale_w)
                scale_w = max(scale_h, scale_w)
            w = int(round(ww * scale_w)) # w after resize
            h = int(round(hh * scale_h)) # h after resize
        else:
            w, h = ww, hh
        if self.scale_w != 1.0:
            scale_w = self.scale_w
            w = int(round(ww * scale_w))
        if scale_h != 1. or scale_w != 1.: # resize the data
            resizedata = ResizeData(size=(h, w), scale_disp=self.scale_disp)
            sample = resizedata(sample)
        x1 = int((w-tw)/2)
        y1 = int((h-th)/2)
        for kk in sample.keys():
            if sample[kk] is None or (kk not in KEY2DIM):
                continue
            seqlen = len(sample[kk])
            datalist = []
            for k in range(seqlen): 
                datalist.append(sample[kk][k][y1:y1+th,x1:x1+tw,...])
            sample[kk] = datalist
        if 'intrinsic_calib' in sample:
            sample['intrinsic_calib'][2] -= x1
            sample['intrinsic_calib'][3] -= y1
        return sample


class ResizeData(object):
    """
    Resize the data in a dict.
    """

    def __init__(self, size, scale_disp=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.scale_disp = scale_disp

    def resize_seq(self, dataseq, w, h):
        seqlen = dataseq.shape[0]
        datalist = []
        for k in range(seqlen): 
            datalist.append(cv2.resize(dataseq[k], (w,h), interpolation=cv2.INTER_LINEAR))
        return np.stack(datalist, axis=0)

    def __call__(self, sample):
        th, tw = self.size
        h, w = get_sample_dimention(sample)
        if w == tw and h == th:
            return sample
        scale_w = float(tw)/w
        scale_h = float(th)/h
        for kk in sample.keys():
            if sample[kk] is None or (kk not in KEY2DIM):
                continue
            seqlen = len(sample[kk])
            datalist = []
            for k in range(seqlen): 
                datalist.append(cv2.resize(sample[kk][k], (tw,th), interpolation=cv2.INTER_LINEAR))
            sample[kk] = datalist
        if 'flow' in sample:
            for k in range(len(sample['flow'])):
                sample['flow'][k][...,0] = sample['flow'][k][...,0] * scale_w
                sample['flow'][k][...,1] = sample['flow'][k][...,1] * scale_h
        if self.scale_disp: # scale the depth
            if 'disp0' in sample:
                for k in range(len(sample['disp0'])):
                    sample['disp0'][k] = sample['disp0'][k] * scale_w
            if 'disp1' in sample:
                for k in range(len(sample['disp1'])):
                    sample['disp1'][k] = sample['disp1'][k] * scale_w
        else:
            sample['scale_w'] = np.array([scale_w ],dtype=np.float32)# used in e2e-stereo-vo
        if 'intrinsic_calib' in sample:
            sample['intrinsic_calib'][0] *= scale_w
            sample['intrinsic_calib'][2] *= scale_w
            sample['intrinsic_calib'][1] *= scale_h
            sample['intrinsic_calib'][3] *= scale_h
        return sample
    

class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        for kk in sample.keys():
            if not kk in KEY2DIM:
                continue
            if KEY2DIM[kk] == 3: # for sequencial data
                data = np.stack(sample[kk], axis=0)
                data = data.transpose(0, 3, 1, 2) # frame x channel x h x w
            elif KEY2DIM[kk] == 2: # for sequencial data
                data = np.stack(sample[kk], axis=0)
                data = data[:, np.newaxis, :, :] # frame x channel x h x w
            data = data.astype(np.float32)
            sample[kk] = torch.from_numpy(data.copy()) # copy to make memory continuous
        return sample
    

class SqueezeBatchDim(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        for kk in sample.keys():
            if not kk in KEY2DIM:
                continue
            sample[kk] = sample[kk].squeeze(0)
        return sample
    

class Normalize(object):
    """
    Given mean: (R, G, B) and std: (R, G, B).
    This option should be before the to tensor.
    """

    def __init__(self, mean=None, std=None, rgbbgr=False, keep_old=False):
        '''
        keep_old: keep both normalized and unnormalized data,
        normalized data will be put under new key xxx_norm.
        '''
        self.mean = mean
        self.std = std
        self.rgbbgr = rgbbgr
        self.keep_old = keep_old

    def __call__(self, sample):
        keys = list(sample.keys())
        for kk in keys:
            if kk.startswith('img0') or kk.startswith('img1'):
                # sample[kk] is a list, sample[kk][k]: h x w x 3
                seqlen = len(sample[kk])
                datalist = []
                for s in range(seqlen):
                    sample[kk][s] = sample[kk][s]/255.0
                    if self.rgbbgr:
                        img = sample[kk][s][...,[2,1,0]] # bgr2rgb
                    if self.mean is not None and self.std is not None:
                        img = np.zeros_like(sample[kk][s])
                        for k in range(3):
                            img[...,k] = (sample[kk][s][...,k] - self.mean[k])/self.std[k]
                    else:
                        img = sample[kk][s]
                    datalist.append(img)
                if self.keep_old:
                    sample[kk+'_norm'] = datalist
                else:
                    sample[kk] = datalist
        return sample


# ===== end-to-end flow and vo =====

class DownscaleFlow(object):
    """
    Scale the flow and mask to a fixed size.
    This function won't resize the RGBs.
    flow/disp values will NOT be changed.
    """

    def __init__(self, scale=4):
        '''
        size: output frame size, this should be NO LARGER than the input frame size! 
        '''
        self.downscale = 1.0/scale

    def __call__(self, sample): 
        if self.downscale==1:
            return sample
        for key in sample.keys():
            if key in {'flow','intrinsic','fmask','disp0','depth0'}:
                imgseq = []
                for k in range(len(sample[key])):
                    imgseq.append(cv2.resize(sample[key][k], 
                        (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_NEAREST))
                sample[key] = imgseq
        return sample

    
def calculate_angle_distance_from_du_dv(du, dv, flagDegree=False):
    a = np.arctan2( dv, du )
    angleShift = np.pi
    if True == flagDegree:
        a = a / np.pi * 180
        angleShift = 180
    d = np.sqrt( du * du + dv * dv )
    return a, d, angleShift


def visrgb(img, mean=None, std=None):
    if mean is not None and std is not None:
        for k in range(3):
            img[...,k] = img[...,k] * std[k] + mean[k]
    return (img*255).astype(np.uint8)


def visflow(flownp, maxF=500.0, n=8, mask=None, hueMax=179, angShift=0.0): 
    """
    Show a optical flow field as the KITTI dataset does.
    Some parts of this function is the transform of the original MATLAB code flow_to_color.m.
    """
    ang, mag, _ = calculate_angle_distance_from_du_dv( flownp[:, :, 0], flownp[:, :, 1], flagDegree=False )
    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros( ( ang.shape[0], ang.shape[1], 3 ) , dtype=np.float32)
    am = ang < 0
    ang[am] = ang[am] + np.pi * 2
    hsv[ :, :, 0 ] = np.remainder( ( ang + angShift ) / (2*np.pi), 1 )
    hsv[ :, :, 1 ] = mag / maxF * n
    hsv[ :, :, 2 ] = (n - hsv[:, :, 1])/n
    hsv[:, :,  0 ] = np.clip( hsv[:, :, 0], 0, 1 ) * hueMax
    hsv[:, :, 1:3] = np.clip( hsv[:, :, 1:3], 0, 1 ) * 255
    hsv = hsv.astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if ( mask is not None ):
        mask = mask != 255
        bgr[mask] = np.array([0, 0 ,0], dtype=np.uint8)
    return bgr


def visdepth(disp, scale=3):
    disp = disp.astype(np.float32)
    min_val = np.min(disp)
    max_val = np.max(disp)
    res = (disp - min_val) / (max_val - min_val) * 255
    return res.astype(np.uint8)


def save_images(dir, data, prefix='', suffix='', mean=None, std=None, fx=1, fy=1):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    # before: (batch, channel, height, width)
    data = data.transpose(0, 2, 3, 1)
    # after: (batch, height, width, channel)
    if data.shape[-1] == 3:
        rgb_images = []
        for i in range(data.shape[0]):
            img = visrgb(data[i], mean=mean, std=std)
            rgb_images.append(cv2.resize(img, None, fx=fx, fy=fy))
        data = np.stack(rgb_images)
    if data.shape[-1] == 2:
        flow_images = []
        for i in range(data.shape[0]):
            img = visflow(data[i])
            flow_images.append(cv2.resize(img, None, fx=fx, fy=fy))
        data = np.stack(flow_images)
    elif data.shape[-1] == 1:
        disp_images = []
        for i in range(data.shape[0]):
            img = visdepth(data[i])
            disp_images.append(cv2.resize(img, None, fx=fx, fy=fy))
        data = np.stack(disp_images)
    for i in range(data.shape[0]):
        cv2.imwrite('{}/{}{}{}.png'.format(dir, prefix, i, suffix), data[i])


def warp_images(dir, data, flow, mean=None, std=None):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    # before: (batch, channel, height, width)
    data = data.transpose(0, 2, 3, 1)
    # after: (batch, height, width, channel)
    fx = 1/4
    fy = 1/4
    img = []
    for i in range(data.shape[0]):
        rgb = visrgb(data[i], mean=mean, std=std)
        img.append(cv2.resize(rgb, None, fx=fx, fy=fy))
    img = np.stack(img)
    if torch.is_tensor(flow):
        flow = flow.detach().cpu().numpy()
    flow = flow.transpose(0, 2, 3, 1)
    fx = 1
    fy = 1
    resized_flow = []
    for i in range(flow.shape[0]):
        resized_flow.append(cv2.resize(flow[i], None, fx=fx, fy=fy))
    flow = np.stack(resized_flow)
    res = []
    for i in range(flow.shape[0]):
        f = flow[i]
        if len(f.shape) == 2:
            f = np.stack((f, np.zeros_like(f)), axis=-1)
        h, w = f.shape[:2]
        grid_x, grid_y = np.meshgrid(np.linspace(0, w-1, w), np.linspace(0, h-1, h))
        uv = np.stack((grid_x, grid_y), axis=-1)
        f = (f + uv).astype(np.float32)
        warp = cv2.remap(img[i], f, None, cv2.INTER_LINEAR)
        res.append(warp)
    res = np.stack(res)
    for i in range(res.shape[0]):
        cv2.imwrite('{}/{}_warp.png'.format(dir, i), res[i])
    return res


# ===== camera intrinsics =====

def make_intrinsics_layer(w, h, fx, fy, ox, oy):
    ww, hh = np.meshgrid(range(w), range(h))
    ww = (ww.astype(np.float32) - ox + 0.5 )/fx
    hh = (hh.astype(np.float32) - oy + 0.5 )/fy
    intrinsicLayer = np.stack((ww,hh)).transpose(1,2,0)
    return intrinsicLayer
