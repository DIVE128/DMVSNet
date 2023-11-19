import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from torch.autograd import Variable

sys.path.append("..")


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        # self.bn = nn.GroupNorm(8, out_channels) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        # self.bn = nn.GroupNorm(8, out_channels) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        # self.bn = nn.GroupNorm(8, out_channels) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        # self.bn = nn.GroupNorm(8, out_channels) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)



def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xyz[:, 2:3][proj_xyz[:, 2:3] == 0] += 0.00001  # NAN BUG, not on dtu, but on blendedmvs

        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    # warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
    #                                padding_mode='zeros').type(torch.float32)
    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros',align_corners=True).type(torch.float32)
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea,grid.view(batch, num_depth,height, width, 2)

class DeConv2dFuse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(DeConv2dFuse, self).__init__()

        self.deconv = Deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                               bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2d(2 * out_channels, out_channels, kernel_size, stride=1, padding=1,
                           bn=bn, relu=relu, bn_momentum=bn_momentum)

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x


class FeatureNet(nn.Module):
    def __init__(self, base_channels, num_stage=3, stride=4, mode="fpn",layernorm=False):
        super(FeatureNet, self).__init__()
        assert mode in ["unet", "fpn"], print("mode must be in 'unet', 'fpn', but get:{}".format(mode))
        self.mode = mode
        self.stride = stride
        self.base_channels = base_channels
        self.num_stage = num_stage
        self.layernorm=layernorm
        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )


        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4 *2, 1, bias=False)
        self.out_channels = [4 * base_channels]
        final_chs = base_channels * 4

        self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

        self.out2 = nn.Conv2d(final_chs, base_channels * 2 *2, 3, padding=1, bias=False)
        self.out3 = nn.Conv2d(final_chs, base_channels *2, 3, padding=1, bias=False)
        self.out_channels.append(base_channels * 2)
        self.out_channels.append(base_channels)
            



    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = {}

        out = self.out1(intra_feat)
        # outputs["stage1"] = out
        outputs["stage1"],outputs["stage1_c"]= out.split([out.shape[1]//2,out.shape[1]//2],1)
        
        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
        out = self.out2(intra_feat)
        # outputs["stage2"] = out
        outputs["stage2"],outputs["stage2_c"]= out.split([out.shape[1]//2,out.shape[1]//2],1)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
        out = self.out3(intra_feat)
        # outputs["stage3"] = out
        outputs["stage3"],outputs["stage3_c"]= out.split([out.shape[1]//2,out.shape[1]//2],1)



        return outputs

class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels,stage=0):
        super(CostRegNet, self).__init__()
        self.cosR_small=CostRegNet_part(in_channels, base_channels,stage=0)
        self.cosR_huge=CostRegNet_part(in_channels, base_channels,stage=0)
    def forward(self, x):
        results=torch.cat((self.cosR_small(x),self.cosR_huge(x)),axis=1)
        return results
class CostRegNet_refine(nn.Module):
    def __init__(self, in_channels, base_channels,stage=0):
        super(CostRegNet_refine, self).__init__()
        self.cosR_small=CostRegNet_part_refine(in_channels, base_channels,stage=0)
        self.cosR_huge=CostRegNet_part_refine(in_channels, base_channels,stage=0)
    def forward(self, x):
        results=torch.cat((self.cosR_small(x),self.cosR_huge(x)),axis=1)
        return results
class CostRegNet_part(nn.Module):
    def __init__(self, in_channels, base_channels,stage=0):
        super(CostRegNet_part, self).__init__()
        self.conv0 = Conv3d(in_channels, base_channels, padding=1)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        # self.prob = nn.Conv3d(base_channels, 1 if stage==0 else 2, 3, stride=1, padding=1, bias=False)
        self.prob = nn.Conv3d(base_channels, 2, 3, stride=1, padding=1, bias=False)


        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x

class CostRegNet_part_refine(nn.Module):
    def __init__(self, in_channels, base_channels,stage=0):
        super(CostRegNet_part_refine, self).__init__()
        self.conv0 = Conv3d(in_channels, base_channels, padding=1)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv2d(base_channels * 4, base_channels * 8,3, stride=2, padding=1)
        self.conv6 = Conv2d(base_channels * 8, base_channels * 8,3, padding=1)

        self.conv7 = Deconv2d(base_channels * 8, base_channels * 4,3, stride=2, padding=1, output_padding=1)

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        # self.prob = nn.Conv3d(base_channels, 1 if stage==0 else 2, 3, stride=1, padding=1, bias=False)
        self.prob = nn.Conv3d(base_channels, 2, 3, stride=1, padding=1, bias=False)


        

    def forward(self, x,stage=0):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2)).squeeze(2)
        x=self.conv6(self.conv5(conv4))
        x=conv4+self.conv7(x)
        x=x.unsqueeze(2)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x
class AggWeightNetVolume(nn.Module):
    def __init__(self, in_channels=32,hid_channels=1,out_channels=1,relu=True):
        super(AggWeightNetVolume, self).__init__()
        self.w_net = nn.Sequential(
            Conv3d(in_channels, hid_channels, kernel_size=1, stride=1, padding=0,relu=relu),
            Conv3d(hid_channels, out_channels, kernel_size=1, stride=1, padding=0,relu=relu)
        )

    def forward(self, x):
        """
        :param x: (b, c, d, h, w)
        :return: (b, 1, d, h, w)
        """
        w = self.w_net(x)
        return w


def depth_regression(p, depth_values,axis=1):
    if depth_values.dim() <= 2:
        # print("regression dim <= 2")
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, axis=axis)

    return depth


def winner_take_all(prob_volume, depth_values):
    """
    :param prob_volume: (b, d, h, w)
    :param depth_values: (b, d, h, w)
    :return: (b, h, w)
    """
    _, idx = torch.max(prob_volume, dim=1, keepdim=True)
    depth = torch.gather(depth_values, 1, idx).squeeze(1)
    return depth




def get_cur_depth_range_samples_n(last_depth, ndepth, depth_inteval_pixel):
    # cur_depth: (B, H, W)
    # return depth_range_values: (B, D, H, W)
    last_depth_min = (last_depth - (ndepth+2) / 2 * depth_inteval_pixel)  # (B, H, W)
    last_depth_max = (last_depth + (ndepth-2) / 2 * depth_inteval_pixel)
    # cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel).clamp(min=0.0)   #(B, H, W)
    # cur_depth_max = (cur_depth_min + (ndepth - 1) * depth_inteval_pixel).clamp(max=max_depth)

    new_interval = (last_depth_max - last_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = last_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=last_depth.device,
                                                                      dtype=last_depth.dtype,
                                                                      requires_grad=False).reshape(1, -1, 1,
                                                                                                   1) * new_interval.unsqueeze(1))

    return depth_range_samples, (ndepth * depth_inteval_pixel) / (ndepth - 1)
def get_cur_depth_range_samples_p(last_depth, ndepth, depth_inteval_pixel):
    # cur_depth: (B, H, W)
    # return depth_range_values: (B, D, H, W)
    last_depth_min = (last_depth - (ndepth-2) / 2 * depth_inteval_pixel)  # (B, H, W)
    last_depth_max = (last_depth + (ndepth+2) / 2 * depth_inteval_pixel)
    # cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel).clamp(min=0.0)   #(B, H, W)
    # cur_depth_max = (cur_depth_min + (ndepth - 1) * depth_inteval_pixel).clamp(max=max_depth)

    new_interval = (last_depth_max - last_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = last_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=last_depth.device,
                                                                      dtype=last_depth.dtype,
                                                                      requires_grad=False).reshape(1, -1, 1,
                                                                                                   1) * new_interval.unsqueeze(1))

    return depth_range_samples, (ndepth * depth_inteval_pixel) / (ndepth - 1)

def get_cur_depth_range_samples_inverse(last_depth, ndepth, depth_inteval_pixel):
    # cur_depth: (B, H, W)
    # return depth_range_values: (B, D, H, W)
    last_depth_min = (last_depth - ndepth / 2 * depth_inteval_pixel)  # (B, H, W)
    last_depth_max = (last_depth + ndepth / 2 * depth_inteval_pixel)
    inverse_min=1/last_depth_min
    inverse_max=1/last_depth_max
    new_interval=(inverse_max-inverse_min)/(ndepth-1)
    inverse_depth_range_samples=inverse_min.unsqueeze(1)+(torch.arange(0, ndepth, device=last_depth.device,
                                                                      dtype=last_depth.dtype,
                                                                      requires_grad=False).reshape(1, -1, 1,
                                                                                                   1) * new_interval.unsqueeze(1))
                
    depth_range_samples=1/inverse_depth_range_samples
    return depth_range_samples, (ndepth * depth_inteval_pixel) / (ndepth - 1)

def get_cur_depth_range_samples_inverse_p(last_depth, ndepth, depth_inteval_pixel):
    # cur_depth: (B, H, W)
    # return depth_range_values: (B, D, H, W)
    last_depth_min = (last_depth - (ndepth-2) / 2 * depth_inteval_pixel)  # (B, H, W)
    last_depth_max = (last_depth + (ndepth+2) / 2 * depth_inteval_pixel)
    inverse_min=1/last_depth_min
    inverse_max=1/last_depth_max
    new_interval=(inverse_max-inverse_min)/(ndepth-1)
    inverse_depth_range_samples=inverse_min.unsqueeze(1)+(torch.arange(0, ndepth, device=last_depth.device,
                                                                      dtype=last_depth.dtype,
                                                                      requires_grad=False).reshape(1, -1, 1,
                                                                                                   1) * new_interval.unsqueeze(1))
                
    depth_range_samples=1/inverse_depth_range_samples
    return depth_range_samples, (ndepth * depth_inteval_pixel) / (ndepth - 1)
def get_cur_depth_range_samples_inverse_n(last_depth, ndepth, depth_inteval_pixel):
    # cur_depth: (B, H, W)
    # return depth_range_values: (B, D, H, W)
    last_depth_min = (last_depth - (ndepth+2) / 2 * depth_inteval_pixel)  # (B, H, W)
    last_depth_max = (last_depth + (ndepth-2) / 2 * depth_inteval_pixel)
    inverse_min=1/last_depth_min
    inverse_max=1/last_depth_max
    new_interval=(inverse_max-inverse_min)/(ndepth-1)
    inverse_depth_range_samples=inverse_min.unsqueeze(1)+(torch.arange(0, ndepth, device=last_depth.device,
                                                                      dtype=last_depth.dtype,
                                                                      requires_grad=False).reshape(1, -1, 1,
                                                                                                   1) * new_interval.unsqueeze(1))
                
    depth_range_samples=1/inverse_depth_range_samples
    return depth_range_samples, (ndepth * depth_inteval_pixel) / (ndepth - 1)

def get_depth_range_samples(last_depth, ndepth, depth_inteval_pixel, shape=None,next_depth_inteval_pixel=None,inverse=False):
    # cur_depth: (B, H, W) or (B, D)
    # return depth_range_samples: (B, D, H, W)
    if not inverse:
        if last_depth.dim() == 2:
            last_depth_min = last_depth[:, 0]  # (B,)
            last_depth_max = last_depth[:, -1]
            new_interval = (last_depth_max - last_depth_min) / (ndepth - 1)  # (B, )
            stage_interval = new_interval[0]

            depth_range_samples = last_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=last_depth.device, dtype=last_depth.dtype,
                                                                            requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(
                1))  # (B, D)

            # (B, D, H, W)
            depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[0], shape[1])
            
            coors=torch.stack( 
                [item.expand_as(depth_range_samples) \
                for item in torch.meshgrid(*[torch.arange(0, s) for s in depth_range_samples.shape[-2:]])],
                axis=-1).to(depth_range_samples.device)
            mask=((coors[:,:,:,:,0]%2==0)&(coors[:,:,:,:,1]%2==0))|((coors[:,:,:,:,0]%2==1)&(coors[:,:,:,:,1]%2==1))

            depth_range_samples=torch.where(mask,depth_range_samples-stage_interval,depth_range_samples+stage_interval)
            # depth_range_samples=torch.ma

        else:

            depth_range_samples_n, stage_interval = get_cur_depth_range_samples_n(last_depth, ndepth, depth_inteval_pixel)
            depth_range_samples_p, stage_interval = get_cur_depth_range_samples_p(last_depth, ndepth, depth_inteval_pixel)
            coors=torch.stack( 
                [item.expand_as(depth_range_samples_n) \
                for item in torch.meshgrid(*[torch.arange(0, s) for s in last_depth.shape[-2:]])],
                axis=-1).to(depth_range_samples_n.device)
            mask=((coors[:,:,:,:,0]%2==0)&(coors[:,:,:,:,1]%2==0))|((coors[:,:,:,:,0]%2==1)&(coors[:,:,:,:,1]%2==1))
            depth_range_samples=torch.where(mask,\
                        depth_range_samples_n,
                        depth_range_samples_p
                        )

        return depth_range_samples, stage_interval
    else:
        if last_depth.dim() == 2:

            last_depth_min = last_depth[:, 0]  # (B,)
            last_depth_max = last_depth[:, -1]
            new_interval = (last_depth_max - last_depth_min) / (ndepth - 1)  # (B, )
            stage_interval = new_interval[0]


            last_depth_min = last_depth[:, 0]-stage_interval
            last_depth_max = last_depth[:, -1]-stage_interval

            new_interval = (last_depth_max - last_depth_min) / (ndepth - 1)  # (B, )
            stage_interval = new_interval[0]
            depth_values=[]
            for bg,end in zip(last_depth_min,last_depth_max):
                depth_values.append(torch.linspace(1 / bg , 1 / end, ndepth,device=last_depth.device))
            depth_values=torch.stack(depth_values,dim=0)
            depth_range_samples_n = (1 / depth_values).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[0], shape[1])
            
            last_depth_min = last_depth[:, 0]+stage_interval
            last_depth_max = last_depth[:, -1]+stage_interval

            new_interval = (last_depth_max - last_depth_min) / (ndepth - 1)  # (B, )
            stage_interval = new_interval[0]
            depth_values=[]
            for bg,end in zip(last_depth_min,last_depth_max):
                depth_values.append(torch.linspace(1 / bg , 1 / end, ndepth,device=last_depth.device))
            depth_values=torch.stack(depth_values,dim=0)
            depth_range_samples_p = (1 / depth_values).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[0], shape[1])
            
            coors=torch.stack( 
                [item.expand_as(depth_range_samples_p) \
                for item in torch.meshgrid(*[torch.arange(0, s) for s in depth_range_samples_p.shape[-2:]])],
                axis=-1).to(depth_range_samples_p.device)
            mask=((coors[:,:,:,:,0]%2==0)&(coors[:,:,:,:,1]%2==0))|((coors[:,:,:,:,0]%2==1)&(coors[:,:,:,:,1]%2==1))

            depth_range_samples=torch.where(mask,depth_range_samples_n,depth_range_samples_p)

        else:
            # depth_range_samples, stage_interval = get_cur_depth_range_samples_inverse(last_depth, ndepth, depth_inteval_pixel)
            depth_range_samples_n, stage_interval = get_cur_depth_range_samples_inverse_n(last_depth, ndepth, depth_inteval_pixel)
            depth_range_samples_p, stage_interval = get_cur_depth_range_samples_inverse_p(last_depth, ndepth, depth_inteval_pixel)
            coors=torch.stack( 
                [item.expand_as(depth_range_samples_n) \
                for item in torch.meshgrid(*[torch.arange(0, s) for s in last_depth.shape[-2:]])],
                axis=-1).to(depth_range_samples_n.device)
            mask=((coors[:,:,:,:,0]%2==0)&(coors[:,:,:,:,1]%2==0))|((coors[:,:,:,:,0]%2==1)&(coors[:,:,:,:,1]%2==1))
            depth_range_samples=torch.where(mask,\
                        depth_range_samples_n,
                        depth_range_samples_p
                        )
        return depth_range_samples.float(), stage_interval.float()
