import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .module import *

Align_Corners_Range = False


class DepthNet(nn.Module):
    def __init__(self, mode="regression"):
        super(DepthNet, self).__init__()

    def forward(self, cost_reg, depth_values, num_depth, interval, prob_volume_init=None,stage=0):



        prob_volume = F.softmax(cost_reg, dim=2)  # (b,2, ndepth, h, w)
        depth_sub_plus = depth_regression(prob_volume, depth_values=depth_values.unsqueeze(1),axis=2)  # (b, h, w)

        depth_sup_plus_small,depth_sup_plus_huge=depth_sub_plus.split([2,2],dim=1)
        
        
        small_min,small_max=depth_sup_plus_small.min(1)[0],depth_sup_plus_small.max(1)[0]
        huge_min,huge_max=depth_sup_plus_huge.min(1)[0],depth_sup_plus_huge.max(1)[0]
        huge_min_d,huge_max_d=2*huge_min-huge_max,2*huge_max-huge_min
        small_min_d,small_max_d=2*small_min-small_max,2*small_max-small_min

        coors=torch.stack( 
            [item.unsqueeze(0).expand_as(depth_sub_plus[:,0]) for item in torch.meshgrid(*[torch.arange(0, s) for s in depth_sub_plus[:,0].shape[-2:]])],
            axis=-1).to(depth_sub_plus[:,0].device)
        mask_00=((coors[:,:,:,0]%4==0)&(coors[:,:,:,1]%2==0))
        mask_01=((coors[:,:,:,0]%4==0)&(coors[:,:,:,1]%2==1))
        mask_10=((coors[:,:,:,0]%4==1)&(coors[:,:,:,1]%2==0))
        mask_11=((coors[:,:,:,0]%4==1)&(coors[:,:,:,1]%2==1))
        mask_20=((coors[:,:,:,0]%4==2)&(coors[:,:,:,1]%2==0))
        mask_21=((coors[:,:,:,0]%4==2)&(coors[:,:,:,1]%2==1))
        mask_30=((coors[:,:,:,0]%4==3)&(coors[:,:,:,1]%2==0))
        mask_31=((coors[:,:,:,0]%4==3)&(coors[:,:,:,1]%2==1))
        
        small_stack=torch.stack((3*small_min-2*small_max,2*small_min-small_max,small_min,small_max,2*small_max-small_min,3*small_max-2*small_min),1)
        small_stack_d=torch.stack((3*small_min_d-2*small_max_d,2*small_min_d-small_max_d,small_min_d,small_max_d,2*small_max_d-small_min_d,3*small_max_d-2*small_min_d),1)
        huge_stack=torch.stack((3*huge_min-2*huge_max,2*huge_min-huge_max,huge_min,huge_max,2*huge_max-huge_min,3*huge_max-2*huge_min),1)
        huge_stack_d=torch.stack((3*huge_min_d-2*huge_max_d,2*huge_min_d-huge_max_d,huge_min_d,huge_max_d,2*huge_max_d-huge_min_d,3*huge_max_d-2*huge_min_d),1)
        
        # depth=torch.zeros_like(depth_sub_plus[:,0])
        depth_values_c=torch.zeros_like(depth_sub_plus)
        depth_values_c=torch.where(mask_00.unsqueeze(1),small_stack[:,:-2],depth_values_c)
        depth_values_c=torch.where(mask_01.unsqueeze(1),small_stack[:,2:],depth_values_c)
        depth_values_c=torch.where(mask_10.unsqueeze(1),huge_stack[:,2:],depth_values_c)
        depth_values_c=torch.where(mask_11.unsqueeze(1),huge_stack[:,:-2],depth_values_c)
        depth_values_c=torch.where(mask_20.unsqueeze(1),small_stack_d[:,:-2],depth_values_c)
        depth_values_c=torch.where(mask_21.unsqueeze(1),small_stack_d[:,2:],depth_values_c)
        depth_values_c=torch.where(mask_30.unsqueeze(1),huge_stack_d[:,2:],depth_values_c)
        depth_values_c=torch.where(mask_31.unsqueeze(1),huge_stack_d[:,:-2],depth_values_c)

       
        with torch.no_grad():
            # photometric confidence
            temp_photometric_confidence=torch.sigmoid(interval/(depth_sub_plus.var(1,unbiased=False).sqrt()+1e-5))
            photometric_confidence=2*(temp_photometric_confidence-0.5)
        

        return {"photometric_confidence": photometric_confidence, "prob_volume": prob_volume,"depth_sub_plus":depth_sub_plus,"depth_values_c":depth_values_c,
                "depth_values": depth_values, "interval": interval}
    def refine(self, cost_reg, depth_values, num_depth, interval,alpha=5):
        prob_volume = F.softmax(cost_reg*alpha, dim=2)  # (b,2, ndepth, h, w)
        depth_sub_plus = depth_regression(prob_volume, depth_values=depth_values.unsqueeze(1),axis=2)  # (b, h, w)

        depth_sup_plus_small,depth_sup_plus_huge=depth_sub_plus.split([2,2],dim=1)
        
        
        small_min,small_max=depth_sup_plus_small.min(1)[0],depth_sup_plus_small.max(1)[0]
        huge_min,huge_max=depth_sup_plus_huge.min(1)[0],depth_sup_plus_huge.max(1)[0]

        coors=torch.stack( 
            [item.unsqueeze(0).expand_as(depth_sub_plus[:,0]) for item in torch.meshgrid(*[torch.arange(0, s) for s in depth_sub_plus[:,0].shape[-2:]])],
            axis=-1).to(depth_sub_plus[:,0].device)
        mask_00=((coors[:,:,:,0]%2==0)&(coors[:,:,:,1]%2==0))
        mask_01=((coors[:,:,:,0]%2==0)&(coors[:,:,:,1]%2==1))
        mask_10=((coors[:,:,:,0]%2==1)&(coors[:,:,:,1]%2==0))
        mask_11=((coors[:,:,:,0]%2==1)&(coors[:,:,:,1]%2==1))
        

        depth=torch.zeros_like(depth_sub_plus[:,0])

        depth=torch.where(mask_00,small_min,depth)
        depth=torch.where(mask_01,small_max,depth)
        depth=torch.where(mask_10,huge_max,depth)
        depth=torch.where(mask_11,huge_min,depth)

       
        with torch.no_grad():
            # photometric confidence
            temp_photometric_confidence=torch.sigmoid(interval/(depth_sub_plus.var(1,unbiased=False).sqrt()+1e-5))
            photometric_confidence=2*(temp_photometric_confidence-0.5)
        

        return {"depth": depth, "photometric_confidence_refine": photometric_confidence,"depth_sub_plus_refine":depth_sub_plus}

class CostAgg(nn.Module):
    def __init__(self, mode="variance", in_channels=None):
        super(CostAgg, self).__init__()
        self.mode = mode
        assert mode in ("variance", "adaptive"), "Don't support {}!".format(mode)
        if self.mode == "adaptive":
            self.weight_net = nn.ModuleList([AggWeightNetVolume(in_channels[i]) for i in range(len(in_channels))])


    def forward(self, features, proj_matrices, depth_values, stage_idx):
        """
        :param stage_idx: stage
        :param features: [ref_fea, src_fea1, src_fea2, ...], fea shape: (b, c, h, w)
        :param proj_matrices: (b, nview, ...) [ref_proj, src_proj1, src_proj2, ...]
        :param depth_values: (b, ndepth, h, w)
        :return: matching cost volume (b, c, ndepth, h, w)
        """
        ref_feature, src_features = features[0], features[1:]
        proj_matrices = torch.unbind(proj_matrices, 1)  # to list
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        num_views = len(features)
        num_depth = depth_values.shape[1]

        ref_volume = ref_feature.unsqueeze(2)
        
        similarity_sum = 0

        b,c,_,h,w=ref_volume.shape
        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume,_ = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)

            similarity=(warped_volume.view(b,c//2,2,warped_volume.shape[2],h,w)*(ref_volume.view(b,c//2,2,1,h,w))).mean(1)
            
            if self.training:
                similarity_sum = similarity_sum + similarity # [B, 2, D, H, W]
                
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                similarity_sum += similarity
                
            
            
            del warped_volume

        # aggregate multiple feature volumes by variance
        return similarity_sum


class MVSNet(nn.Module):
    def __init__(self, ndepths, depth_interval_ratio, cr_base_chs=None, fea_mode="fpn", agg_mode="variance", depth_mode="regression",winner_take_all_to_generate_depth=True,inverse_depth=False):
        super(MVSNet, self).__init__()

        if cr_base_chs is None:
            cr_base_chs = [8] * len(ndepths)
        self.ndepths = ndepths
        self.depth_interval_ratio = depth_interval_ratio
        self.fea_mode = fea_mode
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        self.inverse_depth=inverse_depth

        print("netphs:", ndepths)
        print("depth_intervals_ratio:", depth_interval_ratio)
        print("cr_base_chs:", cr_base_chs)
        print("fea_mode:", fea_mode)
        print("agg_mode:", agg_mode)
        print("depth_mode:", depth_mode)

        assert len(ndepths) == len(depth_interval_ratio)

        self.feature = FeatureNet(base_channels=8, stride=4, num_stage=self.num_stage, mode=self.fea_mode)
        self.cost_aggregation = CostAgg(agg_mode, self.feature.out_channels)

        self.cost_regularization = nn.ModuleList(
            [CostRegNet(in_channels=2, base_channels=self.cr_base_chs[i],stage=i) for i in range(self.num_stage)])
        self.cost_regularization_refine = nn.ModuleList(
            [CostRegNet_refine(in_channels=2, base_channels=self.cr_base_chs[i],stage=i) for i in range(self.num_stage)])

        self.DepthNet = DepthNet(depth_mode)

    def forward(self, imgs, proj_matrices, depth_values):
        """
        :param is_flip: augment only for 3D-UNet
        :param imgs: (b, nview, c, h, w)
        :param proj_matrices:
        :param depth_values:
        :return:
        """
        depth_interval = (depth_values[0, -1] - depth_values[0, 0]) / depth_values.size(1)

        # step 1. feature extraction
        features = []
        for nview_idx in range(imgs.size(1)):  # imgs shape (B, N, C, H, W)
            img = imgs[:, nview_idx]
            features.append(self.feature(img))

        ori_shape = imgs[:, 0].shape[2:]  # (H, W)

        outputs = {}
        last_depth = None
        for stage_idx in range(self.num_stage):
            # print("*********************stage{}*********************".format(stage_idx + 1))
            # stage feature, proj_mats, scales
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            # stage1: 1/4, stage2: 1/2, stage3: 1
            stage_scale = 2 ** (3 - stage_idx - 1)

            stage_shape = [ori_shape[0] // int(stage_scale), ori_shape[1] // int(stage_scale)]

            if stage_idx == 0:
                last_depth = depth_values
            else:
                last_depth = last_depth.detach()

            # (B, D, H, W)
            depth_range_samples, interval = get_depth_range_samples(last_depth=last_depth,
                                                                    ndepth=self.ndepths[stage_idx],
                                                                    depth_inteval_pixel=self.depth_interval_ratio[
                                                                                            stage_idx] * depth_interval,
                                                                    shape=stage_shape,  # only for first stage
                                                                    inverse=self.inverse_depth
                                                                    )

            if stage_idx > 0:
                depth_range_samples = F.interpolate(depth_range_samples, stage_shape, mode='bilinear', align_corners=Align_Corners_Range)

            # (b, c, d, h, w)
            cost_volume = self.cost_aggregation(features_stage, proj_matrices_stage, depth_range_samples, stage_idx)
            # cost volume regularization
            # (b, 1, d, h, w)
            cost_reg = self.cost_regularization[stage_idx](cost_volume)

            # depth
            outputs_stage = self.DepthNet(cost_reg, depth_range_samples, num_depth=self.ndepths[stage_idx], interval=interval,stage=stage_idx)

            
            
            depth_values_c=outputs_stage["depth_values_c"]
            features_stage = [feat["stage{}_c".format(stage_idx + 1)] for feat in features]
           
                         
            cost_volume_c = self.cost_aggregation(features_stage, proj_matrices_stage, depth_values_c, stage_idx)
            cost_reg_c= self.cost_regularization_refine[stage_idx](cost_volume_c)
            outputs_stage_refine = self.DepthNet.refine(cost_reg_c, depth_values_c, num_depth=4, interval=interval)
            
            outputs_stage={**outputs_stage_refine,**outputs_stage}
            last_depth = outputs_stage['depth']

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        return outputs
