import torch
import torch.nn.functional as F
import numpy as np

def mvs_loss(inputs, depth_gt_ms, mask_ms, mode, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", [1.0 for k in inputs.keys() if "stage" in k])
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)
    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        prob_volume = stage_inputs["prob_volume"]  if "global_volume" not in stage_inputs else stage_inputs["global_volume"]# (b, d, h, w)
        depth_values = stage_inputs["depth_values"] if "depth_values_new" not in stage_inputs else  stage_inputs["depth_values_new"]# (b, d, h, w)
        interval = stage_inputs["interval"]  # float
        depth_gt = depth_gt_ms[stage_key]  # (b, h, w)
        mask = mask_ms[stage_key]

        mask = mask > 0.5

        stage_idx = int(stage_key.replace("stage", "")) - 1
        stage_weight = depth_loss_weights[stage_idx]
        

        if mode == "regression":
        
            depth_sub_plus=stage_inputs["depth_sub_plus"]
            depth_sup_plus_small,depth_sup_plus_huge=depth_sub_plus.split([2,2],dim=1)
            loss_depth=2*regression_loss(depth_sup_plus_small, depth_gt.unsqueeze(1).expand_as(depth_sup_plus_small), mask.unsqueeze(1).expand_as(depth_sup_plus_small),torch.ones_like(depth_sup_plus_small)*stage_weight)\
                        +2*regression_loss(depth_sup_plus_huge, depth_gt.unsqueeze(1).expand_as(depth_sup_plus_huge), mask.unsqueeze(1).expand_as(depth_sup_plus_huge),torch.ones_like(depth_sup_plus_huge)*stage_weight)
            
            
            var_gt=torch.where((depth_sub_plus[:,0]-depth_gt).abs()<(depth_sub_plus[:,1]-depth_gt).abs(),(depth_sub_plus[:,1]-depth_gt).abs(),(depth_sub_plus[:,0]-depth_gt).abs())
            loss_var_small=regression_loss((depth_sub_plus[:,0]-depth_sub_plus[:,1]).abs(), var_gt, mask,torch.ones_like(var_gt)*stage_weight)

            var_gt=torch.where((depth_sub_plus[:,2]-depth_gt).abs()<(depth_sub_plus[:,3]-depth_gt).abs(),(depth_sub_plus[:,3]-depth_gt).abs(),(depth_sub_plus[:,2]-depth_gt).abs())
            loss_var_huge=regression_loss((depth_sub_plus[:,2]-depth_sub_plus[:,3]).abs(), var_gt, mask,torch.ones_like(var_gt)*stage_weight)
            
            
            coors=torch.stack( 
            [item.unsqueeze(0).expand_as(depth_sub_plus[:,0]) for item in torch.meshgrid(*[torch.arange(0, s) for s in depth_sub_plus[:,0].shape[-2:]])],
            axis=-1).to(depth_sub_plus[:,0].device)
            coor_mask=((coors[:,:,:,0]%2==0)&(coors[:,:,:,1]%2==0))|((coors[:,:,:,0]%2==1)&(coors[:,:,:,1]%2==1))# 
            
            small_min,small_max=depth_sup_plus_small.min(1)[0],depth_sup_plus_small.max(1)[0]
            huge_min,huge_max=depth_sup_plus_huge.min(1)[0],depth_sup_plus_huge.max(1)[0]
        
            loss_m=Monte_Carlo_sampling_loss(torch.where(coor_mask,small_min,small_max),depth_gt,mask,torch.ones_like(depth_gt)*stage_weight,mode="center",regress_fn=regression_loss)+\
                Monte_Carlo_sampling_loss(torch.where(~coor_mask,small_min,small_max),depth_gt,mask,torch.ones_like(depth_gt)*stage_weight,mode="center",regress_fn=regression_loss)+\
                Monte_Carlo_sampling_loss(torch.where(coor_mask,huge_min,huge_max),depth_gt,mask,torch.ones_like(depth_gt)*stage_weight,mode="center",regress_fn=regression_loss)+\
                Monte_Carlo_sampling_loss(torch.where(~coor_mask,huge_min,huge_max),depth_gt,mask,torch.ones_like(depth_gt)*stage_weight,mode="center",regress_fn=regression_loss)

            total_loss+=(loss_depth+loss_var_small+loss_var_huge+loss_m)


            ###refine***********************

            depth_sub_plus=stage_inputs["depth_sub_plus_refine"]
            depth_sup_plus_small,depth_sup_plus_huge=depth_sub_plus.split([2,2],dim=1)
            loss_depth=2*regression_loss(depth_sup_plus_small, depth_gt.unsqueeze(1).expand_as(depth_sup_plus_small), mask.unsqueeze(1).expand_as(depth_sup_plus_small),torch.ones_like(depth_sup_plus_small)*stage_weight)\
                        +2*regression_loss(depth_sup_plus_huge, depth_gt.unsqueeze(1).expand_as(depth_sup_plus_huge), mask.unsqueeze(1).expand_as(depth_sup_plus_huge),torch.ones_like(depth_sup_plus_huge)*stage_weight)
            
            var_gt=torch.where((depth_sub_plus[:,0]-depth_gt).abs()<(depth_sub_plus[:,1]-depth_gt).abs(),(depth_sub_plus[:,1]-depth_gt).abs(),(depth_sub_plus[:,0]-depth_gt).abs())
            loss_var_small=regression_loss((depth_sub_plus[:,0]-depth_sub_plus[:,1]).abs(), var_gt, mask,torch.ones_like(var_gt)*stage_weight)

            var_gt=torch.where((depth_sub_plus[:,2]-depth_gt).abs()<(depth_sub_plus[:,3]-depth_gt).abs(),(depth_sub_plus[:,3]-depth_gt).abs(),(depth_sub_plus[:,2]-depth_gt).abs())
            loss_var_huge=regression_loss((depth_sub_plus[:,2]-depth_sub_plus[:,3]).abs(), var_gt, mask,torch.ones_like(var_gt)*stage_weight)
            
            
            coors=torch.stack( 
            [item.unsqueeze(0).expand_as(depth_sub_plus[:,0]) for item in torch.meshgrid(*[torch.arange(0, s) for s in depth_sub_plus[:,0].shape[-2:]])],
            axis=-1).to(depth_sub_plus[:,0].device)
            coor_mask=((coors[:,:,:,0]%2==0)&(coors[:,:,:,1]%2==0))|((coors[:,:,:,0]%2==1)&(coors[:,:,:,1]%2==1))# 
            
            small_min,small_max=depth_sup_plus_small.min(1)[0],depth_sup_plus_small.max(1)[0]
            huge_min,huge_max=depth_sup_plus_huge.min(1)[0],depth_sup_plus_huge.max(1)[0]
        
            loss_m=Monte_Carlo_sampling_loss(torch.where(coor_mask,small_min,small_max),depth_gt,mask,torch.ones_like(depth_gt)*stage_weight,mode="center",regress_fn=regression_loss)+\
                Monte_Carlo_sampling_loss(torch.where(~coor_mask,small_min,small_max),depth_gt,mask,torch.ones_like(depth_gt)*stage_weight,mode="center",regress_fn=regression_loss)+\
                Monte_Carlo_sampling_loss(torch.where(coor_mask,huge_min,huge_max),depth_gt,mask,torch.ones_like(depth_gt)*stage_weight,mode="center",regress_fn=regression_loss)+\
                Monte_Carlo_sampling_loss(torch.where(~coor_mask,huge_min,huge_max),depth_gt,mask,torch.ones_like(depth_gt)*stage_weight,mode="center",regress_fn=regression_loss)


            total_loss+=(loss_depth+loss_var_small+loss_var_huge+loss_m)

        elif mode == "classification":
            # loss = classification_loss(prob_volume, depth_values, interval, depth_gt, mask, stage_weight)
            loss = classification_loss_1(prob_volume, depth_values, interval, depth_gt, mask, stage_weight)

            total_loss += loss
        elif mode =="gfocal":
            fl_gamas = [2, 1, 0]
            fl_alphas = [0.75, 0.5, 0.25]
            gamma = fl_gamas[stage_idx]
            alpha = fl_alphas[stage_idx]
            loss = gfocal_loss(prob_volume, depth_values, interval, depth_gt, mask, stage_weight, gamma, alpha)
            total_loss += loss
        elif mode == "unification":
            fl_gamas = [2, 1, 0]
            fl_alphas = [0.75, 0.5, 0.25]
            gamma = fl_gamas[stage_idx]
            alpha = fl_alphas[stage_idx]
            loss = unified_focal_loss(prob_volume, depth_values, interval, depth_gt, mask, stage_weight, gamma, alpha)
            total_loss += loss
        else:
            raise NotImplementedError("Only support regression, classification and unification!")

    return total_loss

def Monte_Carlo_sampling_loss(depth_est, depth_gt, mask, weight,mode="center",reflect=False,regress_fn=None):
    
    batch,height, width= depth_gt.shape

    if mode=="center":
        x_offset,y_offset=0.5*torch.ones((batch,height-1, width-1)),0.5*torch.ones((batch,height-1, width-1))
    else:
        x_offset,y_offset=torch.rand(batch,height-1, width-1),torch.rand((batch,height-1, width-1))
    
    x_offset,y_offset=x_offset.to(depth_gt.device),y_offset.to(depth_gt.device)

    y, x = torch.meshgrid([torch.arange(0, height-1, dtype=torch.float32, device=depth_gt.device),
                        torch.arange(0, width-1, dtype=torch.float32, device=depth_gt.device)])
    y, x = y.contiguous().unsqueeze(0).repeat(batch,1,1)+y_offset, x.contiguous().unsqueeze(0).repeat(batch,1,1)+x_offset
    x=x/((width - 1) / 2) - 1
    y=y/((height - 1) / 2) - 1

    grid=torch.stack((x, y), dim=3)

    sampled_gt=F.grid_sample(depth_gt.unsqueeze(1), grid, mode='bilinear',padding_mode='zeros',align_corners=True).type(torch.float32)
    sampled_est=F.grid_sample(depth_est.unsqueeze(1), grid, mode='bilinear',padding_mode='zeros',align_corners=True).type(torch.float32)
    sampled_weight=F.grid_sample(weight.unsqueeze(1), grid, mode='bilinear',padding_mode='zeros',align_corners=True).type(torch.float32)
    sampled_mask=F.grid_sample(mask.float().unsqueeze(1), grid, mode='bilinear',padding_mode='zeros',align_corners=True).type(torch.float32)
    #mask!=1 mean there is zero depth\
    sampled_mask=sampled_mask>=1.
    
    
    if reflect== False:
        # loss = F.smooth_l1_loss(sampled_est[sampled_mask], sampled_gt[sampled_mask], reduction='mean')
        loss =regress_fn(sampled_est, sampled_gt, sampled_mask,sampled_weight)
        
    else:
        with torch.no_grad():
            err=depth_est-depth_gt   
            kernel = torch.ones((2,2)).unsqueeze(0).unsqueeze(0).to(depth_gt.device)
            kernel_weight = torch.nn.Parameter(data=kernel, requires_grad=False)
            
            up_sum=F.conv2d((err.unsqueeze(1)>0).float(),kernel_weight)
            dn_sum=F.conv2d((err.unsqueeze(1)<0).float(),kernel_weight)
            
            reflect_weight=torch.where((up_sum==4.)|(dn_sum==4.),2*torch.ones_like(sampled_gt),torch.ones_like(sampled_gt))
            # reflect_weight=reflect_weight[sampled_mask]

        loss = F.smooth_l1_loss((reflect_weight*sampled_est)[sampled_mask], (reflect_weight*sampled_gt)[sampled_mask], reduction='mean')

    # loss = loss* weight

    
    
    return loss
def regression_loss(depth_est, depth_gt, mask, weight):
    loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='none')
    loss = (loss* weight[mask]).mean()
    return loss

def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None,
                                     reduce=False, reduction='elementwise_mean', pos_weight=None,mask=None):

    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)

    if pos_weight is None:
        ce_loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    else:
        log_weight = 1 + (pos_weight - 1) * target
        ce_loss = input - input * target + log_weight * (max_val + ((-max_val).exp() + (-input - max_val).exp()).log())

            
    if weight is not None:
        ce_loss = ce_loss * weight
    if mask is not None:
        
        ce_loss = ce_loss[mask.unsqueeze(1).repeat(1,ce_loss.shape[1],1,1)]

    if reduction == False:
        return ce_loss
    elif reduction == 'elementwise_mean':
        return ce_loss.mean()
    else:
        return ce_loss.sum()
def classification_loss_1(prob_volume, depth_values, interval, depth_gt, mask, weight):
    depth_gt_volume = depth_gt.unsqueeze(1).expand_as(depth_values)  # (b, d, h, w)

    gt_index_volume = (
            ((depth_values - interval / 2) <= depth_gt_volume).float() * ((depth_values + interval / 2) > depth_gt_volume).float())

    pos_w = (depth_gt_volume.shape[1]-1)/1.0 # pos_w = neg_num / pos_num
    loss = binary_cross_entropy_with_logits(prob_volume, gt_index_volume, pos_weight=pos_w,mask=mask,weight=weight)
    return loss

def classification_loss(prob_volume, depth_values, interval, depth_gt, mask, weight):
    depth_gt_volume = depth_gt.unsqueeze(1).expand_as(depth_values)  # (b, d, h, w)

    gt_index_volume = (
            ((depth_values - interval / 2) <= depth_gt_volume).float() * ((depth_values + interval / 2) > depth_gt_volume).float())

    NEAR_0 = 1e-4  # Prevent overflow
    prob_volume = torch.where(prob_volume <= 0.0, torch.zeros_like(prob_volume) + NEAR_0, prob_volume)

    loss = -torch.sum(gt_index_volume * torch.log(prob_volume), dim=1)[mask].mean()
    loss = loss * weight
    return loss


def gfocal_loss(prob_volume, depth_values, interval, depth_gt, mask, weight, gamma, alpha):
    depth_gt_volume = depth_gt.unsqueeze(1).expand_as(depth_values)  # (b, d, h, w)

    gt_index_volume = ((depth_values <= depth_gt_volume) * ((depth_values + interval) > depth_gt_volume)) #gt 在哪一个value里面而已
    gt_index_volume=gt_index_volume.float()

    pos_weight = (gt_index_volume - prob_volume).abs() 
    neg_weight = prob_volume
    focal_weight = (pos_weight.pow(gamma)) * (gt_index_volume > 0.0).float()\
               + alpha*(neg_weight.pow(gamma)) * (gt_index_volume <= 0.0).float()
    
    NEAR_0 = 1e-4  # Prevent overflow
    prob_volume = torch.where(prob_volume <= 0.0, torch.zeros_like(prob_volume) + NEAR_0, prob_volume)
    
    mask = mask.unsqueeze(1).expand_as(depth_values).float() # b d h w 
    loss = (F.binary_cross_entropy(prob_volume, gt_index_volume, reduction="none") * focal_weight * mask).sum() / mask.sum() # all
    loss = loss * weight
    return loss

def unified_step_focal_loss(prob_volume, depth_values, interval, depth_gt, mask, weight, gamma, alpha):
    depth_gt_volume = depth_gt.unsqueeze(1).expand_as(depth_values)  # (b, d, h, w)

    gt_index_volume = (depth_values-depth_gt_volume).abs()<=interval

    gt_unity_index_volume = torch.zeros_like(prob_volume, requires_grad=False)
    gt_unity_index_volume[gt_index_volume] = 1.0 - (depth_gt_volume[gt_index_volume] - depth_values[gt_index_volume]).abs() / interval

    gt_unity, _ = torch.max(gt_unity_index_volume, dim=1, keepdim=True)
    gt_unity = torch.where(gt_unity > 0.0, gt_unity, torch.ones_like(gt_unity))  # (b, 1, h, w)
    pos_weight = (sigmoid((gt_unity - prob_volume).abs() / gt_unity, base=5) - 0.5) * 4 + 1  # [1, 3]
    neg_weight = (sigmoid(prob_volume / gt_unity, base=5) - 0.5) * 2  # [0, 1]
    focal_weight =  (gt_unity_index_volume > 0.0).float() + alpha  * (gt_unity_index_volume <= 0.0).float()

    mask = mask.unsqueeze(1).expand_as(depth_values).float()
    # offset=prob_volume-1
    # torch.where
    prob_volume=prob_volume/(prob_volume.max())
    loss = (F.binary_cross_entropy(prob_volume, gt_unity_index_volume, reduction="none") * focal_weight * mask).sum() / mask.sum()
    loss = loss * weight
    return loss
def unified_focal_loss(prob_volume, depth_values, interval, depth_gt, mask, weight, gamma, alpha):
    depth_gt_volume = depth_gt.unsqueeze(1).expand_as(depth_values)  # (b, d, h, w)

    gt_index_volume = ((depth_values <= depth_gt_volume) * ((depth_values + interval) > depth_gt_volume))

    gt_unity_index_volume = torch.zeros_like(prob_volume, requires_grad=False)
    gt_unity_index_volume[gt_index_volume] = 1.0 - (depth_gt_volume[gt_index_volume] - depth_values[gt_index_volume]) / interval

    gt_unity, _ = torch.max(gt_unity_index_volume, dim=1, keepdim=True)
    gt_unity = torch.where(gt_unity > 0.0, gt_unity, torch.ones_like(gt_unity))  # (b, 1, h, w)
    pos_weight = (sigmoid((gt_unity - prob_volume).abs() / gt_unity, base=5) - 0.5) * 4 + 1  # [1, 3]
    neg_weight = (sigmoid(prob_volume / gt_unity, base=5) - 0.5) * 2  # [0, 1]
    focal_weight = pos_weight.pow(gamma) * (gt_unity_index_volume > 0.0).float() + alpha * neg_weight.pow(gamma) * (
            gt_unity_index_volume <= 0.0).float()

    mask = mask.unsqueeze(1).expand_as(depth_values).float()
    loss = (F.binary_cross_entropy(prob_volume, gt_unity_index_volume, reduction="none") * focal_weight * mask).sum() / mask.sum()
    loss = loss * weight
    return loss
def sigmoid(x, base=2.71828):
    return 1 / (1 + torch.pow(base, -x))
def entropy_loss(prob_volume, depth_gt, mask, depth_value, return_prob_map=False):
    # from AA
    mask_true = mask
    valid_pixel_num = torch.sum(mask_true, dim=[1,2]) + 1e-6

    shape = depth_gt.shape          # B,H,W

    depth_num = depth_value.shape[1]
    if len(depth_value.shape) < 3:
        depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2,3,0,1)     # B,N,H,W
    else:
        depth_value_mat = depth_value

    gt_index_image = torch.argmin(torch.abs(depth_value_mat-depth_gt.unsqueeze(1)), dim=1)
    temp=gt_index_image

    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1) # B, 1, H, W

    # gt index map -> gt one hot volume (B x 1 x H x W )
    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1, gt_index_image, 1)

    # cross entropy image (B x D X H x W)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume + 1e-6), dim=1).squeeze(1) # B, 1, H, W

    # masked cross entropy loss
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image) # valid pixel
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])

    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num) # Origin use sum : aggregate with batch
    # winner-take-all depth map
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long).squeeze(1)

    return masked_cross_entropy

def entropy_loss_expand(prob_volume, depth_gt, mask, depth_value, return_prob_map=False):
    # from AA
    
    
    shape = depth_gt.shape          # B,H,W
    depth_gt=depth_gt.unsqueeze(1).repeat(1,3,1,1).view(-1,shape[-2],shape[-1])
    mask=mask.unsqueeze(1).repeat(1,3,1,1).view(-1,shape[-2],shape[-1])
    shape = depth_gt.shape
    
    mask_true = mask
    valid_pixel_num = torch.sum(mask_true, dim=[1,2]) + 1e-6

    depth_num = depth_value.shape[1]
    if len(depth_value.shape) < 3:
        depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2,3,0,1)     # B,N,H,W
    else:
        depth_value_mat = depth_value

    gt_index_image = torch.argmin(torch.abs(depth_value_mat-depth_gt.unsqueeze(1)), dim=1)
    temp=gt_index_image

    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1) # B, 1, H, W

    # gt index map -> gt one hot volume (B x 1 x H x W )
    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1, gt_index_image, 1)

    # cross entropy image (B x D X H x W)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume + 1e-6), dim=1).squeeze(1) # B, 1, H, W

    # masked cross entropy loss
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image) # valid pixel
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])

    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num) # Origin use sum : aggregate with batch
    # winner-take-all depth map
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long).squeeze(1)

    return masked_cross_entropy
