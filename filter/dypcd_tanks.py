import os
import cv2
import signal
import numpy as np
from PIL import Image
from functools import partial
from multiprocessing import Pool
from plyfile import PlyData, PlyElement

from datasets.data_io import read_pfm
from filter.tank_test_config import tank_cfg

from datasets.data_io import save_pfm, read_pfm

import torch.nn.functional as F
import torch
# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] /= 4
    return intrinsics, extrinsics


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    K_xyz_reprojected[2:3][K_xyz_reprojected[2:3]==0] += 0.00001
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src

@torch.no_grad()
def reproject_with_depth_pytorch(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    def tocuda(varlist:list):
        out=[]
        for var in varlist:
            if isinstance(var,np.ndarray):
                var=torch.from_numpy(var.copy())
            out.append(var.cuda())
        return out
    def tonumpy(varlist:list):
        out=[]
        for var in varlist:
            out.append(var.cpu().numpy())
        return out

    [depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src]=tocuda([depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src])

    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    #np.meshgrid(a,b)=torch.meshgrid(b,a)
    y_ref,x_ref = torch.meshgrid(torch.arange(0, height),torch.arange(0, width))
    
    x_ref, y_ref = x_ref.reshape([-1]).cuda(), y_ref.reshape([-1]).cuda()
    # reference 3D space
    xyz_ref = torch.matmul(torch.linalg.inv(intrinsics_ref),
                        torch.vstack((x_ref, y_ref, torch.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = torch.matmul(torch.matmul(extrinsics_src, torch.linalg.inv(extrinsics_ref)),
                        torch.vstack((xyz_ref, torch.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0]/ ((width - 1) / 2) - 1
    y_src = xy_src[1]/ ((height - 1) / 2) - 1
    proj_xy = torch.stack((x_src, y_src), dim=-1)  # [H*W, 2]
    sampled_depth_src = F.grid_sample(depth_src.unsqueeze(0).unsqueeze(0), proj_xy.view(1, height, width, 2), mode='bilinear',padding_mode='zeros',align_corners=True).type(torch.float32).squeeze(0).squeeze(0)

   
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = torch.matmul(torch.linalg.inv(intrinsics_src),
                        torch.vstack((xy_src, torch.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = torch.matmul(torch.matmul(extrinsics_ref, torch.linalg.inv(extrinsics_src)),
                                torch.vstack((xyz_src, torch.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width])
    K_xyz_reprojected = torch.matmul(intrinsics_ref, xyz_reprojected)
    K_xyz_reprojected[2:3][K_xyz_reprojected[2:3]==0] += 0.00001
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width])
    y_reprojected = xy_reprojected[1].reshape([height, width])

    [depth_reprojected, x_reprojected, y_reprojected, x_src, y_src]=tonumpy([depth_reprojected, x_reprojected, y_reprojected, x_src, y_src])
    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(args, depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                                                                 depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = None
    masks = []
    for i in range(2, 11):
        # mask = np.logical_and(dist < i / 4, relative_depth_diff < i / 1300)
        mask = np.logical_and(dist < i * args.dist_base, relative_depth_diff < i * args.rel_diff_base)
        masks.append(mask)
    depth_reprojected[~mask] = 0

    return masks, mask, depth_reprojected, x2d_src, y2d_src

def filter_depth(args, pair_folder, scan_folder, out_folder, plyfilename):
    num_stage = len(args.ndepths)

    # the pair file
    pair_file = os.path.join(pair_folder, "pair.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        # src_views = src_views[:args.num_view]
        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        # load the reference image
        ref_img = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)))
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        # load the photometric mask of the reference view
        confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]

        if not os.path.exists(os.path.join(out_folder, 'confidence/{:0>8}_stage2.pfm'.format(ref_view))):
            confidence2=confidence1=confidence
        else:
            confidence2 = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}_stage2.pfm'.format(ref_view)))[0]
            confidence1 = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}_stage1.pfm'.format(ref_view)))[0]
        photo_mask = np.logical_and(np.logical_and(confidence > args.conf[2], confidence2 > args.conf[1]), confidence1 > args.conf[0])

        # save_pfm(depth_filename, depth_est)


        if not (os.path.exists(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view))) and os.path.exists(os.path.join(out_folder, 'depth_est/{:0>8}_averaged.pfm'.format(ref_view)))):
            all_srcview_depth_ests = []
            # all_srcview_x = []
            # all_srcview_y = []
            all_srcview_geomask = []

            # compute the geometric mask
            geo_mask_sum = 0
            dy_range = len(src_views) + 1
            geo_mask_sums = [0] * (dy_range - 2)
            for src_view in src_views:
                # camera parameters of the source view
                src_intrinsics, src_extrinsics = read_camera_parameters(
                    os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
                # the estimated depth of the source view
                src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]

                masks, geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(args, ref_depth_est, ref_intrinsics,
                                                                                                ref_extrinsics, src_depth_est,
                                                                                                src_intrinsics, src_extrinsics)
                geo_mask_sum += geo_mask.astype(np.int32)
                for i in range(2, dy_range):
                    geo_mask_sums[i - 2] += masks[i - 2].astype(np.int32)

                all_srcview_depth_ests.append(depth_reprojected)
                all_srcview_geomask.append(geo_mask)

            depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)

            save_pfm(os.path.join(out_folder, 'depth_est/{:0>8}_averaged.pfm'.format(ref_view)), depth_est_averaged.astype(np.float32))
            # at least args.thres_view source views matched

            geo_mask = geo_mask_sum >= dy_range
            for i in range(2, dy_range):
                geo_mask = np.logical_or(geo_mask, geo_mask_sums[i - 2] >= i)
        else:
            geo_mask_path=os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view))
            import cv2
            geo_mask=cv2.imread(geo_mask_path,-1)>0
            depth_est_averaged=read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}_averaged.pfm'.format(ref_view)))[0]
            print("finished")

        final_mask = np.logical_and(photo_mask, geo_mask)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

        print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(scan_folder, ref_view,
                                                                                    photo_mask.mean(),
                                                                                    geo_mask.mean(), final_mask.mean()))

        if args.display:
            import cv2
            cv2.imshow('ref_img', ref_img[:, :, ::-1])
            cv2.imshow('ref_depth', ref_depth_est / 800)
            cv2.imshow('ref_depth * photo_mask', ref_depth_est * photo_mask.astype(np.float32) / 800)
            cv2.imshow('ref_depth * geo_mask', ref_depth_est * geo_mask.astype(np.float32) / 800)
            cv2.imshow('ref_depth * mask', ref_depth_est * final_mask.astype(np.float32) / 800)
            cv2.waitKey(0)

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))

        valid_points = final_mask
        print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        # color = ref_img[1:-16:4, 1::4, :][valid_points]  # hardcoded for DTU dataset

        if num_stage == 1:
            color = ref_img[1::4, 1::4, :][valid_points]
        elif num_stage == 2:
            color = ref_img[1::2, 1::2, :][valid_points]
        elif num_stage == 3:
            color = ref_img[valid_points]

        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

        # # set used_mask[ref_view]
        # used_mask[ref_view][...] = True
        # for idx, src_view in enumerate(src_views):
        #     src_mask = np.logical_and(final_mask, all_srcview_geomask[idx])
        #     src_y = all_srcview_y[idx].astype(np.int)
        #     src_x = all_srcview_x[idx].astype(np.int)
        #     used_mask[src_view][src_y[src_mask], src_x[src_mask]] = True

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)
    
def check_geometric_consistency_geomean(args, depth_ref, intrinsics_ref, extrinsics_ref, depth_src_up,depth_src,depth_src_dn, intrinsics_src, extrinsics_src,f=2):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected_u, x2d_reprojected_u, y2d_reprojected_u, x2d_src_u, y2d_src_u = reproject_with_depth_pytorch(depth_ref, intrinsics_ref, extrinsics_ref,
                                                                                                 depth_src_up, intrinsics_src, extrinsics_src)
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth_pytorch(depth_ref, intrinsics_ref, extrinsics_ref,
                                                                                                 depth_src, intrinsics_src, extrinsics_src)
    depth_reprojected_d, x2d_reprojected_d, y2d_reprojected_d, x2d_src_d, y2d_src_d = reproject_with_depth_pytorch(depth_ref, intrinsics_ref, extrinsics_ref,
                                                                                                 depth_src_dn, intrinsics_src, extrinsics_src)
    
    
    depth_reprojected=np.stack((depth_reprojected_u,depth_reprojected,depth_reprojected_d),axis=0)
    x2d_reprojected=np.stack((x2d_reprojected_u,x2d_reprojected,x2d_reprojected_d),axis=0)
    y2d_reprojected=np.stack((y2d_reprojected_u,y2d_reprojected,y2d_reprojected_d),axis=0)
    
    dist = np.sqrt((x2d_reprojected - x_ref[np.newaxis]) ** 2 + (y2d_reprojected - y_ref[np.newaxis]) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref[np.newaxis])
    relative_depth_diff = depth_diff / depth_ref

    mask = None
    masks = []
    for i in range(f, 6):
        # mask = np.logical_and(dist < i / 4, relative_depth_diff < i / 1300)
        mask = np.logical_and(dist < i * args.dist_base, relative_depth_diff < i * args.rel_diff_base)
        mask_ = np.logical_and(dist < (i) * args.dist_base, relative_depth_diff < (i) * args.rel_diff_base)
        mask_=mask_.mean(0)
        
        masks.append(mask_)
    div_=np.where(mask.sum(0)>0,(mask.sum(0)),np.ones_like(depth_reprojected[0]))
    depth_reprojected[~mask]=0
    depth_reprojected=np.where(mask.sum(0)>0,(depth_reprojected.sum(0))/div_,np.zeros_like(depth_reprojected[0]))
    
    mask_for_view=mask.mean(0)
    mask=mask.sum(0)>0
    

    return masks, mask,mask_for_view, depth_reprojected, x2d_src, y2d_src


def dypcd_filter_worker(args, scene,suffix=None):
    if args.testlist != "all":
        scan_id = int(scene[4:])
        save_name = 'mvsnet{:0>3}_l3.ply'.format(scan_id)
    else:
        save_name = '{}.ply'.format(scene)
    pair_folder = os.path.join(args.datapath, scene)
    scan_folder = os.path.join(args.outdir, scene)
    out_folder = os.path.join(args.outdir, scene)

    if scene in tank_cfg.scenes:
        scene_cfg = getattr(tank_cfg, scene)
        args.conf = scene_cfg.conf
        
    filter_depth(args, pair_folder, scan_folder, out_folder, os.path.join(args.outdir,"dypcd" ,save_name))
    
def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def dypcd_filter(args, testlist, number_worker,suffix=None):
    if not os.path.exists(os.path.join(args.outdir,"dypcd")):
        os.makedirs(os.path.join(args.outdir,"dypcd"))
    if number_worker>1:
        partial_func = partial(dypcd_filter_worker, args)

        p = Pool(number_worker, init_worker)
        try:
            p.map(partial_func, testlist)
        except KeyboardInterrupt:
            print("....\n Caught KeyboardInterrupt, terminating workers")
            p.terminate()
        else:
            p.close()
        p.join()
    else:
        if suffix is not None:
            if not os.path.exists(os.path.join(args.outdir,"dypcd_{}".format(suffix))):
                os.makedirs(os.path.join(args.outdir,"dypcd_{}".format(suffix)))

        for scene in testlist:
            dypcd_filter_worker(args,scene,suffix=suffix)
