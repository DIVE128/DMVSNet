# (ICCV 2023) Constraining Depth Map Geometry for Multi-View Stereo: A Dual-Depth Approach with Saddle-shaped Depth Cells

- Xinyi Ye, Weiyue Zhao, Tianqi Liu, Zihao Huang, Zhiguo Cao, Xin Li

## [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Ye_Constraining_Depth_Map_Geometry_for_Multi-View_Stereo_A_Dual-Depth_Approach_ICCV_2023_paper.html) |  Project Page |[Arxiv](https://arxiv.org/abs/2307.09160) | [Model](https://pan.baidu.com/s/1sw4lkIzoOymBJNFp622iVA) | [Points](https://pan.baidu.com/s/1bos9KatNs7WlrE3-JbNvqA )
![image](assets/pipeline.png)

# Highlights
![image](assets/motivation.png)

In this work,**we propose a fresh viewpoint for considering depth geometry in multi-view stereo, a factor that has not been adequately concerned in prior works**. We demonstrated that different depth geometries suffer from significant performance gaps, even for the same depth estimation error case in the MVS reconstruction task both qualitatively and quantitatively. Based on the concept, we proposed the depth geometry with saddle-shaped cells  for the first time and a dual-depth approach to constraint depth map to approach the proposed geometry.

# Abstract

Learning-based multi-view stereo (MVS) methods deal with predicting accurate depth maps to achieve an accurate and complete 3D representation. Despite the excellent performance, existing methods ignore the fact that a suitable depth geometry is also critical in MVS. In this paper, we demonstrate that different depth geometries have significant performance gaps, even using the same depth prediction error. Therefore, we introduce an ideal depth geometry composed of **Saddle-Shaped Cell**s, whose predicted depth map oscillates upward and downward around the ground-truth surface, rather than maintaining a continuous and smooth depth plane. To achieve it, we develop a coarse-to-fine framework called Dual-MVSNet (DMVSNet),  which can produce an oscillating depth plane. Technically, we predict two depth values for each pixel (**Dual-Depth**), and propose a novel loss function and a checkerboard-shaped selecting strategy to constrain the predicted depth geometry. Compared to existing methods,DMVSNet achieves a high rank on the DTU benchmark and obtains the top performance on challenging scenes of Tanks and Temples, demonstrating its strong performance and generalization ability. Our method also points to a new research direction for considering depth geometry in MVS.

# Prepare Data
#### 1. DTU Dataset

**Training Data**. We adopt the full resolution ground-truth depth provided in CasMVSNet or MVSNet. Download [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) and [Depth raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip). 
Unzip them and put the `Depth_raw` to `dtu_training` folder. The structure is just like:
```
dtu_training                          
       ├── Cameras                
       ├── Depths   
       ├── Depths_raw
       └── Rectified
```
**Testing Data**. Download [DTU testing data](https://drive.google.com/file/d/135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_/view) and unzip it. The structure is just like:
```
dtu_testing                          
       ├── Cameras                
       ├── scan1   
       ├── scan2
       ├── ...
```

#### 2. BlendedMVS Dataset

**Training Data** and **Validation Data**. Download [BlendedMVS](https://drive.google.com/file/d/1ilxls-VJNvJnB7IaFj7P0ehMPr7ikRCb/view) and 
unzip it. And we only adopt 
BlendedMVS for finetuning and not testing on it. The structure is just like:
```
blendedmvs                          
       ├── 5a0271884e62597cdee0d0eb                
       ├── 5a3ca9cb270f0e3f14d0eddb   
       ├── ...
       ├── training_list.txt
       ├── ...
```

#### 3. Tanks and Temples Dataset

**Testing Data**. Download [Tanks and Temples](https://drive.google.com/file/d/1YArOJaX9WVLJh4757uE8AEREYkgszrCo/view) and 
unzip it. Here, we adopt the camera parameters of short depth range version (Included in your download), therefore, **you should 
replace the `cams` folder in `intermediate` folder with the short depth range version manually.** The 
structure is just like:
```
tanksandtemples                          
       ├── advanced                 
       │   ├── Auditorium       
       │   ├── ...  
       └── intermediate
           ├── Family       
           ├── ...          
```
# Environment
- PyTorch 1.8.1
- Python 3.7
- progressbar 2.5
- thop 0.1

# Scripts
#### 1. train on DTU
- modify `datapath` in `scripts/train.sh`
```bash
bash scripts/train.sh
```
#### 2. evaluate on DTU
- modify `datapath` and `resume` in `scripts/dtu_test.sh`
```bash
bash scripts/dtu_test.sh
```
- modify `datapath`, `plyPath`, `resultsPath` in `scripts/evaluation_dtu/BaseEvalMain_web.m`
- modify `datapath`, `resultsPath` in `scripts/evaluation_dtu/ComputeStat_web.m`
```
cd ./scripts/evaluation_dtu/
matlab -nodisplay

BaseEvalMain_web 

ComputeStat_web
```
#### 3. finetune on BlendedMVS
- modify `datapath` and `resume` in `scripts/blendedmvs_finetune.sh`
```bash
bash scripts/blendedmvs_finetune.sh
```

#### 4. evaluate on Tanks and Temple
- modify `datapath` and `resume` in `scripts/dtu_test.sh`

#### 5 points and model
- [Points](https://pan.baidu.com/s/1bos9KatNs7WlrE3-JbNvqA) (extraction code: 2ygz)
- [Model](https://pan.baidu.com/s/1sw4lkIzoOymBJNFp622iVA)(extraction code: 8lly)
# Citation
```bibtex
@inproceedings{Ye2023Dmvsnet,
  title={Constraining Depth Map Geometry for Multi-View Stereo: A Dual-Depth Approach with Saddle-shaped Depth Cells},
  author={Xinyi Ye, Weiyue Zhao, Tianqi Liu, Zihao Huang, Zhiguo Cao, Xin Li},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```

# Acknowledge
We have incorporated certain release codes from [Unimvsnet](https://github.com/prstrive/UniMVSNet) and extend our gratitude for their excellent work