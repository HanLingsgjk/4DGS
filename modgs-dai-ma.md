---
title: 'MODGS代码'
date: 2025-04-22 15:49:13
tags: []
published: true
hideInList: false
feature: 
isTop: false
---
# NVPSimplified
这篇代码里用到NVPSimplified，它的每一层 CouplingLayer 都是设计上就保证可逆的：它对输入的某一部分变量保持不变，对另一部分变量进行 仿射变换（scale & shift），并且这种变换可以 显式求反函数。三个参数分别是时间，在本代码中利用神经网络GaborNet生成的时间特征和点云位置。
思想：把每帧里动来动去的点都对齐回 canonical 状态
## **forward**
```python
    def forward(self, t, feat, x):
        y = x
        if self.affine:
            y = self._affine_input(t, y)
        for i in self.layer_idx:
            feat_i = self.code_projectors[i](feat)
            feat_i = self._expand_features(feat_i, y)
            l1 = self.layers1[i]
            y, _ = self._call(l1, feat_i, y)
        return y
```
###  **y = self._affine_input(t, y)**
```python
def _affine_input(self, t, x, inverse=False):
    depth = x[..., -1]  # 取出 z 坐标（depth）[B, N]
    net_in = torch.stack([t.expand_as(depth), depth], dim=-1)  #组合成 net_in ∈ [B, N, 2]，表示每个点的 (time, depth) 信息
    affine = self.get_affine(self.affine_mlp(net_in), inverse=inverse)  # [B, N, 3, 3]
    '''
        self.affine_mlp = pe_relu.MLP(input_dim=2,
                                                   hidden_size=256,
                                                   n_layers=2,
                                                   skip_layers=[],
                                                   use_pe=True,
                                                   pe_dims=[1],
                                                   pe_freq=pe_freq,
                                                   output_dim=5).to(device)
        '''
    xy = x[..., :2]  # 点云xy坐标[B, N, 2]
    xy = apply_homography(affine, xy)  #将仿射矩阵应用到 x, y 坐标上 [B, N, 2]
    x = torch.cat([xy, depth.unsqueeze(-1)], dim=-1)  # 与原始的 z（depth）拼接[B, N, 3]
    return x
```
其中：
```python
affine = self.get_affine(self.affine_mlp(net_in), inverse=inverse)  # [B, N, 3, 3]
```
self.affine_mlp从每个点的 [时间t, 深度z] 输入中预测出该点的仿射变换参数 [angle, scale1, tx, scale2, ty]，self.get_affine从而构造点的仿射矩阵。
```python
[ a  b  tx ]
[ c  d  ty ]
[ 0  0   1 ]
a = cos(angle) * scale1
b = -sin(angle) * scale1
c = sin(angle) * scale2
d = cos(angle) * scale2
```
```python
    def get_affine(self, theta, inverse=False):
        """
        expands the 5 parameters into 3x3 affine transformation matrix
        :param theta (..., 5)
        :returns mat (..., 3, 3)
        """
        angle = theta[..., 0:1]
        scale1 = torch.exp(theta[..., 1:2])
        scale2 = torch.exp(theta[..., 3:4])
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        a = cos * scale1
        b = -sin * scale1
        c = sin * scale2
        d = cos * scale2
        tx = theta[..., 2:3]
        ty = theta[..., 4:5]
        zeros = torch.zeros_like(a)
        ones = torch.ones_like(a)
        if inverse:
            return self.invert_affine(a, b, c, d, tx, ty, zeros, ones)
        else:
            return torch.cat([a, b, tx, c, d, ty, zeros, zeros, ones], dim=-1).reshape(*theta.shape[:-1], 3, 3)
```
### **for i in self.layer_idx:**
layer_idx=4，一共4次
#### **先一层MLP**
先用MLP变换一次时间特征[B,128]->[B,128]
```python
        feat_i = self.code_projectors[i](feat)       # 投影特征
        '''
        self.code_projectors.append(
                MLP(
                    feature_dims,
                    feature_dims,
                    code_proj_hidden_size,
                    bn=normalization,
                    act=activation,
                )
            )
        '''
```
```python
        feat_i = self._expand_features(feat_i, y)     # 将 [B, 128] 扩展为 [B, N, 128]
        '''
        def _expand_features(self, F, x):
    B, N, _ = x.shape               # 获取点云的 batch size 和点数
    return F.unsqueeze(1).expand(B, N, -1)
    '''
```
#### **一层CouplingLayer（耦合层）**
```python
        l1 = self.layers1[i]
```
layers1：
```python
        input_dims = 3
        i = 0
        mask_selection = []
        while i < n_layers:
            mask_selection.append(torch.randperm(input_dims))
            i += input_dims
        mask_selection = torch.cat(mask_selection) ## 2024年3月26日10:58:20：【0,1,2,0,2,1】'240326_105658'
        # 每一层都需要一个随机的掩码（mask），指定输入的哪些维度参与变换。

        for i in self.layer_idx:
            # get mask
            mask2 = torch.zeros(input_dims, device=device)
            mask2[mask_selection[i]] = 1 #mask2 表示：哪些维度将被更新
            mask1 = 1 - mask2  # mask1 表示：哪些维度保持不变


            # get transformation
            map_st = nn.Sequential(
                MLP(
                    proj_dims + feature_dims,
                    2,
                    hidden_size,
                    bn=normalization,
                    act=activation,
                    )
            )# 构造变换网络（map_st），一个小型 MLP，用于预测仿射变换参数 s 和 t

            proj = get_projection_layer(proj_dims=proj_dims, type=proj_type, pe_freq=pe_freq)# 构造投影层
            # self.layers1.append(CouplingLayer(map_st, proj, mask1[None, None, None]))
            self.layers1.append(CouplingLayer(map_st, proj, mask1[ None, None])) # mask1[ None, None]给 mask1 的前面加两个维度
```
```python
class CouplingLayer(nn.Module):
    def __init__(self, map_st, projection, mask):
        super().__init__()
        self.map_st = map_st
        self.projection = projection
        # self.mask = mask ## 
        self.register_buffer("mask", mask)

    def forward(self, F, y):
        y1 = y * self.mask# 举个例子，xyz中是y维度被更新，这里把 y 的部分维度屏蔽掉（乘 0）
        F_y1 = torch.cat([F, self.projection(y[..., self.mask.squeeze().bool()])], dim=-1)#这里把没有被屏蔽的[x,z]拿出来，维度为[B,N,2]，经过一个带 positional encoding 的小 MLP 映射变成[B,N,256]，再与前面得到的时间特征[B,N,128]拼接，变成[B,N,384]
        st = self.map_st(F_y1)# 用 map_st（一个小的 MLP）对 F_y1 做预测，输出两个通道分别代表 scale 和 translation。输出: [B, N, 2]
        s, t = torch.split(st, split_size_or_sections=1, dim=-1)
        s = torch.clamp(s, min=-8, max=8)
        x = y1 + (1 - self.mask) * ((y - t) * torch.exp(-s))
        ldj = (-s).sum(-1)#不是用 Flow 建模分布，忽略

        return x, ldj
```
如果简写为[x,y,z]的话：
$$[x',z']=[x,z]$$
$$y'=(y-t)*exp(-s)$$
t和s的取值是由[x,z]训练出来的，是关于未被变换的维度的函数。
对于耦合层的逆向过程，只需要对刚才变换的那一部分“逆操作”即可，显然每一步都是可逆的。
$$y=y'*exp(s)+t$$
```python
        y, _ = self._call(l1, feat_i, y)
```
所以整个前向传播过程（gpt）：
步骤|作用|
--|--|
affine（可选)|对点坐标做刚性或仿射变换，对点整体施加旋转、缩放、平移|
code_projectors|提取时间（或序列）特征|
_expand_features|将时间特征广播到每个点|
CouplingLayer|每个点位置由时间控制进行扭曲、变形|

## **inverse**
```python
    def inverse(self, t, feat, y):
        x = y
        for i in reversed(self.layer_idx):
            feat_i = self.code_projectors[i](feat)
            feat_i = self._expand_features(feat_i, x)
            l1 = self.layers1[i]
            x, _ = self._call(l1.inverse, feat_i, x)
        if self.affine:
            x = self._affine_input(t, x, inverse=True)
        return x
```
forward	观察坐标 → 规范坐标	把某一时刻的点 x 映射到一个参考帧下的位置 y（用于统一渲染）
inverse	规范坐标 → 观察坐标	从参考帧的 y 推出这个点在时间 t 的实际位置 x（用于训练对齐）
这个过程是可逆的。
# stage1
stage1在训练Neural_InverseTrajectory_Trainer是一个用于生成时间特征的神经网络GaborNet 和神经网络模型NVPSimplified 的混合。NVPSimplified使用位置编码、特征提取和映射机制，将时间帧的坐标映射到标准空间。
从当前时刻的点云位置映射到规范空间，再从规范空间映射到下一帧点云，第一个损失函数是推测的下一帧点云和真实的下一帧点云位置的l2距离。
```python
def train_exhautive_one_step(self,step,data):
        """used for point track model version 4.0
        """
        self.optimizer.zero_grad()
        x = data["pcd"] #
        t= data["time"] #当前帧时间
        fromTo = data["fromTo"][0] #例如'00001_00002'
        next_time = data["target_gt"]["time"] # 下一帧的时间戳
        next_pcd = data["target_gt"]["pcd"]
        next_msk= data["target_gt"]["pcd_target_msk"]# 有效点mask（哪些点被追踪到了）
        
        x_canno_msked, time_feature = self.forward_to_canonical(x[next_msk].unsqueeze(0),t)
        next_xyz_pred_msked = self.inverse_other_t(x_canno_msked,next_time)
        flow_loss= l2_loss(next_xyz_pred_msked,next_pcd[next_msk].unsqueeze(0)
        
        
        loss=0.0
        loss = flow_loss
```
梯度裁剪，防止梯度爆炸，本次训练没用。
```python
        # loss = torch.nn.functional.mse_loss(x_pred, x)
        if self.args.grad_clip > 0:
            for param in self.learnable_params:
                grad_norm = torch.nn.utils.clip_grad_norm_(param, self.args.grad_clip)
                if grad_norm > self.args.grad_clip:
                    print("Warning! Clip gradient from {} to {}".format(grad_norm, self.args.grad_clip))
```
第二个损失函数是当前帧预测的光流（预测的下一帧位置减当前帧位置）损失，每个点的 flow 与其邻居 flow 的差异越小越好。
```python
        if self.args.local_smoothness_loss>0:
            index = None
            if fromTo in  self.precompute_index:
                index = self.precompute_index[fromTo].cuda()
            pcd = x[next_msk]
            flow = next_xyz_pred_msked - pcd
            dic= self.get_local_smoothness_loss(pcd,flow.squeeze(0),index,self.args.neighbor_K)
            
            loss += self.args.local_smoothness_loss*dic["loss"]
            if not fromTo  in  self.precompute_index:
                self.precompute_index[fromTo]=dic["index"].cpu()
        if self.args.local_smoothness_loss:    
            self.scalars_to_log['localSmoothness_loss'] = dic["loss"].detach().item()
        self.scalars_to_log['flow_loss'] = flow_loss.detach().item())
···
```
计算局部 flow 平滑损失，每个点的 flow 与其邻居 flow 的差异越小越好。
```python
    def get_local_smoothness_loss(self,pcd,flow,index=None,neighbor_K=10,loss_type="l2"):
        if index is None:
            pairwise_dist = knn_points(pcd.unsqueeze(0), pcd.unsqueeze(0), K=neighbor_K, return_sorted=False)
            index = pairwise_dist.idx
        neighbor_flows = knn_gather(flow.unsqueeze(0), index)#neighbor_K)
        neighbor_flows=neighbor_flows[:,:,1:,:] ## remove the first point which is the point itself
        if loss_type=="l1":
            loss = torch.mean(torch.abs(flow.unsqueeze(0).unsqueeze(2)-neighbor_flows))
        else:
            loss = torch.mean(torch.square(flow.unsqueeze(0).unsqueeze(2)-neighbor_flows))   
            # loss = torch.mean(torch.square(flow.unsqueeze(0).unsqueeze(2)-neighbor_flows))
        return {"loss":loss,"index":index}
        # pass 
```
```python
    def forward_to_canonical(self, x,t): 
        """ 
        从时间t帧的点坐标x转换到时间t0的标准空间点坐标。
            [B, N, 3] -> [B,N,3]
            
        t：##torch.Size([B, 1])
        x：##torch.Size([B, N, 3])
        """

        # GaborNet 是一种带有Gabor 激活函数的神经网络
        """
        self.feature_mlp = GaborNet(
        in_size=1,
        hidden_size=256,
        n_layers=2,
        alpha=4.5,
        out_size=128
        ).to(self.device)
        """
        time_feature = self.feature_mlp(t)#torch.Size([B, feature_dim])


        # 将一个点（x）从某一时间帧（t）对应的观察坐标系中，映射到“标准空间”（canonical space）中
        '''
        self.deform_mlp = NVPSimplified(n_layers=4,
                                        feature_dims=128,
                                        hidden_size=[256, 256, 256],
                                        proj_dims=256,
                                        code_proj_hidden_size=[],
                                        proj_type='fixed_positional_encoding',
                                        pe_freq=training_args.pe_freq,
                                        normalization=False,
                                        affine=False,
                                        ).to(self.device)
        '''
        x = self.deform_mlp(t,time_feature,x)
        
        return x,time_feature
```
 ```python
        def inverse_cycle_t(self, x,t, time_feature):
        """反向到同一个时刻,这个时候用在fwd时间步得到的time feature ，不用再次计算。

        Args:
            x (_type_): _description_
            t (_type_): _description_
            time_feature (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.deform_mlp.inverse(t,time_feature,x)
        return x
```      
# stage2
准备好进入优化流程：
```python
print(f"Optimizing outdir:{outdir}")
```
把点云的位置从观察坐标（observation frame）转换到标准坐标系（canonical frame），这里应该用了stage1训练好的Neural_InverseTrajectory_Trainer，我第一次跑的时候没传参，第二次跑传了结果没变，可能是我传错了。
```python
init_pcd = get_gaussians_init_pcd(timePcddataset,net_trainer,number_init_pnts=300000) #net_trainer= Neural_InverseTrajectory_Trainer
```
get_gaussians_init_pcd：
```python
def get_gaussians_init_pcd(dataset:BaseCorrespondenceDataset,trainer:Neural_InverseTrajectory_Trainer,mode="average",number_init_pnts=-1,save_pcd=False):
    """_summary_

    Args:
        table (_type_): _description_
        mask (_type_): _description_
        N_se (_type_): _description_
        net_query_fn (_type_): _description_
    """
    print("Get Canonical Space PCD...")
    # 将 trainer 模式设为 eval（推理模式），避免梯度计算
    trainer.to_eval()
    network_query_fn_2canno = lambda inputs, times : trainer.forward_to_canonical(inputs.unsqueeze(0), times.unsqueeze(-1))
        with torch.no_grad():
            ···
               elif isinstance(dataset,ExhaustiveFlowPairsDataset):
            print("Get Canonical Space PCD from ExhaustiveFlowPairsDataset")
            
            pcd_pairs = dataset.time_pcd # 点云
            T = len(pcd_pairs) # 帧数
            canonical_list = []
            for index , item in tqdm(enumerate(pcd_pairs)):
                pcd_t =  item["pcd"] # [N,6]
                #  校验 frame_id 与遍历索引是否一致
                assert int(item["frame_id"])/dataset.PCD_INTERVAL==index,"error"  ### 在Exhaustive paring的时候 frame_id 存储的是 image_id 比如 "000000", "000001", "000002"...
                time_t = int(item["frame_id"])/dataset.PCD_INTERVAL/float(T)
                time_t = torch.Tensor([time_t])
                # 前面定义的network_query_fn_2canno函数在这把点云位置转换到标准空间
                xyz_cano ,_= network_query_fn_2canno(pcd_t[:,:3].cuda(),time_t.cuda()) #[1,N,3]
                rgb_cano = pcd_t[:,3:6]
                canonical_list.append(torch.cat([xyz_cano.cpu().squeeze(0),rgb_cano.cpu()],1)) #[N,6]
                # break
            xyzrgb=torch.cat(canonical_list,0)    #[N*T,6]
            # xyzrgb = torch.stack(canonical_list,0)
            ## save the pcd
            assert xyzrgb.dim()==2,"error"
    ···
```
使用 init_pcd 初始化高斯模型点云坐标（变换好的标准空间坐标）、颜色、特征（SH分量），并通过一系列计算得到与场景相关的属性（如缩放、旋转、不透明度）。
```python
gaussians.create_from_pcd(init_pcd, scene.cameras_extent)
```
gaussians在前面的定义：
```python
ModelClass=GaussianModelTypes[args.gs_model_version]

    if args.gs_model_version=="TimeTable_GaussianModel":
        
        gaussians =ModelClass(args.TimePcd_dir,table_frame_interval=args.PointTrack_frame_interval)
    elif args.gs_model_version=="PointTrackIsotropicGaussianModel":
        gaussians =ModelClass(dataset_args.sh_degree)
    elif args.gs_model_version=="Original_GaussianModel":
        gaussians =ModelClass(dataset_args.sh_degree) # 3
    else:
        raise NotImplementedError("Not implemented yet")
```
初始化3d高斯训练所需的配置或参数，启动训练模式：
```python
gaussians.training_setup(opt)
```
```python
resume_step =1#testing_iterations.remove(15000)

    timePcddataset.clean()
    del init_pcd,
    torch.cuda.empty_cache()
    bg_color = [1, 1, 1] if dataset_args.white_background else [0, 0, 0] # 设置背景颜色
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    black_bg =  torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
```
Neural_InverseTrajectory_Trainer启动训练模式：
```python
net_trainer.to_trainning()
idx_stack = None
    progress_bar = tqdm(range(resume_step, args.stageCoTrain_max_steps), desc="Training progress")
    
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    ema_loss_for_log = 0.0
```
整个训练过程，3d高斯和Neural_InverseTrajectory_Trainer同时训练优化：
```python
    for iteration in range(resume_step,args.stageCoTrain_max_steps+1,1):
        # for idx,data in enumerate(dataloader):
        iter_start.record()    # 开始记录迭代开始时间
        # if (iteration+1) % 3000==0 and args.all_SH:
        #     print("SHdegree up")
        #     gaussians.oneupSHdegree()
        # ## NOTE: Canceling SHdegree up
        
        if not idx_stack:
            # viewpoint_stack = scene.getTrainCameras().copy()
            # viewpoint_PCD_stack = copy.deepcopy(scene.getCoTrainingCameras())
            # 从场景中获得了一组训练相机（或视角）与点云对
            viewpoint_PCD_stack = scene.getCoTrainingCameras() ## FIXME :canceled copy.deepcopy exhaustive pairs occupy too much memory
            if scene.is_overfit_aftergrowth:
                
                idx_stack = scene.getOverfitIdxStack(iteration=iteration)
            else:
                idx_stack = torch.randperm(len(viewpoint_PCD_stack)).tolist()

        idx = idx_stack.pop()
        viewpoint,pcd_pair  = viewpoint_PCD_stack[idx]
        
        #### Predict Gaussian position. 
        xyz= gaussians.get_xyz
        # time = viewpoint_time.unsqueeze(0)(viewpoint.time, pcd_pair["time"])
        time = viewpoint.time.unsqueeze(0)
        
        predicted_xyz= net_trainer.inverse_other_t(xyz.unsqueeze(0),time.unsqueeze(0))
        ###

        ## Render using Gaussian
        bg = torch.rand((3), device="cuda") if args.random_background else background
        render_pkg = renderFunc(viewpoint, gaussians, pipe, bg, override_color = None, specified_xyz=predicted_xyz,
                            )
        
        # render_pkg = original_render(viewpoint, gaussians, pipe, background, override_color = None,specified_xyz=predicted_xyz)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[ "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        render_depth,render_alpha = render_pkg["depth"],render_pkg["alpha"]
        
        depth_mask = viewpoint.depth > 0
        mask = viewpoint.mask
        if mask is not None:
            mask = mask[None].to(image) ##(3,H,w)-->(1,H,w)
        ### Calculate Loss
        gt_image = viewpoint.original_image.cuda()
        Ll1 = opt.lambda_recon*l1_loss(image, gt_image)
        Lssim = opt.lambda_dssim*(1.0 - ssim(image, gt_image))
        loss = Ll1 + Lssim  


        L2d_render_flow= None
            # L2d_render_flow= torch.Tensor([0.0]).cuda()


            

        if opt.lambda_depthOderLoss>0:    
            # depth_mask = viewpoint.depth > 0
            

            LdepthOrder = get_depth_order_loss(render_depth,viewpoint.depth,depth_mask,method_name=args.depth_order_loss_type
                                        ,alpha=args.Alpha_tanh
                                        )
        
            loss += opt.lambda_depthOderLoss*LdepthOrder   
        else:
            LdepthOrder= None

        
        # loss = Ll1 + Lssim + opt.lambda_gs_approx_flow*Lgs_flow +opt.lambda_pcd_flow*Lpcd_flow +\
        #     opt.lambda_depth_plane*Ldepthplane+ opt.lambda_opacity_sparse*LsparseOpacity + opt.lambda_depthloss*Ldepth +\
        #         opt.lambda_2dflowloss*L2d_render_flow
        ### Calculate Loss
        loss.backward()
        
        
        
        iter_end.record()
        loss_dict= {"Ll1":Ll1,"Lssim":Lssim,
                    "LdepthOrder":LdepthOrder,
                    "loss_total":loss}
        
        ## record error information
        if iteration > opt.custom_densification_start and \
            iteration < opt.custom_densification_end:
            info_dict = {"render":image.detach(),"render_depth":render_depth.detach(),"render_alpha":render_alpha.detach(),}
        
        
        with torch.no_grad():

               
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == args.stageCoTrain_max_steps:
                progress_bar.close()
                
            net_trainer.log(iteration,writer) ## log lr of deform and feature mlp   
            # training_report(writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, original_render, (pipe, background))  
            cotraining_report(writer, iteration, loss_dict, iter_start.elapsed_time(iter_end), testing_iterations, scene,renderFunc, (pipe, background))
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset_args.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    
            ## custom Densification for Mono3D GS region that are underconstructed.

            # Optimizer iteration
            if iteration < args.stageCoTrain_max_steps: ## FUCK YOU 
                # Optimizer step
                # if iteration < opt.iterations:
                #step
                gaussians.optimizer.step()
                net_trainer.optimizer.step()
                ## zero grad
                gaussians.optimizer.zero_grad(set_to_none=True)
                net_trainer.optimizer.zero_grad()
                ## update lr rate
                net_trainer.scheduler.step()
                # net_trainer.update_learning_rate(iteration) TODO: update learning rate   
                gaussians.update_learning_rate(iteration)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), os.path.join(scene.model_path,args.timestamp) + "/chkpnt" + str(iteration) + ".pth")
                net_trainer.save_model(iteration)
            
            
        iteration+=1
    # return trainer
    try: 
        evaluation_on_metricCam( scene,net_trainer,gaussians,args,pipe,renderFunc,black_bg)
    except Exception  as e:
        pass
    pass 
```
过程中：
```python
#### Predict Gaussian position. 
        xyz= gaussians.get_xyz
        # time = viewpoint_time.unsqueeze(0)(viewpoint.time, pcd_pair["time"])
        time = viewpoint.time.unsqueeze(0)
        
        predicted_xyz= net_trainer.inverse_other_t(xyz.unsqueeze(0),time.unsqueeze(0))
        ###
        ## Render using Gaussian
        bg = torch.rand((3), device="cuda") if args.random_background else background
        render_pkg = renderFunc(viewpoint, gaussians, pipe, bg, override_color = None, specified_xyz=predicted_xyz,
                            )
        
        # render_pkg = original_render(viewpoint, gaussians, pipe, background, override_color = None,specified_xyz=predicted_xyz)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[ "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        render_depth,render_alpha = render_pkg["depth"],render_pkg["alpha"]
        
        depth_mask = viewpoint.depth > 0
        mask = viewpoint.mask
        if mask is not None:
            mask = mask[None].to(image) ##(3,H,w)-->(1,H,w)
        ### Calculate Loss
        gt_image = viewpoint.original_image.cuda()
        Ll1 = opt.lambda_recon*l1_loss(image, gt_image)
        Lssim = opt.lambda_dssim*(1.0 - ssim(image, gt_image))
        loss = Ll1 + Lssim
```
理解一下这个训练步骤：
拿到标准空间坐标：
```python 
#### Predict Gaussian position. 
        xyz= gaussians.get_xyz
```
随机到的帧：
```python
# time = viewpoint_time.unsqueeze(0)(viewpoint.time, pcd_pair["time"])
time = viewpoint.time.unsqueeze(0)
```
利用Neural_InverseTrajectory_Trainer反推当前帧点云位置：
```python
predicted_xyz= net_trainer.inverse_other_t(xyz.unsqueeze(0),time.unsqueeze(0))
```
利用高斯模型参数和推导出的位置渲染该帧图像，计算和真实图像间的损失：
```python
## Render using Gaussian
        bg = torch.rand((3), device="cuda") if args.random_background else background
        render_pkg = renderFunc(viewpoint, gaussians, pipe, bg, override_color = None, specified_xyz=predicted_xyz,
                            )
        
        # render_pkg = original_render(viewpoint, gaussians, pipe, background, override_color = None,specified_xyz=predicted_xyz)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[ "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        render_depth,render_alpha = render_pkg["depth"],render_pkg["alpha"]
        
        depth_mask = viewpoint.depth > 0
        mask = viewpoint.mask
        if mask is not None:
            mask = mask[None].to(image) ##(3,H,w)-->(1,H,w)
        ### Calculate Loss
        gt_image = viewpoint.original_image.cuda()
        Ll1 = opt.lambda_recon*l1_loss(image, gt_image)
        Lssim = opt.lambda_dssim*(1.0 - ssim(image, gt_image))
        loss = Ll1 + Lssim
```
整个过程优化两个模型的参数。