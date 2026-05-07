class Config2D:
    # ----- 随机种子 -----
    seed = 2026
    deterministic = False

    # ----- 数据路径（请根据实际修改）-----
    data_root = "./data"
    vp_path = "./data/Vp.npy"
    vs_path = "./data/Vs.npy"
    den_path = "./data/Den.npy"
    seismic_mat_path = "./data/elastic_impedance_results.mat"
    gmm_prior_path = None               # 2D 版本不使用 GMM 先验，直接手动定义
    spatial_prior_path = "./data/spatial_prior3.pt"   # 空间先验 (K, H, W)

    # ----- 数据预处理参数 -----
    sigma_low = 30                      # 低频模型平滑尺度
    # 训练/验证道索引
    num_train = 12
    num_val = 3
    train_start, train_end = 20, 1300
    val_start, val_end = 30, 1300
    radius = 7                          # 切片窗口半径（仅 slice 模式）
    batch_size = 30

    # ----- 正演参数 -----
    dt = 0.002
    f = 20.0
    angles_deg = [5, 10, 15, 20, 25, 30]
    wavelet_len_sec = 0.1

    # ----- 模型架构 -----
    seismic_channels = 6
    low_freq_channels = 3
    base_channels = 32
    norm_type = 'group'                # 'batch', 'group', 'instance'
    num_experts = 3
    initial_temperature = 1.0
    final_temperature = 0.05
    film_hidden_dim = 8
    film_kernel_size = 3
    modulation_scale = 1.0

    # ----- 先验参数（手动给出，来自 notebook）-----
    # 每个专家的均值 (K, 3)
    prior_means_np = [[-0.9952, -0.9990, -0.9788],
                      [ 1.2965,  1.2997,  0.8848],
                      [-0.0301, -0.0286,  0.3536]]
    # 每个专家的协方差矩阵 (K, 3, 3)
    prior_cov_np = [
        [[0.1157, 0.1137, 0.0987],
         [0.1137, 0.1152, 0.0890],
         [0.0987, 0.0890, 0.1721]],
        [[ 0.3275,  0.3264, -0.0785],
         [ 0.3264,  0.3258, -0.0803],
         [-0.0785, -0.0803,  1.0151]],
        [[0.2051, 0.2164, 0.1411],
         [0.2164, 0.2314, 0.1390],
         [0.1411, 0.1390, 0.1950]]
    ]

    # ----- 损失函数参数 -----
    seismic_obs_noise_init = 5.0
    well_obs_noise_init = 1.0
    clip_diff = 10.0
    expert_lambdas = [0.1, 0.01757, 0.06074]   # 专家 KL 权重

    # ----- 阶段1训练参数（联合训练，强 KL）-----
    phase1_epochs = 50
    phase1_lr = 1e-3
    phase1_kl_z = 0.1
    phase1_kl_theta = 0.01
    phase1_num_samples = 2               # 地震似然采样次数

    # ----- 阶段2训练参数（冻结门控，弱 KL）-----
    phase2_epochs = 500
    phase2_lr = 1e-2
    phase2_kl_z = 0.001
    phase2_kl_theta = 0.001
    phase2_num_samples = 1

    save_path = "best_model.pth"
    