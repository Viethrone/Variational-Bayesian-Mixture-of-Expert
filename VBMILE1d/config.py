import torch
import numpy as np

class Config:
    # ----- 随机种子 -----
    seed = 2026
    deterministic = False

    # ----- 数据路径（请根据实际情况修改）-----
    data_root = "./data"   # 存放 Vp.npy, Vs.npy, Den.npy, elastic_impedance_results.mat, gmm_priors3.npz, spatial_prior3.pt
    # 或者将数据单独用变量指定
    vp_path = "./data/Vp.npy"
    vs_path = "./data/Vs.npy"
    den_path = "./data/Den.npy"
    seismic_mat_path = "./data/elastic_impedance_results.mat"
    gmm_prior_path = "./data/gmm_priors3.npz"
    spatial_prior_path = "./data/spatial_prior3.pt"
    
    # ----- 数据预处理参数 -----
    sigma_low = 30          # 高斯平滑参数

    # ----- 训练/验证索引 -----
    num_train = 12
    num_val = 5
    train_start, train_end = 20, 1300
    val_start, val_end = 30, 1300

    # ----- 正演参数 -----
    dt = 0.002
    f = 20.0
    angles_deg = [5, 10, 15, 20, 25, 30]
    wavelet_length = 101

    # ----- 模型架构 -----
    seismic_channels = 6
    low_freq_channels = 3
    base_channels = 16
    norm_type = 'group'
    num_experts = 3
    initial_temperature = 1.0
    final_temperature = 0.05

    # ----- 损失参数 -----
    seismic_obs_noise_init = 5.0
    well_obs_noise_init = 1.0
    clip_diff = 10.0

    # ----- 训练超参数 -----
    total_epochs = 500
    warmup_epochs = 50
    learning_rate = 1e-3
    weight_decay = 1e-2
    warmup_start_lr = 1e-5
    eta_min = 1e-4
    grad_clip_norm = 1.0
    batch_size = 20
    save_path = "best_model.pth"
    