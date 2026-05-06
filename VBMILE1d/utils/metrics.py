import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
from typing import Dict

def compute_metrics(pred: np.ndarray, true: np.ndarray, mask: np.ndarray = None) -> Dict[str, float]:
    """
    计算逐像素的回归指标（R², PCC, RMSE, MAE, SSIM）。
    若提供 mask，则仅计算 mask=1 的像素。

    Returns:
        字典：包含每个通道的指标，以及整体平均。
    """
    C = pred.shape[0]
    metrics = {}

    # 展平
    if mask is not None:
        mask_flat = mask.flatten() > 0
        pred_flat = pred.reshape(C, -1)[:, mask_flat]
        true_flat = true.reshape(C, -1)[:, mask_flat]
    else:
        pred_flat = pred.reshape(C, -1)
        true_flat = true.reshape(C, -1)

    for c in range(C):
        # R²
        r2 = r2_score(true_flat[c], pred_flat[c])
        # PCC
        corr, _ = pearsonr(true_flat[c], pred_flat[c])
        # RMSE
        rmse = np.sqrt(np.mean((pred_flat[c] - true_flat[c]) ** 2))
        # MAE
        mae = np.mean(np.abs(pred_flat[c] - true_flat[c]))
        # SSIM
        global_range = 6.0   # 如果归一化后数据基本在 [-3, 3]
        ssim_val = ssim(pred[c], true[c], data_range=global_range)

        metrics[f'R2_ch{c}'] = r2
        metrics[f'PCC_ch{c}'] = corr
        metrics[f'RMSE_ch{c}'] = rmse
        metrics[f'MAE_ch{c}'] = mae
        metrics[f'SSIM_ch{c}'] = ssim_val

    # 平均指标
    for name in ['R2', 'PCC', 'RMSE', 'MAE', 'SSIM']:
        vals = [metrics[f'{name}_ch{c}'] for c in range(C)]
        metrics[f'{name}_avg'] = np.mean(vals)

    # 打印 Vp（通道0）的指标
    print("===== Vp Metrics =====")
    for name in ['R2', 'PCC', 'RMSE', 'MAE', 'SSIM']:
        print(f"{name}: {metrics[f'{name}_ch0']:.4f}")

    return metrics

