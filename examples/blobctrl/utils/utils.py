import importlib
from typing import Any, Union, TypeVar, Tuple, Optional, Dict, OrderedDict
from copy import deepcopy
from PIL import Image

import einops
import cv2

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from matplotlib import cm

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from PIL import ImageDraw, Image, ImageFont
import torchvision.transforms.functional as F

einsum = torch.einsum

BLOB_VIS_COLORS = torch.tensor(
    [
        [0.9804, 0.9451, 0.9176],                                                                                            
        [1.0, 0.494, 0.357],
        [0.961, 0.882, 0.827],
        [0.8980, 0.5255, 0.0235],                                                                                            
        [0.3647, 0.4118, 0.6941],                                                                                            
        [0.3216, 0.7373, 0.6392],                                                                                            
        [0.6000, 0.7882, 0.2706],                                                                                            
        [0.1843, 0.5412, 0.7686],                                                                                            
        [0.6471, 0.6667, 0.6000],                                                                                            
        [0.8549, 0.6471, 0.1059],                                                                                            
        [0.4627, 0.3059, 0.6235],                                                                                            
        [0.8000, 0.3804, 0.6902],                                                                                            
        [0.9294, 0.3922, 0.3529],                                                                                            
        [0.1412, 0.4745, 0.4235],                                                                                            
        [0.4000, 0.7725, 0.8000],                                                                                            
        [0.9647, 0.8118, 0.4431],                                                                                            
        [0.9725, 0.6118, 0.4549],                                                                                            
        [0.8627, 0.6902, 0.9490],                                                                                            
        [0.5294, 0.7725, 0.3725],                                                                                            
        [0.6196, 0.7255, 0.9529],                                                                                            
        [0.9961, 0.5333, 0.6941],                                                                                            
        [0.7882, 0.8588, 0.4549],                                                                                            
        [0.5451, 0.8784, 0.6431],                                                                                            
        [0.7059, 0.5922, 0.9059],                                                                                            
        [0.7020, 0.7020, 0.7020],                                                                                            
        [0.5216, 0.3608, 0.4588],                                                                                            
        [0.8510, 0.6863, 0.4196],                                                                                            
        [0.6863, 0.3922, 0.3451],                                                                                            
        [0.4510, 0.4353, 0.298]
    ])



def splat_features_from_scores(scores: Tensor, features: Tensor, size: Optional[int],
                               channels_last: bool = True) -> Tensor:
    """

    Args:
        channels_last: ∂
        scores: [N, H, W, M] (or [N, M, H, W] if not channels last)
        features: [N, M, C]
        size: dimension of map to return
    Returns: [N, C, H, W]

    """
    features = features.to(dtype=scores.dtype, device=scores.device)
    if size and not (scores.shape[2] == size):
        if channels_last:
            scores = einops.rearrange(scores, 'n h w m -> n m h w')
        scores = torch.nn.functional.interpolate(scores, size, mode='bilinear', align_corners=False)
        einstr = 'nmhw,nmc->nchw'
    else:
        einstr = 'nhwm,nmc->nchw' if channels_last else 'nmhw,nmc->nchw'
    return einsum(einstr, scores, features).contiguous()


def splat_features(
            xs: Tensor, 
            ys: Tensor, 
            covs: Tensor, 
            sizes: Tensor, 
            score_size: Optional[int] = None, 
            interp_size: int = None,
            features: Tensor = None, 
            viz_size: Optional[int] = None, 
            is_viz: bool = False,
            ret_layout: bool = True,
            viz_score_fn=None,
            return_d_score=False,
            only_vis: bool = False,
            only_splatting_fg: bool = False,
            only_splatting_bg: bool = False,
            **kwargs) -> Dict:
        """
        Args:
            xs: [N, M] X-coord location in [0,1]
            ys: [N, M] Y-coord location in [0,1]
            features: [N, M+1, dim] feature vectors to splat (and bg feature vector)
            covs: [N, M, 2, 2] xy covariance matrices for each feature
            sizes: [N, M+1] distributions of per feature (and bg) weights
            interp_size: output grid size
            score_size: size at which to render score grid before downsampling to size
            viz_size: visualized grid in RGB dimension
            viz: whether to visualize
            ret_layout: whether to return dict with layout info
            viz_score_fn: map from raw score to new raw score for generating blob maps. if you want to artificially enlarge blob borders, e.g., you can send in lambda s: s*1.5
            return_d_score: only return d_score
            only_vis: only return visualize result
            only_splatting_fg: only splatting fg
            only_splatting_bg: only splatting bg
            **kwargs: unused

        Returns: dict with requested information
        """


        if not isinstance(viz_size, int) and viz_size is not None:
            height, width = viz_size
            feature_coords = torch.stack((xs.mul(width), ys.mul(height)), -1)  # [n, m, 2]
            grid_x = torch.arange(width).repeat(height)  
            grid_y = torch.arange(height).repeat_interleave(width)
            grid_coords = torch.stack((grid_x, grid_y), dim=0).to(xs.device)
            delta = (grid_coords[None, None] - feature_coords[..., None])  # [n, m, 2, size*size]
            delta[:,:,0,:] /= width
            delta[:,:,1,:] /= height
            
            # # Now compute the Mahalanobis distance
            sq_mahalanobis = (delta * torch.linalg.solve(covs, delta)).sum(2)
            batch = 1
            n_gaussians = 1
            sq_mahalanobis = sq_mahalanobis.view(batch, n_gaussians, height, width).contiguous()
            sq_mahalanobis = sq_mahalanobis.permute(0,2,3,1).contiguous()
        else:
            if isinstance(score_size, int):
                feature_coords = torch.stack((xs, ys), -1).mul(score_size)  # [n, m, 2]
                grid_coords = torch.stack(
                    (torch.arange(score_size).repeat(score_size), torch.arange(score_size).repeat_interleave(score_size))).to(
                    xs.device)  # [2, size*size]
                delta = (grid_coords[None, None] - feature_coords[..., None]).div(score_size)  # [n, m, 2, size*size]
                sq_mahalanobis = (delta * torch.linalg.solve(covs, delta)).sum(2)
                sq_mahalanobis = rearrange(sq_mahalanobis, 'n m (s1 s2) -> n s1 s2 m', s1=score_size)
            else:
                height, width = score_size
                feature_coords = torch.stack((xs.mul(width), ys.mul(height)), -1)  # [n, m, 2]
                grid_x = torch.arange(width).repeat(height)  
                grid_y = torch.arange(height).repeat_interleave(width)
                grid_coords = torch.stack((grid_x, grid_y), dim=0).to(xs.device)
                delta = (grid_coords[None, None] - feature_coords[..., None])  # [n, m, 2, size*size]
                delta[:,:,0,:] /= width
                delta[:,:,1,:] /= height
                
                # # Now compute the Mahalanobis distance
                sq_mahalanobis = (delta * torch.linalg.solve(covs, delta)).sum(2)
                batch = 1
                n_gaussians = 1
                sq_mahalanobis = sq_mahalanobis.view(batch, n_gaussians, height, width).contiguous()
                sq_mahalanobis = sq_mahalanobis.permute(0,2,3,1).contiguous()

        scores = sq_mahalanobis.div(-1).sigmoid()
        scores = scores.mul(2).clamp_(max=1)
        
        if sizes.ndim == 3:
            sizes = sizes.squeeze(-1)
        is_not_exits = sizes < 0.5
        is_not_exits = is_not_exits[:, None, None, :]
        is_not_exits_expanded = is_not_exits.expand(-1, scores.shape[1], scores.shape[2], -1)
        
        # scores[is_exits_expanded] = 1e-5
        scores = torch.where(is_not_exits_expanded, torch.tensor(1e-6, device=scores.device), scores)

    
        bg_scores = torch.ones_like(scores[..., :1])
        scores = torch.cat((bg_scores, scores), -1)  
        
        # alpha composite
        rev = list(range(scores.size(-1) - 1, -1, -1))  # flip, but without copy
        d_scores = (1 - scores[..., rev]).cumprod(-1)[..., rev].roll(-1, -1) * scores
        d_scores[..., -1] = scores[..., -1]

        if only_splatting_bg:
            d_scores = d_scores[..., 0]
        elif only_splatting_fg:
            d_scores = d_scores[..., 1:]
        else:
            d_scores = d_scores

        if d_scores.ndim == 3:
            d_scores = d_scores.unsqueeze(-1)

        if return_d_score:
            return rearrange(d_scores, 'n h w m -> n m h w')

        ret = {}
        
        if is_viz:
            if viz_score_fn is not None:
                viz_posterior = viz_score_fn(scores)
                
                viz_posterior_inv = deepcopy(viz_posterior)
                viz_posterior_inv[:,:,:,1:] = 1 - viz_posterior_inv[:,:,:,1:]
                
                scores_viz = (1 - viz_posterior[..., rev]).cumprod(-1)[..., rev].roll(-1, -1) * viz_posterior
                scores_viz[..., -1] = viz_posterior[..., -1]
                
                scores_viz_inv = (1 - viz_posterior_inv[..., rev]).cumprod(-1)[..., rev].roll(-1, -1) * viz_posterior_inv
                scores_viz_inv[..., -1] = viz_posterior_inv[..., -1]
            else:
                scores_viz = d_scores
                
            n_gaussians = xs.shape[-1]
            ret.update(visualize_features(viz_size, n_gaussians, scores_viz, kwargs.get("viz_colors", None)))
            # from PIL import Image
            # import numpy as np
            # viz_colors = torch.load("BED_CONF_COLORS.pt")
            # kwargs["viz_colors"] = viz_colors
            # layout_img = Image.fromarray((((ret['feature_img'][0]+1) / 2) * 255).cpu().numpy().transpose(1,2,0).astype(np.uint8))


        if only_vis:
            return ret

        
        score_img = rearrange(d_scores, 'n h w m -> n m h w')
        ret['scores_pyramid'] = pyramid_resize(score_img, cutoff=interp_size)
    
        
        feature_grid = splat_features_from_scores(ret['scores_pyramid'][interp_size], 
                                                features, 
                                                interp_size, 
                                                channels_last=False,)
        
        ret.update({'feature_grid': feature_grid, 'feature_img': None, 'entropy_img': None})
        if ret_layout:
            layout = {'xs': xs, 'ys': ys, 'covs': covs, 'raw_scores': scores, 'sizes': sizes,
                        'composed_scores': d_scores, 'features': features}
            ret.update(layout)
            
        return ret


@torch.no_grad()
def visualize_features(viz_size=64, n_gaussians=None, scores=None,
                        viz_colors=None) -> Dict[str, Tensor]:
    n_gaussians = n_gaussians+1
   
    rand_colors = viz_colors is None
    viz_colors = viz_colors.to(scores.device) if not rand_colors else torch.rand((n_gaussians,3)).to(scores.device)
    if viz_colors.ndim == 2:
        # viz colors should be [Kmax, 3]
        viz_colors = viz_colors[:n_gaussians][None].repeat_interleave(len(scores), 0)
    elif viz_colors.ndim == 3:
        # viz colors should be [Nbatch, Kmax, 3]
        viz_colors = viz_colors[:, :n_gaussians]
    else:
        viz_colors = torch.rand((n_gaussians,3))
    img = splat_features_from_scores(scores, viz_colors, viz_size)
    if rand_colors:
        imax = img.amax((2, 3))[:, :, None, None]
        imin = img.amin((2, 3))[:, :, None, None]
        feature_img = img.sub(imin).div((imax - imin).clamp(min=1e-5)).mul(2).sub(1)
    else:
        feature_img = img
        
    out = {
        'feature_img': feature_img
    }        
    return out


def rotation_matrix(theta):
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    return torch.stack([cos, sin, -sin, cos], dim=-1).view(*theta.shape, 2, 2)



def pyramid_resize(img, cutoff):
    """

    Args:
        img: [N x C x H x W]
        cutoff: threshold at which to stop pyramid

    Returns: gaussian pyramid

    """
    out = [img]
    while img.shape[-1] > cutoff:
        img = torch.nn.functional.interpolate(img, img.shape[-1] // 2, mode='bilinear', align_corners=False)
        out.append(img)
    return {i.size(-1): i for i in out}


def ellipse_to_gaussian(x, y, a, b, theta):
    """
    将椭圆参数转换为高斯分布的均值和协方差矩阵。

    参数:
    x (float): 椭圆中心的 x 坐标。
    y (float): 椭圆中心的 y 坐标。
    a (float): 椭圆的短半轴长度。
    b (float): 椭圆的长半轴长度。
    theta (float): 椭圆的旋转角度（以弧度为单位）, 长半轴逆时针角度。

    返回:
    mean (numpy.ndarray): 高斯分布的均值，形状为 (2,) 的数组，表示 (x, y) 坐标。
    cov_matrix (numpy.ndarray): 高斯分布的协方差矩阵，形状为 (2, 2) 的数组。
    """
    # 均值
    mean = np.array([x, y])
    
    # 协方差的主对角线元素
    # sigma_x = b / np.sqrt(2)
    # sigma_y = a / np.sqrt(2)
    # 不除以 sqrt(2) 也是可以的。这个转换主要是为了在特定的统计上下文中，
    # 使得椭圆的半轴长度对应于高斯分布的一个标准差。
    # 这样做的目的是为了使得椭圆的面积包含了高斯分布约68%的概率质量（在一维高斯分布中，一个标准差的范围内包含了约68%的概率质量）。

    # 协方差的主对角线元素
    sigma_x = b 
    sigma_y = a 
    # 协方差矩阵（未旋转）
    cov_matrix = np.array([[sigma_x**2, 0],
                            [0, sigma_y**2]])
    
    # 旋转矩阵
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    # 旋转协方差矩阵
    cov_matrix_rotated = R @ cov_matrix @ R.T
    
    cov_matrix_rotated[0, 1] *= -1  # 反转协方差矩阵的非对角元素
    cov_matrix_rotated[1, 0] *= -1  # 反转协方差矩阵的非对角元素
    
    # eigenvalues, eigenvectors = np.linalg.eig(cov_matrix_rotated)
    
    return mean, cov_matrix_rotated


def gaussian_to_ellipse(mean, cov_matrix):
    """
    将高斯分布的均值和协方差矩阵转换为椭圆参数。

    参数:
    mean (numpy.ndarray): 高斯分布的均值，形状为 (2,) 的数组，表示 (x, y) 坐标。
    cov_matrix (numpy.ndarray): 高斯分布的协方差矩阵，形状为 (2, 2) 的数组。

    返回:
    x (float): 椭圆中心的 x 坐标。
    y (float): 椭圆中心的 y 坐标。
    a (float): 椭圆的短半轴长度。
    b (float): 椭圆的长半轴长度。
    angle_clockwise (float): 椭圆的旋转角度, 短轴顺时针绕x旋转, 0~180度, t以角度为单位。
    """
    # 提取均值
    x, y = mean

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 计算长半轴和短半轴
    # b = np.sqrt(2 * max(eigenvalues))  # 长半轴
    # a = np.sqrt(2 * min(eigenvalues))  # 短半轴
    
    b = np.sqrt(max(eigenvalues))  # 长半轴
    a = np.sqrt(min(eigenvalues))  # 短半轴

    # 计算短轴的方向
    # 找到最小特征值对应的特征向量
    min_index = np.argmin(eigenvalues)
    min_axis_vector = eigenvectors[:, min_index]
    theta = np.arctan2(min_axis_vector[1], min_axis_vector[0])

    angle_clockwise = np.degrees(theta)  # 转换为度
    
    if angle_clockwise < 0:
        angle_clockwise += 180


    return x, y, a, b, angle_clockwise


def viz_score_fn(score):
    # score = score.clone()
    # score[..., 1:].mul_(2).clamp_(max=1)
    return score


def vis_scores(blob_d_score, viz_size):
    n_gaussians = blob_d_score.shape[1] - 1

    scores_viz = rearrange(blob_d_score, 'n m h w -> n h w m')

    viz_colors = torch.load("BED_CONF_COLORS.pt")
    kwargs = {"viz_colors": viz_colors}
    ret = visualize_features(viz_size=viz_size, n_gaussians=n_gaussians, scores=scores_viz, **kwargs)

    return ret['feature_img']


def vis_gt_ellipse_from_norm_gs(img, gt_mus, gt_covs, color=None):
    '''img: h,w,c 0~255, mus: n,2 covs:n,2,2'''
    # color = (127,127,127)
    result = img.copy()
    height,width,c = img.shape
    max_length = np.sqrt(width**2 + height**2)
    for mu, cov in zip(gt_mus, gt_covs):
        mean = tuple(mu.numpy() for mu in mu.cpu().unbind(-1))
        cov_matrix = cov.cpu().numpy()
    
        xc, yc, a, b, angle_clockwise_short_axis = gaussian_to_ellipse(mean, cov_matrix)
        # print(xc, yc, 2*a, 2*b, angle_clockwise_short_axis)
        
        xc = xc * width
        yc = yc * height
        a = a * max_length
        b = b * max_length
        ellipse = (xc,yc),(a*2,b*2),angle_clockwise_short_axis
        
        if color is None:
            color = [255,0,0]
            
        
        cv2.ellipse(result, ellipse, color, 3)
    
    return result


def vis_gt_ellipse_from_norm_ellipse(img, norm_ellipse, color=None):
    '''img: h,w,c 0~255, norm_ellipse: xc,yc,d1,d2,theta'''
    height,width,c = img.shape
    max_length = np.sqrt(width**2 + height**2)
    (xc,yc), (d1,d2), theta = norm_ellipse
    xc = xc * width
    yc = yc * height
    d1 = d1 * max_length
    d2 = d2 * max_length
    ellipse = (xc,yc),(d1,d2),theta
    if color is None:
        color = [255,0,0]
    cv2.ellipse(img, ellipse, color, 3)
    return img


def vis_gt_ellipse_from_ellipse(img, ellipse, color=None):
    '''img: h,w,c 0~255, ellipse: xc,yc,d1,d2,theta'''
    (xc,yc),(d1,d2),theta = ellipse
    ellipse = (xc,yc),(d1,d2),theta
    if color is None:
        color = [255,0,0]
    cv2.ellipse(img, ellipse, color, 3)
    return img
