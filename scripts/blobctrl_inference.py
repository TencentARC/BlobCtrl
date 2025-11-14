import os
import sys
import uuid

import torch
import numpy as np
from PIL import Image
import cv2
import argparse

from diffusers import (
    UNet2DConditionModel, 
    UniPCMultistepScheduler, 
    DDIMScheduler, 
    DPMSolverMultistepScheduler,
)
from transformers import AutoImageProcessor, Dinov2Model

from blobctrl.utils.utils import splat_features
from blobctrl.models.blobnet import BlobNetModel
from blobctrl.pipelines.pipeline_blobnet import StableDiffusionBlobNetPipeline

def ellipse_to_gaussian(x, y, a, b, theta):
    """
    Convert ellipse parameters to mean and covariance matrix of a Gaussian distribution.

    Parameters:
    x (float): x-coordinate of the ellipse center.
    y (float): y-coordinate of the ellipse center.
    a (float): Length of the minor semi-axis of the ellipse.
    b (float): Length of the major semi-axis of the ellipse.
    theta (float): Rotation angle of the ellipse (in radians), counterclockwise angle of the major axis.

    Returns:
    mean (numpy.ndarray): Mean of the Gaussian distribution, an array of shape (2,) representing (x, y) coordinates.
    cov_matrix (numpy.ndarray): Covariance matrix of the Gaussian distribution, an array of shape (2, 2).
    """
    # Mean
    mean = np.array([x, y])
    
    # Diagonal elements of the covariance matrix
    # sigma_x = b / np.sqrt(2)
    # sigma_y = a / np.sqrt(2)
    # Not dividing by sqrt(2) is also acceptable. This conversion is mainly for specific statistical contexts,
    # to make the semi-axis length of the ellipse correspond to one standard deviation of the Gaussian distribution.
    # The purpose is to make the ellipse area contain about 68% of the probability mass of the Gaussian distribution
    # (in a one-dimensional Gaussian distribution, one standard deviation contains about 68% of the probability mass).

    # Diagonal elements of the covariance matrix
    sigma_x = b 
    sigma_y = a 
    # Covariance matrix (before rotation)
    cov_matrix = np.array([[sigma_x**2, 0],
                            [0, sigma_y**2]])
    
    # Rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    
    # Rotate the covariance matrix
    cov_matrix_rotated = R @ cov_matrix @ R.T
    
    cov_matrix_rotated[0, 1] *= -1  # Reverse the non-diagonal elements of the covariance matrix
    cov_matrix_rotated[1, 0] *= -1  # Reverse the non-diagonal elements of the covariance matrix
    
    # eigenvalues, eigenvectors = np.linalg.eig(cov_matrix_rotated)
    
    return mean, cov_matrix_rotated


def get_theta_anti_clockwise_long_axis(angle_clockwise_short_axis):
    angle_anti_clockwise_short_axis = (180 - angle_clockwise_short_axis) % 180
    angle_anti_clockwise_long_axis = (angle_anti_clockwise_short_axis + 90) % 180
    theta_anti_clockwise_long_axis = np.radians(angle_anti_clockwise_long_axis)
    return theta_anti_clockwise_long_axis


def get_gs_from_ellipse(ellipse):
    (xc,yc), (d1,d2), angle_clockwise_short_axis = ellipse
    theta_anti_clockwise_long_axis = get_theta_anti_clockwise_long_axis(angle_clockwise_short_axis)

    a = d1 / 2
    b = d2 / 2
    mean, cov_matrix = ellipse_to_gaussian(xc, yc, a, b, theta_anti_clockwise_long_axis)
    return mean, cov_matrix


def normalize_gs(mean, cov_matrix_rotated, width, height):
    # Normalize mean
    normalized_mean = mean / np.array([width, height])
    
    # Calculate maximum length for normalizing the covariance matrix
    max_length = np.sqrt(width**2 + height**2)
    
    # Normalize covariance matrix
    normalized_cov_matrix = cov_matrix_rotated / (max_length ** 2)
    
    return normalized_mean, normalized_cov_matrix


def get_blob_dict_from_norm_gs(normalized_mean, normalized_cov_matrix):
    xs, ys = normalized_mean
    blob = {
        "xs": torch.tensor(xs).unsqueeze(0),
        "ys": torch.tensor(ys).unsqueeze(0),
        "covs":  torch.tensor(normalized_cov_matrix).unsqueeze(0).unsqueeze(0),
        "sizes": torch.tensor([1.0]).unsqueeze(0),
        }
    return blob


def get_blob_score_from_blob_dict(blob, score_size=64):
    blob_score = splat_features(**blob,
                                score_size=score_size,
                                return_d_score=True,
                                )[0]
    return blob_score


def composite_mask_and_image(mask, image, masked_color=[0,0,0]):
    if isinstance(mask, Image.Image):
        mask_np = np.array(mask)
    else:
        mask_np = mask
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    if mask_np.ndim == 2:
        mask_indicator = (mask_np>0).astype(np.uint8)
    else:

        mask_indicator = (mask_np.sum(-1)>255).astype(np.uint8)
    masked_image = image_np * (1-mask_indicator[:,:,np.newaxis]) + masked_color * mask_indicator[:,:,np.newaxis]
    return Image.fromarray(masked_image.astype(np.uint8)).convert("RGB")


def inference_function(
                original_image,
                scene_prompt, 
                ellipse_mask_path, 
                object_white_background_center_path,
                edited_ellipse_masked_image_path, 
                ellipse_lists, 
                blobnet_control_strength, 
                blobnet_control_guidance_start,
                blobnet_control_guidance_end,
                remove_blob_box,
                num_samples, 
                seed,
                guidance_scale,
                num_inference_steps,
                pipeline,
                device,
                ):
    
    
    generator = torch.Generator(device=device).manual_seed(seed)

    ## prepare img: object_region_center, edited_background_region
    validation_object_region_center = Image.open(object_white_background_center_path)
    ori_ellipse_mask = Image.open(ellipse_mask_path)
    width, height = validation_object_region_center.size
    latent_height, latent_width = height // 8, width // 8


    if not remove_blob_box:
        validation_edited_background_region = Image.open(edited_ellipse_masked_image_path)
        ## prepare gs_score
        final_ellipse = ellipse_lists[-1]
        mean, cov_matrix = get_gs_from_ellipse(final_ellipse)
        normalized_mean, normalized_cov_matrix = normalize_gs(mean, cov_matrix, width, height)
        blob_dict = get_blob_dict_from_norm_gs(normalized_mean, normalized_cov_matrix)   
        validation_gs_score = get_blob_score_from_blob_dict(blob_dict, score_size=(latent_height, latent_width)).unsqueeze(0).to(device) # bnhw
    else:
        img_tmp = original_image.copy()
        validation_edited_background_region = composite_mask_and_image(ori_ellipse_mask, img_tmp, masked_color=[255,255,255])
        ## prepare gs_score
        start_ellipse = ellipse_lists[0]
        mean, cov_matrix = get_gs_from_ellipse(start_ellipse)
        normalized_mean, normalized_cov_matrix = normalize_gs(mean, cov_matrix, width, height)
        blob_dict = get_blob_dict_from_norm_gs(normalized_mean, normalized_cov_matrix)
        validation_gs_score = get_blob_score_from_blob_dict(blob_dict, score_size=(latent_height, latent_width)).unsqueeze(0).to(device) # bnhw
        validation_gs_score[:,0] = 1.0
        validation_gs_score[:,1] = 0.0
        final_ellipse = start_ellipse
        ## set blobnet control strength to 0.0
        blobnet_control_strength = 0.0

    with torch.autocast("cuda"):
        output = pipeline(
                fg_image=validation_object_region_center,
                bg_image=validation_edited_background_region,
                gs_score=validation_gs_score,
                generator=generator,
                prompt=[scene_prompt]*num_samples,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                blobnet_control_guidance_start=float(blobnet_control_guidance_start),
                blobnet_control_guidance_end=float(blobnet_control_guidance_end),
                blobnet_conditioning_scale=float(blobnet_control_strength),
                width=width,
                height=height,
                return_sample=False,
                )
        edited_images = output.images

    edit_image_plots = []
    for i in range(num_samples):
        edit_image_np = np.array(edited_images[i])
        edit_image_np_plot = cv2.ellipse(edit_image_np, final_ellipse, [0,255,0], 3)
        edit_image_plot = Image.fromarray(edit_image_np_plot).convert("RGB")
        edit_image_plots.append(edit_image_plot)

    results_gallery = [*edited_images, *edit_image_plots]

    torch.cuda.empty_cache()
    return results_gallery

def construct_pipeline(args, device, weight_dtype):
    ## load models and pipeline
    blobnet_path = args.blobnet_path
    unet_lora_path = args.unet_lora_path
    stabel_diffusion_model_path = args.stabel_diffusion_model_path
    dinov2_path = args.dinov2_path

    ## unet
    print(f"Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        stabel_diffusion_model_path, 
        subfolder="unet", 
    )
    with torch.no_grad():
        initial_input_channels = unet.config.in_channels
        new_conv_in = torch.nn.Conv2d(
            initial_input_channels + 1,
            unet.conv_in.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=unet.conv_in.bias is not None,
            dtype=unet.dtype,
            device=unet.device,
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :initial_input_channels].copy_(unet.conv_in.weight)
        if unet.conv_in.bias is not None:
            new_conv_in.bias.copy_(unet.conv_in.bias)
        unet.conv_in = new_conv_in

    ## blobnet
    print(f"Loading BlobNet...")
    blobnet = BlobNetModel.from_pretrained(blobnet_path, ignore_mismatched_sizes=True)

    ## dinov2
    print(f"Loading Dinov2...")
    dinov2_processor = AutoImageProcessor.from_pretrained(dinov2_path)
    dinov2 = Dinov2Model.from_pretrained(dinov2_path).to(device)

    print(f"Loading StableDiffusionBlobNetPipeline...")
    pipeline = StableDiffusionBlobNetPipeline.from_pretrained(
            stabel_diffusion_model_path,
            unet=unet,
            blobnet=blobnet,
            torch_dtype=weight_dtype,
            dinov2_processor=dinov2_processor,
            dinov2=dinov2,
    )
    print(f"Loading UNetLora...")
    pipeline.load_lora_weights(
        unet_lora_path,
        adapter_name="default",
    )
    pipeline.set_adapters(["default"])

    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    # pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.to(device)
    pipeline.set_progress_bar_config(leave=False)

    return pipeline




def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--original_image", type=str, default="./assets/results/demo/move_hat/input_image/input_image.png")
    parser.add_argument("--scene_prompt", type=str, default="A frog sits on a rock in a pond, with a top hat beside it, surrounded by butterflies and vibrant flowers.")
    parser.add_argument("--ellipse_mask_path", type=str, default="./assets/results/demo/move_hat/ori_result_gallery/ori_result_gallery_3.png")
    parser.add_argument("--object_white_background_center_path", type=str, default="./assets/results/demo/move_hat/object_image_gallery/validation_object_region_center.png")
    parser.add_argument("--edited_ellipse_masked_image_path", type=str, default="./assets/results/demo/move_hat/edited_result_gallery/edited_result_gallery_0.png")
    parser.add_argument(
    "--ellipse_lists",
    type=list,
    default=[
        [[227.1067, 118.8526], [85.4812, 103.6543], 87.3739],
        [[361.1067, 367.8526], [85.4812, 103.6543], 87.3739]
    ],
    help="List of ellipses, each as [[x, y], [a, b], angle]",
    )

    parser.add_argument("--blobnet_control_strength", type=float, default=1.0)
    parser.add_argument("--blobnet_control_guidance_start", type=float, default=0.0)
    parser.add_argument("--blobnet_control_guidance_end", type=float, default=0.9)
    parser.add_argument("--remove_blob_box", type=bool, default=False)
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1248464818)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)

    parser.add_argument("--save_dir", type=str, default="./assets/results/inference")

    parser.add_argument("--blobnet_path", type=str, default="./models/blobnet")
    parser.add_argument("--unet_lora_path", type=str, default="./models/unet_lora")
    parser.add_argument("--stabel_diffusion_model_path", type=str, default="./models/stable-diffusion-v1-5")
    parser.add_argument("--dinov2_path", type=str, default="./models/dinov2-large")

    return parser.parse_args()




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16
    args = args_parser()

    pipeline = construct_pipeline(args, device, weight_dtype)

    original_image = Image.open(args.original_image)
    scene_prompt = args.scene_prompt
    ellipse_mask_path = args.ellipse_mask_path
    object_white_background_center_path = args.object_white_background_center_path
    edited_ellipse_masked_image_path = args.edited_ellipse_masked_image_path
    ellipse_lists = args.ellipse_lists
    
    blobnet_control_strength= args.blobnet_control_strength
    blobnet_control_guidance_start= args.blobnet_control_guidance_start
    blobnet_control_guidance_end= args.blobnet_control_guidance_end
    remove_blob_box= args.remove_blob_box
    num_samples= args.num_samples
    seed = args.seed
    guidance_scale= args.guidance_scale
    num_inference_steps= args.num_inference_steps

    results_gallery = inference_function(
            original_image,
            scene_prompt, 
            ellipse_mask_path, 
            object_white_background_center_path,
            edited_ellipse_masked_image_path, 
            ellipse_lists, 
            blobnet_control_strength, 
            blobnet_control_guidance_start,
            blobnet_control_guidance_end,
            remove_blob_box,
            num_samples, 
            seed,
            guidance_scale,
            num_inference_steps,
            pipeline, 
            device)
    
    uuid = str(uuid.uuid4())
    save_path = f"{args.save_dir}/{uuid}"
    os.makedirs(save_path, exist_ok=True)
    for i, result in enumerate(results_gallery):
        result.save(os.path.join(save_path, f"inference_result_{i}.png"))
    original_image.save(os.path.join(save_path, "original_image.png"))
    edited_ellipse_masked_image = Image.open(edited_ellipse_masked_image_path)
    edited_ellipse_masked_image.save(os.path.join(save_path, "edited_ellipse_masked_image.png"))
    ellipse_mask = Image.open(ellipse_mask_path)
    ellipse_mask.save(os.path.join(save_path, "ellipse_mask.png"))
    object_white_background_center = Image.open(object_white_background_center_path)
    object_white_background_center.save(os.path.join(save_path, "object_white_background_center.png"))
    print(f"Inference results saved to {save_path}")