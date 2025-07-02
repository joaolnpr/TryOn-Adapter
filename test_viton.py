import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchvision
from utils.emasc import EMASC

from ldm.data.cp_dataset import CPDataset
from ldm.resizer import Resizer
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.data.deepfashions import DFPairDataset
import torchgeometry as tgm
from torch import nn
from utils.data_utils import mask_features


import clip
from torchvision.transforms import Resize

from torch.nn import functional as F
import subprocess
import shutil
import logging

# Memory management settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.7, 0)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise ValueError("'{}' is not a valid checkpoint path".format(checkpoint_path))
    model.load_state_dict(torch.load(checkpoint_path))

def clear_gpu_memory():
    """Helper function to clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def convert_mask_to_3channel(mask_tensor):
    """Convert single-channel mask to 3-channel by repeating"""
    if mask_tensor.shape[1] == 1:  # If single channel
        mask_tensor = mask_tensor.repeat(1, 3, 1, 1)  # Repeat to make 3 channels
    return mask_tensor

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def gen_noise(shape):
    noise = np.zeros(shape, dtype=np.uint8)
    ### noise
    noise = cv2.randn(noise, 0, 255)
    noise = np.asarray(noise / 255, dtype=np.uint8)
    noise = torch.tensor(noise, dtype=torch.float32)
    return noise

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    try:
        if torch.cuda.is_available():
            # Clear memory before loading to GPU
            clear_gpu_memory()
            model = model.cuda()
            # Use mixed precision more carefully
            try:
                if hasattr(model, 'diffusion_model'):
                    model.diffusion_model = model.diffusion_model.half()
                if hasattr(model, 'first_stage_model'):
                    model.first_stage_model = model.first_stage_model.half()
            except RuntimeError as e:
                print(f"Half precision failed: {e}, keeping float32")
    except RuntimeError as e:
        print(f"CUDA OOM: {e}\nFalling back to CPU.")
        model = model.cpu()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)


def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


def run_human_parser(input_image_path, output_mask_path):
    HOME = os.path.expanduser("~")
    CIHP_PGN_DIR = os.path.join(HOME, "CIHP_PGN")
    CHECKPOINT_DIR = os.path.join(HOME, "checkpoint")
    DATASETS_DIR = os.path.join(HOME, "datasets")
    OUTPUT_DIR = os.path.join(HOME, "output")

    logging.info(f"Input image path: {input_image_path}")
    logging.info(f"Output mask path: {output_mask_path}")
    logging.info(f"CIHP_PGN_DIR: {CIHP_PGN_DIR}")
    logging.info(f"CHECKPOINT_DIR: {CHECKPOINT_DIR}")
    logging.info(f"DATASETS_DIR: {DATASETS_DIR}")
    logging.info(f"OUTPUT_DIR: {OUTPUT_DIR}")

    # Prepare input
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DATASETS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    input_copy_path = os.path.join(DATASETS_DIR, "input.jpg")
    try:
        shutil.copy2(input_image_path, input_copy_path)
        logging.info(f"Copied input image to {input_copy_path}")
    except Exception as e:
        logging.error(f"Failed to copy input image: {e}")
        raise

    # Use conda run to execute in the correct environment
    command = [ "conda", "run", "-n", "cihp_pgn", "python", "test_pgn.py"
    ]
    logging.info(f"Running CIHP_PGN human parser in cihp_pgn environment with command: {' '.join(command)}")
    result = subprocess.run(
        command,
        cwd=CIHP_PGN_DIR,
        capture_output=True,
        text=True
    )
    logging.info(f"STDOUT from parser:\n{result.stdout}")
    if result.returncode != 0:
        logging.error(f"STDERR from parser:\n{result.stderr}")
        raise RuntimeError("CIHP_PGN parser failed!")

    # The output mask will be in OUTPUT_DIR, named 'input.png'
    mask_path = os.path.join(OUTPUT_DIR, "input.png")
    logging.info(f"Looking for mask at {mask_path}")
    if not os.path.exists(mask_path):
        # List files in OUTPUT_DIR for debugging
        logging.error(f"Mask not found at {mask_path}. Files in output dir: {os.listdir(OUTPUT_DIR)}")
        raise FileNotFoundError(f"Parser did not create mask: {mask_path}")

    # Copy mask to TryOn-Adapter temp dataset
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    try:
        shutil.copy2(mask_path, output_mask_path)
        logging.info(f"Mask created and copied to: {output_mask_path}")
    except Exception as e:
        logging.error(f"Failed to copy mask: {e}")
        raise

    # Clean up (optional)
    try:
        os.remove(input_copy_path)
        os.remove(mask_path)
        logging.info("Temporary files cleaned up.")
    except Exception as e:
        logging.warning(f"Cleanup failed: {e}")

def run_single_pair(person_image_path, cloth_image_path, mask_path, output_path, config_path, ckpt_path, ckpt_elbm_path, device, H=256, W=192):
    # Clear GPU memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Prepare temp dataset structure
    test_id = "test_001"
    temp_dataset_dir = os.path.dirname(mask_path)  # Should be .../test/image-parse-v3/
    dataset_dir = os.path.dirname(os.path.dirname(temp_dataset_dir))  # .../test/
    dataroot = os.getcwd()  # Use current working directory where test_pairs.txt is located
    
    # All files should already be present: person, cloth, cloth-mask, mask

    # Check all required files
    assert os.path.exists(person_image_path), f"Person image not found: {person_image_path}"
    assert os.path.exists(cloth_image_path), f"Cloth image not found: {cloth_image_path}"
    assert os.path.exists(mask_path), f"Mask not found: {mask_path}"

    # Run the pipeline (reuse main logic, but for this dataset)
    config = OmegaConf.load(f"{config_path}")
    model = load_model_from_config(config, f"{ckpt_path}")
    
    # Clear memory after model loading
    clear_gpu_memory()
    
    dataset = CPDataset(dataroot, H, mode='test', unpaired=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)
    vae_normalize  = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    clip_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    model.vae.decoder.blend_fusion = nn.ModuleList()
    feature_channels = [ 512, 512, 512, 256, 128]
    for blend_in_ch, blend_out_ch in zip(feature_channels, feature_channels):
        model.vae.decoder.blend_fusion.append(nn.Conv2d(blend_in_ch, blend_out_ch, kernel_size=3, bias=True, padding=1, stride=1))
    model.vae.use_blend_fusion = True
    model.vae.load_state_dict(torch.load(os.path.join(ckpt_elbm_path,'checkpoint/checkpoint-40000/pytorch_model_1.bin')))
    in_feature_channels = [128, 128, 128, 256, 512]
    out_feature_channels = [128, 256, 512, 512, 512]
    int_layers = [1, 2, 3, 4, 5]
    emasc = EMASC(in_feature_channels, out_feature_channels, kernel_size=3, padding=1, stride=1, type='nonlinear')
    emasac_sd = torch.load(os.path.join(ckpt_elbm_path,'emasc_40000.pth'), map_location='cpu')
    emasc.load_state_dict(emasac_sd)
    del emasac_sd  # Free memory
    if torch.cuda.is_available():
        emasc.cuda()
        clear_gpu_memory()  # Clear memory after EMASC loading
    emasc.eval()
    sampler = DDIMSampler(model)
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    if torch.cuda.is_available():
        gauss.cuda()
    with torch.no_grad():
        for data in loader:
            mask_tensor = data['inpaint_mask']
            inpaint_image = data['inpaint_image']
            ref_tensor = data['ref_imgs']
            feat_tensor = data['warp_feat']
            image_tensor = data['GT']
            pose = data['pose']
            sobel_img = data['sobel_img']
            parse_agnostic = data['parse_agnostic']
            warp_mask = data['warp_mask']
            new_mask = warp_mask
            resize = transforms.Resize((H, int(H / 256 * 192)))
            key = 'unpaired' if dataset.unpaired else 'paired'
            cm = data['cloth_mask'][key]
            c = data['cloth'][key]
            test_model_kwargs = {}
            # Helper to move tensors to correct dtype/device for VAE/UNet
            def to_model_dtype(t):
                if next(model.parameters()).is_cuda:
                    return t.half().cuda()
                else:
                    return t.float()
            mask_tensor = to_model_dtype(mask_tensor)
            inpaint_image = to_model_dtype(inpaint_image)
            feat_tensor = to_model_dtype(feat_tensor)
            sobel_img = to_model_dtype(sobel_img)
            parse_agnostic = to_model_dtype(parse_agnostic)
            warp_mask = to_model_dtype(warp_mask)
            new_mask = to_model_dtype(new_mask)
            cm = to_model_dtype(cm)
            c = to_model_dtype(c)
            image_tensor = to_model_dtype(image_tensor)
            pose = to_model_dtype(pose)
            
            # Debug the original cloth tensor format
            print(f"DEBUG: Original cloth tensor shape: {c.shape}")
            print(f"DEBUG: Original cloth tensor dtype: {c.dtype}")
            print(f"DEBUG: Cloth tensor min/max: {c.min()}/{c.max()}")
            
            # Check if the cloth tensor is already processed features or raw image
            if len(c.shape) == 4 and c.shape[1] == 3:
                # This looks like a raw RGB image, needs processing
                print("DEBUG: Cloth appears to be raw image, processing with CLIP+VAE...")
                
                # Process cloth conditioning properly like in the training code
                vae_normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                clip_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                
                # Process with both VAE and CLIP like the original model
                c_vae = model.encode_first_stage(vae_normalize(c))
                c_vae = model.get_first_stage_encoding(c_vae).detach()
                clear_gpu_memory()  # Clear after VAE encoding
                
                # Resize cloth to 224x224 for CLIP processing (CLIP expects this size)
                c_resized = F.interpolate(c, size=(224, 224), mode='bilinear', align_corners=False)
                c_clip, patches = model.get_learned_conditioning(clip_normalize(c_resized))
                del c_resized  # Clean up resized tensor
                clear_gpu_memory()  # Clear after CLIP processing
                
                # Fuse the features (this might need the fuse_adapter method)
                if hasattr(model, 'fuse_adapter'):
                    try:
                        # Reshape VAE features to match adapter expectations
                        print(f"DEBUG: c_vae shape before adapter: {c_vae.shape}")
                        print(f"DEBUG: patches shape before adapter: {patches.shape}")
                        
                        # The adapter expects flattened spatial features
                        # Reshape c_vae from [B, C, H, W] to [B, H*W, C] if needed
                        if len(c_vae.shape) == 4:
                            B, C, H, W = c_vae.shape
                            c_vae_reshaped = c_vae.permute(0, 2, 3, 1).reshape(B, H*W, C)
                        else:
                            c_vae_reshaped = c_vae
                        
                        print(f"DEBUG: c_vae_reshaped shape: {c_vae_reshaped.shape}")
                        patches = model.fuse_adapter(patches, c_vae_reshaped)
                        print("DEBUG: Adapter fusion successful")
                    except Exception as e:
                        print(f"DEBUG: Adapter fusion failed: {e}")
                        print("DEBUG: Continuing without adapter fusion...")
                        # Continue without fusion if it fails
                
                # Project the features
                c_proj = model.proj_out(c_clip)
                patches_proj = model.proj_out_patches(patches)
                
                # Concatenate the projected features
                c_encoded = torch.cat([c_proj, patches_proj], dim=1)
                
                # Clean up intermediate tensors
                try:
                    del c_vae_reshaped
                except:
                    pass
                del c_vae, c_clip, patches, c_proj, patches_proj
                clear_gpu_memory()
            else:
                # This might already be processed features
                print("DEBUG: Cloth appears to be processed features, using as-is...")
                c_encoded = c
            # Prepare ref_tensor_half for VAE/UNet
            if next(model.parameters()).is_cuda:
                ref_tensor_half = ref_tensor.half().cuda()
            else:
                ref_tensor_half = ref_tensor.float()
            # Prepare parse_agnostic_input and sobel_img_input for adapters
            parse_agnostic_input = parse_agnostic.repeat(1, 3, 1, 1)
            sobel_img_input = sobel_img
            if next(model.parameters()).is_cuda:
                parse_agnostic_input = parse_agnostic_input.half().cuda()
                sobel_img_input = sobel_img_input.half().cuda()
            else:
                parse_agnostic_input = parse_agnostic_input.float()
                sobel_img_input = sobel_img_input.float()
            mask_resduial = model.adapter_mask(parse_agnostic_input)
            sobel_resduial = model.adapter_canny(sobel_img_input)
            # Initialize down_block_additional_residuals before appending
            down_block_additional_residuals = []
            for i in range(min(len(mask_resduial), len(sobel_resduial))):
                m = mask_resduial[i]
                s = sobel_resduial[i]
                # Ensure both are 3D (C, H, W)
                if m.dim() > 3:
                    m = m.squeeze(0)
                if s.dim() > 3:
                    s = s.squeeze(0)
                # If spatial shapes don't match, resize s to match m
                if m.shape[-2:] != s.shape[-2:]:
                    s_ = s.unsqueeze(0) if s.dim() == 3 else s
                    s_resized = F.interpolate(s_, size=m.shape[-2:], mode='bilinear', align_corners=False)
                    s = s_resized.squeeze(0)
                # If channel count doesn't match, try to broadcast or raise error
                if m.shape[0] != s.shape[0]:
                    if s.shape[0] == 1 and m.shape[0] > 1:
                        s = s.expand(m.shape[0], *s.shape[1:])
                    elif m.shape[0] == 1 and s.shape[0] > 1:
                        m = m.expand(s.shape[0], *m.shape[1:])
                    else:
                        raise ValueError(f"Cannot match channel count: mask {m.shape}, sobel {s.shape}")
                down_block_additional_residuals.append(torch.cat([m.unsqueeze(0), s.unsqueeze(0)], dim=0))
            # Initial setup with feat_tensor in test_model_kwargs (will be overridden later with latent versions)
            test_model_kwargs['warp_feat'] = feat_tensor
            test_model_kwargs['new_mask'] = new_mask
            # Clear memory before encoding
            clear_gpu_memory()
            
            # Encode inpaint_image to latent space
            z_inpaint = model.encode_first_stage(inpaint_image)
            z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
            
            clear_gpu_memory()  # Clear memory after encoding inpaint_image
            
            # Encode feat_tensor to get the proper latent dimensions
            warp_feat_encoded = model.encode_first_stage(feat_tensor)
            warp_feat_encoded = model.get_first_stage_encoding(warp_feat_encoded).detach()
            
            # Use the encoded warp_feat shape for proper latent space dimensions
            latent_spatial_shape = warp_feat_encoded.shape[-2:]
            z_inpaint_resized = F.interpolate(z_inpaint, size=latent_spatial_shape, mode='bilinear', align_corners=False)
            
            # For the UNet input, keep mask as single channel and resize to latent space
            mask_single_channel = mask_tensor if mask_tensor.shape[1] == 1 else mask_tensor[:, :1, :, :]  # Ensure single channel
            mask_latent_resized = F.interpolate(mask_single_channel, size=latent_spatial_shape, mode='nearest')
            
            # Set up the properly sized latent tensors for diffusion
            test_model_kwargs['inpaint_mask'] = mask_latent_resized  # Single channel for UNet
            test_model_kwargs['inpaint_image'] = z_inpaint_resized   # 4 channels (latent)
            test_model_kwargs['warp_feat'] = feat_tensor
            test_model_kwargs['new_mask'] = new_mask
            test_model_kwargs['x_inpaint'] = z_inpaint
            
            # Verify all required keys are present
            required_keys = ['inpaint_image', 'inpaint_mask', 'warp_feat', 'new_mask']
            for k in required_keys:
                if k not in test_model_kwargs:
                    raise KeyError(f"Missing key '{k}' in test_model_kwargs. Current keys: {list(test_model_kwargs.keys())}")
            
            print(f"DEBUG: Final tensor shapes for diffusion:")
            print(f"  inpaint_image: {test_model_kwargs['inpaint_image'].shape}")
            print(f"  inpaint_mask: {test_model_kwargs['inpaint_mask'].shape}")
            print(f"  Expected total channels for UNet: 4 + 4 + 1 = 9")
            
            # Clear memory after encoding operations
            del feat_tensor  # Now we can delete this as we have the encoded version
            clear_gpu_memory()
            model_device = next(model.parameters()).device
            warp_feat_encoded = warp_feat_encoded.to(model_device)
            ts = torch.full((1,), 999, device=device, dtype=torch.long).to(model_device)
            uc = None
            start_code = model.q_sample(warp_feat_encoded, ts)
            shape = [4, H // 8, W // 8]
            # Debug print shapes before sampling
            print("DEBUG: start_code shape:", start_code.shape)
            print("DEBUG: z_inpaint_resized shape:", test_model_kwargs['inpaint_image'].shape)
            print("DEBUG: mask_latent_resized shape:", test_model_kwargs['inpaint_mask'].shape)
            print("DEBUG: warp_feat_encoded shape:", warp_feat_encoded.shape)
            print("DEBUG: Total input channels will be:", start_code.shape[1] + test_model_kwargs['inpaint_image'].shape[1] + test_model_kwargs['inpaint_mask'].shape[1])
            print("DEBUG: Cloth conditioning shape:", c_encoded.shape)
            samples_ddim, _ = sampler.sample(S=20, conditioning=c_encoded, batch_size=1, shape=shape, down_block_additional_residuals=down_block_additional_residuals, verbose=False, unconditional_guidance_scale=1, unconditional_conditioning=uc, eta=0.0, x_T=start_code, use_T_repaint=True, test_model_kwargs=test_model_kwargs, **test_model_kwargs)
            samples_ddim = 1/ 0.18215 * samples_ddim
            
            # Clear memory after sampling
            del start_code, down_block_additional_residuals, warp_feat_encoded, c, c_encoded
            clear_gpu_memory()
            # Convert im_mask to 3 channels before encoding
            im_mask_3ch = convert_mask_to_3channel(data["im_mask"])
            # Move to the same device as the model (not just check if CUDA is available)
            model_device = next(model.parameters()).device
            print(f"DEBUG: Model is on device: {model_device}")
            im_mask_3ch = im_mask_3ch.to(model_device)
            _, intermediate_features = model.vae.encode(im_mask_3ch)
            del im_mask_3ch  # Clean up
            intermediate_features = [intermediate_features[i] for i in int_layers]
            
            # Ensure emasc is on the same device as the main model
            emasc_device = next(emasc.parameters()).device
            if emasc_device != model_device:
                print(f"DEBUG: Moving emasc from {emasc_device} to {model_device}")
                emasc = emasc.to(model_device)
                clear_gpu_memory()
            
            # Move intermediate features to the correct device
            intermediate_features = [f.to(model_device) for f in intermediate_features]
            
            processed_intermediate_features = emasc(intermediate_features)
            processed_intermediate_features = mask_features(processed_intermediate_features,(1- data["inpaint_mask"]).to(model_device))
            x_samples_ddim = model.vae.decode(samples_ddim, processed_intermediate_features, int_layers).sample
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
            x_checked_image = x_samples_ddim
            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
            x_result = x_checked_image_torch
            for i, x_sample in enumerate(x_result):
                save_x = resize(x_sample)
                save_x = 255. * rearrange(save_x.cpu().numpy(), 'c h w -> h w c')
                img = Image.fromarray(save_x.astype(np.uint8))
                img.save(output_path)
    
    # Final memory cleanup
    clear_gpu_memory()

def main():
    # Clear GPU memory at start and set up memory management
    clear_gpu_memory()
    
    # Set additional memory management for stability
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--person_image', type=str, required=True)
    parser.add_argument('--cloth_image', type=str, required=True)
    parser.add_argument('--mask', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--ckpt_elbm_path', type=str, required=True)
    parser.add_argument('--H', type=int, default=256)
    parser.add_argument('--W', type=int, default=192)
    opt = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_single_pair(opt.person_image, opt.cloth_image, opt.mask, opt.output, opt.config, opt.ckpt, opt.ckpt_elbm_path, device, H=opt.H, W=opt.W)

if __name__ == "__main__":
    main()