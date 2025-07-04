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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8, 0)
    # Enable memory efficient attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except:
        pass

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# -----------------------------------------------------------------------------
# HOTFIX: Keras-3 removed backend.is_tensor and backend.is_keras_tensor which
# older versions of einops expect.  This shim re-adds no-op implementations so
# einops' Keras backend no longer crashes during backend detection.
# -----------------------------------------------------------------------------
try:
    import keras
    import types
    kb = getattr(keras, 'backend', None)
    if kb is None:
        # In Keras-3 the backend submodule is generated lazily; create placeholder
        kb = types.ModuleType('backend')
        keras.backend = kb  # type: ignore
        import sys
        sys.modules['keras.backend'] = kb
    # Add missing attributes if necessary
    if not hasattr(kb, 'is_tensor'):
        kb.is_tensor = lambda x: False  # type: ignore
    if not hasattr(kb, 'is_keras_tensor'):
        kb.is_keras_tensor = lambda x: False  # type: ignore
    # Expose same helpers at top module level (einops also asks keras.*)
    if not hasattr(keras, 'is_tensor'):
        keras.is_tensor = kb.is_tensor  # type: ignore
    if not hasattr(keras, 'is_keras_tensor'):
        keras.is_keras_tensor = kb.is_keras_tensor  # type: ignore
except Exception:
    # If keras is not installed nothing to patch.
    pass

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
            # Use half precision more aggressively for speed
            try:
                model = model.half()  # Convert entire model to half precision
                print("DEBUG: Model converted to half precision for faster inference")
            except RuntimeError as e:
                print(f"Full half precision failed: {e}, trying partial conversion")
                try:
                    if hasattr(model, 'diffusion_model'):
                        model.diffusion_model = model.diffusion_model.half()
                    if hasattr(model, 'first_stage_model'):
                        model.first_stage_model = model.first_stage_model.half()
                    if hasattr(model, 'cond_stage_model'):
                        model.cond_stage_model = model.cond_stage_model.half()
                except RuntimeError as e2:
                    print(f"Partial half precision failed: {e2}, keeping float32")
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
    """
    Run human parsing using CIHP_PGN model
    Returns True if successful, False otherwise
    """
    try:
        # Change to CIHP_PGN directory for imports
        original_cwd = os.getcwd()
        cihp_dir = os.path.join(os.path.dirname(__file__), "..", "CIHP_PGN")
        os.chdir(cihp_dir)
        
        # Add CIHP_PGN to Python path
        sys.path.insert(0, cihp_dir)
        
        logging.info(f"Input image path: {input_image_path}")
        logging.info(f"Output mask path: {output_mask_path}")
        
        # Import after changing directory
        from utils.pgn_keras import PGNKeras
        import numpy as np
        import cv2
        
        # Preprocess image
        def preprocess_image(image_path, target_size=(512, 512)):
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB and swap to BGR for model compatibility
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_b, img_g, img_r = img[:,:,2], img[:,:,1], img[:,:,0]
            img = np.stack([img_b, img_g, img_r], axis=2)
            
            # Resize
            img = cv2.resize(img, target_size)
            
            # Convert to float and normalize
            img = img.astype(np.float32)
            img -= np.array([125.0, 114.4, 107.9])  # IMG_MEAN
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
        
        # Load model
        checkpoint_paths = [
            '/home/paperspace/checkpoint/CIHP_pgn',
            './checkpoint/CIHP_pgn'
        ]
        
        checkpoint_path = None
        for path in checkpoint_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        if not checkpoint_path:
            raise FileNotFoundError("CIHP_PGN checkpoint not found in expected locations")
        
        logging.info(f"Loading model from: {checkpoint_path}")
        model = PGNKeras(n_classes=20, checkpoint_path=checkpoint_path)
        
        # Preprocess image
        image_batch = preprocess_image(input_image_path)
        logging.info(f"[DEBUG] Input batch shape: {image_batch.shape}, dtype: {image_batch.dtype}")
        
        # Run inference
        parsing_fc, parsing_rf_fc, edge_rf_fc = model(image_batch)
        
        # Process outputs
        parsing_out = np.argmax(parsing_fc, axis=-1)
        parsing_np = parsing_out.astype(np.uint8)[0]  # Remove batch dimension
        
        # Debug: Print mask stats
        logging.info(f"[DEBUG] Mask unique values: {np.unique(parsing_np)}")
        logging.info(f"[DEBUG] Mask shape: {parsing_np.shape}, dtype: {parsing_np.dtype}")
        
        # Create output directory
        OUTPUT_DIR = os.path.dirname(output_mask_path)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Save grayscale mask
        cv2.imwrite(output_mask_path, parsing_np)
        logging.info(f"Mask created and copied to: {output_mask_path}")
        
        # Save colorized mask for debugging
        try:
            from utils.utils import decode_labels
            color_mask = decode_labels(parsing_np[np.newaxis, ..., np.newaxis], num_images=1, num_classes=20)[0]
            color_output_path = output_mask_path.replace('.png', '_color.png')
            cv2.imwrite(color_output_path, color_mask)
            logging.info(f"[DEBUG] Colorized mask saved to: {color_output_path}")
        except Exception as e:
            logging.warning(f"Could not save colorized mask: {e}")
        
        return True
        
    except Exception as e:
        logging.error(f"Human parsing failed: {e}")
        return False
    finally:
        # Restore original working directory and clean up path
        os.chdir(original_cwd)
        if cihp_dir in sys.path:
            sys.path.remove(cihp_dir)

"""
TryOn-Adapter ‑ Inference script for VITON-HD single-pair virtual try-on
=======================================================================

This script is an **extended / debug-friendly** version of the original
`test_viton.py`.  It performs a *single* person-cloth inference run and is
intended to be called from the API layer.  The changes compared to upstream
include:

•  EXTRA LOGGING – tons of `DEBUG:` prints so we can see tensor shapes,
   statistics and where something explodes.
•  ROBUST MEMORY MANAGEMENT – falls back to CPU if the GPU runs out of RAM; it
   also enables PyTorch’s `max_split_size_mb` allocator, clears caches, etc.
•  ADAPTER FUSION FIX – the original Embedding-Adapter expects a 28 × 28 latent
   grid (784 tokens).  We therefore *adaptively pool* the VAE latent to the
   right size before calling `model.fuse_adapter()`.
•  MASK WORKAROUNDS – if the provided human-parsing mask is empty, we generate
   a coarse torso rectangle so diffusion still has something to in-paint.

CLI Arguments
-------------

    --person_image      Path to the resized (384×512) person JPG
    --cloth_image       Path to the resized (384×512) clothing JPG
    --mask              Grayscale PNG produced by the human-parsing stage
    --output            Where to write `result.png`
    --config, --ckpt    Stable-Diffusion-based config / checkpoint
    --ckpt_elbm_path    Pre-trained VAE blend-fusion weights directory
    --H, --W            Spatial size used by CP-VTON datasets (defaults 256×192)

The pipeline entry point is `run_single_pair()` – see the docstring right
below for an overview of every stage.
"""
def run_single_pair(person_image_path, cloth_image_path, mask_path, output_path, config_path, ckpt_path, ckpt_elbm_path, device, H=256, W=192):
    """Run a **single** virtual try-on pair end-to-end.

    1.  PRE-CONDITIONING
        a. Load diffusion / VAE / CLIP model (optionally in half-precision).
        b. Build an on-disk *mini-dataset* so `CPDataset` is happy.

    2.  DATA PIPELINE (see the big loop further below):
        ‑ Prepare in-paint tensors, warp features, masks, etc.
        ‑ Convert raw cloth RGB to CLIP + VAE latents.
        ‑ Optionally fuse VAE latent + CLIP tokens through the custom
          `Embedding_Adapter` so cloth matches body shape.
        ‑ Build conditioning dicts for UNet (9-channel in-paint scheme).
        ‑ Run DDIM sampler with classifier-free guidance.
        ‑ Decode to RGB and save *result.png*.

    3.  Every step spits out `DEBUG:` logs with tensor shapes / stats so that
        we can trace crashes (shape mismatches, NaNs, CUDA OOMs …).

    NOTE: the function is extremely long because it contains many
    workarounds.  When hunting for bugs, search for the **DEBUG:** tags – they
    tell you exactly which branch is executing.
    """
    # Ensure functional API is available throughout this function
    import torch.nn.functional as F
    try:
        print("DEBUG: Starting run_single_pair function...")
        
        # Clear GPU memory before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Prepare temp dataset structure
        test_id = "test_001"
        temp_dataset_dir = os.path.dirname(mask_path)  # Should be .../test/image-parse-v3/
        dataset_dir = os.path.dirname(os.path.dirname(temp_dataset_dir))  # .../test/
        dataroot = os.getcwd()  # Use current working directory where test_pairs.txt is located
        
        print(f"DEBUG: Dataset structure:")
        print(f"  temp_dataset_dir: {temp_dataset_dir}")
        print(f"  dataset_dir: {dataset_dir}")
        print(f"  dataroot: {dataroot}")
        
        # All files should already be present: person, cloth, cloth-mask, mask

        # Check all required files
        assert os.path.exists(person_image_path), f"Person image not found: {person_image_path}"
        assert os.path.exists(cloth_image_path), f"Cloth image not found: {cloth_image_path}"
        assert os.path.exists(mask_path), f"Mask not found: {mask_path}"
        print("DEBUG: All input files exist")

        # Run the pipeline (reuse main logic, but for this dataset)
        print(f"DEBUG: Loading config from {config_path}")
        config = OmegaConf.load(f"{config_path}")
        print(f"DEBUG: Loading model from {ckpt_path}")
        model = load_model_from_config(config, f"{ckpt_path}")
        print("DEBUG: Model loaded successfully")
        
    except Exception as e:
        print(f"ERROR in setup: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Clear memory after model loading
    clear_gpu_memory()
    
    try:
        print("DEBUG: Creating dataset...")
        # FIXED: Use the output directory structure created by the API provider
        # The API provider creates: output_dir/test/ with proper subdirectories
        # and output_dir/test_pairs.txt
        potential_output_dir = os.path.dirname(output_path)
        test_pairs_path = os.path.join(potential_output_dir, "test_pairs.txt")
        
        if os.path.exists(test_pairs_path):
            # Use the structured dataset created by API provider
            dataroot = potential_output_dir
            print(f"DEBUG: Using structured dataset at {dataroot}")
        else:
            # Fallback to current directory (for command line usage)
            dataroot = os.getcwd()
            print(f"DEBUG: Using fallback dataroot at {dataroot}")
        
        dataset = CPDataset(dataroot, H, mode='test', unpaired=True)
        print(f"DEBUG: Dataset created with {len(dataset)} samples")
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False)
        print("DEBUG: DataLoader created successfully")
    except Exception as e:
        print(f"ERROR in dataset creation: {e}")
        import traceback
        traceback.print_exc()
        raise
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
        try:
            print("DEBUG: Starting inference loop...")
            for data in loader:
                print("DEBUG: Processing batch...")
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
                
                # CRITICAL FIX: Ensure new_mask has correct channels for latent space
                # The DDIM sampler needs new_mask to have 4 channels to match latent space
                if new_mask.shape[1] == 3:  # If it's RGB (3 channels)
                    # Convert to single channel by taking the mean
                    new_mask_single = torch.mean(new_mask, dim=1, keepdim=True)
                    # Expand to 4 channels for latent space compatibility  
                    new_mask = new_mask_single.repeat(1, 4, 1, 1)
                    print(f"DEBUG: Fixed new_mask channels from 3 to 4 for latent space compatibility")
                elif new_mask.shape[1] == 1:  # If it's grayscale (1 channel)
                    # Expand to 4 channels for latent space compatibility
                    new_mask = new_mask.repeat(1, 4, 1, 1)
                    print(f"DEBUG: Expanded new_mask from 1 to 4 channels for latent space compatibility")
                
                print(f"DEBUG: new_mask final shape: {new_mask.shape}")
                
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
                    
                                    # OPTIMIZED: Moderate amplification for faster convergence
                print(f"DEBUG: Original c_clip stats - min: {c_clip.min()}, max: {c_clip.max()}, std: {c_clip.std()}")
                print(f"DEBUG: Original patches stats - min: {patches.min()}, max: {patches.max()}, std: {patches.std()}")
                
                # OPTIMIZED amplification for faster convergence
                c_clip = c_clip * 1.8  # Reduced from 2.5x for speed
                patches = patches * 1.5  # Reduced from 2.2x for speed
                
                print(f"DEBUG: OPTIMIZED c_clip stats - min: {c_clip.min()}, max: {c_clip.max()}, std: {c_clip.std()}")
                print(f"DEBUG: OPTIMIZED patches stats - min: {patches.min()}, max: {patches.max()}, std: {patches.std()}")
                
                # Fuse the features (this might need the fuse_adapter method)
                if hasattr(model, 'fuse_adapter'):
                    try:
                        print(f"DEBUG: c_vae shape before adapter: {c_vae.shape}")
                        print(f"DEBUG: patches shape before adapter: {patches.shape}")
                        
                        # IMPROVED: Adapter fusion with proper shape handling
                        # The adapter expects specific input dimensions
                        TARGET_HW = (28, 28)  # (H, W) expected by Embedding_Adapter
                        if c_vae.dim() == 4 and c_vae.shape[-2:] != TARGET_HW:
                            c_vae = F.adaptive_avg_pool2d(c_vae, TARGET_HW)
                            print(f"DEBUG: Pooled c_vae to {TARGET_HW} for adapter compatibility")
                        original_c_vae_shape = c_vae.shape
                        original_patches_shape = patches.shape
                        
                        # Ensure c_vae is in correct format [B, C, H, W]
                        if c_vae.dim() == 2:  # [B, Features] -> need to reshape
                            # Calculate spatial dimensions from feature count
                            B, total_features = c_vae.shape
                            # Common VAE latent dimensions
                            spatial_size = int((total_features // 4) ** 0.5)
                            if spatial_size * spatial_size * 4 == total_features:
                                c_vae = c_vae.view(B, 4, spatial_size, spatial_size)
                                print(f"DEBUG: Reshaped c_vae from {original_c_vae_shape} to {c_vae.shape}")
                        
                        # Ensure patches is in correct format [B, Tokens, Features]
                        if patches.dim() == 2:  # [B*Tokens, Features] -> [B, Tokens, Features]
                            B = 1  # We know batch size is 1
                            tokens = patches.shape[0] // B
                            features = patches.shape[1]
                            patches = patches.view(B, tokens, features)
                            print(f"DEBUG: Reshaped patches from {original_patches_shape} to {patches.shape}")
                        
                        # Verify dimensions are compatible before fusion
                        if c_vae.dim() == 4 and patches.dim() == 3:
                            print(f"DEBUG: Final shapes - c_vae: {c_vae.shape}, patches: {patches.shape}")
                            patches = model.fuse_adapter(patches, c_vae)
                            print("DEBUG: ✅ Adapter fusion successful - clothing should now adapt to body shape!")
                        else:
                            print(f"DEBUG: Shape mismatch - c_vae.dim()={c_vae.dim()}, patches.dim()={patches.dim()}")
                            print("DEBUG: Skipping adapter fusion due to incompatible shapes...")
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
                    del c_vae
                except:
                    pass
                del c_clip, patches, c_proj, patches_proj
                clear_gpu_memory()

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
                
                # IMPROVED: Use smaller batch processing to avoid OOM
                torch.backends.cudnn.benchmark = False  # Disable for memory efficiency
                
                # Set memory management
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # Set conservative memory split
                    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
                
                # Encode inpaint_image to latent space with memory management
                with torch.no_grad():  # Ensure no gradients
                    z_inpaint = model.encode_first_stage(inpaint_image)
                    z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
                
                clear_gpu_memory()  # Clear memory after encoding inpaint_image
                
                # Encode feat_tensor to get the proper latent dimensions with memory management
                with torch.no_grad():  # Ensure no gradients
                    warp_feat_encoded = model.encode_first_stage(feat_tensor)
                    warp_feat_encoded = model.get_first_stage_encoding(warp_feat_encoded).detach()
                
                # Use the encoded warp_feat shape for proper latent space dimensions
                latent_spatial_shape = warp_feat_encoded.shape[-2:]
                z_inpaint_resized = F.interpolate(z_inpaint, size=latent_spatial_shape, mode='bilinear', align_corners=False)
                
                # For the UNet input, keep mask as single channel and resize to latent space
                mask_single_channel = mask_tensor if mask_tensor.shape[1] == 1 else mask_tensor[:, :1, :, :]  # Ensure single channel
                
                # OPTIMIZED: More permissive mask validation for CIHP_PGN compatibility
                mask_mean = mask_single_channel.mean()
                mask_max = mask_single_channel.max()
                mask_nonzero = (mask_single_channel > 0).float().mean()
                
                print(f"DEBUG: Mask validation - mean: {mask_mean:.4f}, max: {mask_max:.4f}, nonzero: {mask_nonzero:.4f}")
                
                # IMPROVED: More permissive validation for CIHP_PGN masks
                # CIHP_PGN produces segmentation masks with values 0-19 (normalized to 0-1)
                if mask_nonzero < 0.001 or mask_max < 0.02:  # Much more permissive
                    print("DEBUG: Human parsing mask is invalid/empty, creating SIMPLE REPLACEMENT mask...")
                    B, C, H, W = mask_single_channel.shape
                    enhanced_mask = torch.zeros_like(mask_single_channel)
                    
                    # SIMPLE coverage for faster processing
                    torso_top = int(0.15 * H)      # Standard coverage
                    torso_bottom = int(0.75 * H)   # Standard coverage
                    torso_left = int(0.20 * W)     # Standard coverage
                    torso_right = int(0.80 * W)    # Standard coverage
                    
                    # Create simple torso region
                    enhanced_mask[:, :, torso_top:torso_bottom, torso_left:torso_right] = 1.0
                    
                    # Simple Gaussian blur
                    enhanced_mask = gauss(enhanced_mask)
                    enhanced_mask = (enhanced_mask > 0.3).float()
                    
                    mask_single_channel = enhanced_mask
                    print(f"DEBUG: SIMPLE REPLACEMENT mask stats - min: {mask_single_channel.min()}, max: {mask_single_channel.max()}, mean: {mask_single_channel.mean()}")
                else:
                    # We have a valid human parsing mask, use IMPROVED processing
                    print("DEBUG: Using VALID human parsing mask with improved processing...")
                    
                    # IMPROVED: Better handling of CIHP_PGN masks
                    # Normalize mask values to 0-1 range if needed
                    if mask_max > 1.0:
                        print("DEBUG: Normalizing mask values to 0-1 range")
                        mask_single_channel = mask_single_channel / mask_max
                    
                    # IMPROVED: Focus on torso/upper body regions (classes 5,6,7 in CIHP_PGN)
                    # If the mask has multiple classes, focus on clothing areas
                    unique_values = torch.unique(mask_single_channel)
                    print(f"DEBUG: Mask unique values: {unique_values}")
                    
                    if len(unique_values) > 2:  # Multi-class segmentation
                        print("DEBUG: Processing multi-class segmentation mask...")
                        # Focus on upper body clothing regions
                        clothing_mask = torch.zeros_like(mask_single_channel)
                        
                        # CIHP_PGN classes: 5=Upper-clothes, 6=Dress, 7=Coat, 4=Jumpsuits
                        clothing_classes = [4/19, 5/19, 6/19, 7/19]  # Normalized values
                        for class_val in clothing_classes:
                            tolerance = 0.02
                            class_pixels = torch.abs(mask_single_channel - class_val) < tolerance
                            clothing_mask[class_pixels] = 1.0
                        
                        if clothing_mask.sum() > 0:
                            mask_single_channel = clothing_mask
                            print(f"DEBUG: Extracted clothing regions: {clothing_mask.sum()} pixels")
                        else:
                            print("DEBUG: No clothing classes found, using all non-background")
                            mask_single_channel = (mask_single_channel > 0.02).float()
                    else:
                        print("DEBUG: Processing binary mask...")
                        mask_single_channel = (mask_single_channel > 0.02).float()
                    
                    # Simple inversion check - if most of the image is masked, invert
                    if mask_single_channel.mean() > 0.7:
                        print("DEBUG: Inverting mask - most of image was masked!")
                        mask_single_channel = 1.0 - mask_single_channel
                    
                    # Simple smoothing
                    mask_single_channel = gauss(mask_single_channel)
                    mask_single_channel = (mask_single_channel > 0.1).float()
                    
                    # IMPROVED: Moderate expansion with clothing-aware kernel
                    print("DEBUG: Applying clothing-aware mask expansion...")
                    kernel = torch.ones(1, 1, 11, 11, device=mask_single_channel.device) / 121  # Smaller, more precise
                    mask_single_channel = F.conv2d(mask_single_channel, kernel, padding=5)
                    mask_single_channel = (mask_single_channel > 0.2).float()
                    print(f"DEBUG: IMPROVED mask stats - min: {mask_single_channel.min()}, max: {mask_single_channel.max()}, mean: {mask_single_channel.mean()}")
                
                # OPTIMIZED: Simple coverage check for faster processing
                mask_coverage = mask_single_channel.mean()
                if mask_coverage < 0.15:  # Simple threshold
                    print(f"⚠️  WARNING: Low coverage {mask_coverage*100:.1f}%, applying simple expansion...")
                    
                    # Simple expansion
                    B, C, H, W = mask_single_channel.shape
                    simple_mask = torch.zeros_like(mask_single_channel)
                    simple_top = int(0.15 * H)
                    simple_bottom = int(0.75 * H)
                    simple_left = int(0.20 * W)
                    simple_right = int(0.80 * W)
                    simple_mask[:, :, simple_top:simple_bottom, simple_left:simple_right] = 1.0
                    
                    # Blend with existing mask
                    mask_single_channel = torch.maximum(mask_single_channel, simple_mask)
                    print(f"🔧 SIMPLE EXPANSION - coverage: {mask_single_channel.mean()*100:.1f}%")
                    
                elif mask_coverage > 0.8:
                    print(f"⚠️  WARNING: High coverage {mask_coverage*100:.1f}%, reducing...")
                    mask_single_channel = mask_single_channel * 0.8
                    mask_single_channel = (mask_single_channel > 0.3).float()
                    print(f"🔧 REDUCED - coverage: {mask_single_channel.mean()*100:.1f}%")
                else:
                    print(f"✅ GOOD: Coverage {mask_coverage*100:.1f}% is acceptable")
                
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
                
                # Create proper unconditional conditioning for guidance (CRITICAL FIX!)
                print("DEBUG: Creating unconditional conditioning for proper guidance...")
                # Create unconditional conditioning using empty/zero cloth features
                try:
                    # Create zero cloth input for unconditional conditioning
                    zero_cloth = torch.zeros_like(c)
                    zero_cloth_resized = F.interpolate(zero_cloth, size=(224, 224), mode='bilinear', align_corners=False)
                    uc_clip, uc_patches = model.get_learned_conditioning(clip_normalize(zero_cloth_resized))
                    uc_proj = model.proj_out(uc_clip)
                    uc_patches_proj = model.proj_out_patches(uc_patches)
                    uc = torch.cat([uc_proj, uc_patches_proj], dim=1)
                    del zero_cloth, zero_cloth_resized, uc_clip, uc_patches, uc_proj, uc_patches_proj
                except Exception as e:
                    print(f"DEBUG: Failed to create unconditional conditioning: {e}")
                    print("DEBUG: Using zeros_like as fallback...")
                    uc = torch.zeros_like(c_encoded)
                
                # CRITICAL FIX: Use random noise instead of q_sample from warped features
                # This prevents the model from just reconstructing the input
                print("DEBUG: Using random noise initialization instead of q_sample...")
                start_code = torch.randn_like(warp_feat_encoded) * 0.8
                # Optionally blend with some original signal but heavily weighted toward noise
                # start_code = 0.8 * torch.randn_like(warp_feat_encoded) + 0.2 * model.q_sample(warp_feat_encoded, ts)
                # Calculate proper latent shape based on the actual encoded dimensions
                actual_latent_shape = warp_feat_encoded.shape[-2:]
                shape = [4, actual_latent_shape[0], actual_latent_shape[1]]
                print(f"DEBUG: Calculated shape for sampling: {shape}")
                # Debug print shapes before sampling
                print("DEBUG: start_code shape:", start_code.shape)
                print("DEBUG: z_inpaint_resized shape:", test_model_kwargs['inpaint_image'].shape)
                print("DEBUG: mask_latent_resized shape:", test_model_kwargs['inpaint_mask'].shape)
                print("DEBUG: warp_feat_encoded shape:", warp_feat_encoded.shape)
                print("DEBUG: Total input channels will be:", start_code.shape[1] + test_model_kwargs['inpaint_image'].shape[1] + test_model_kwargs['inpaint_mask'].shape[1])
                print("DEBUG: Cloth conditioning shape:", c_encoded.shape)
                print("DEBUG: Unconditional conditioning shape:", uc.shape)
                
                # Dynamically choose sampling steps based on environment or device
                sampling_env = os.getenv("SAMPLING_STEPS")
                if sampling_env is not None and sampling_env.isdigit():
                    sampling_steps = int(sampling_env)
                else:
                    sampling_steps = 30 if torch.cuda.is_available() else 50  # quality on CPU
                eta_value = 0.0  # Deterministic for faster convergence

                # Slightly higher guidance for sharper results
                if torch.cuda.is_available():
                    guidance_scale = 7.5

                print(f"DEBUG: SAMPLING parameters:")
                print(f"  ⚡ Guidance scale: {guidance_scale}")
                print(f"  🎯 Sampling steps: {sampling_steps}")
                print(f"  🌟 Eta value: {eta_value}")
                
                # OPTIMIZED: Fast sampling for reasonable quality
                print(f"🚀 Starting diffusion sampling with {sampling_steps} steps...")
                start_time = time.time()
                
                samples_ddim, _ = sampler.sample(
                    S=sampling_steps, 
                    conditioning=c_encoded, 
                    batch_size=1, 
                    shape=shape, 
                    down_block_additional_residuals=down_block_additional_residuals, 
                    verbose=True, 
                    unconditional_guidance_scale=guidance_scale, 
                    unconditional_conditioning=uc, 
                    eta=eta_value, 
                    x_T=start_code, 
                    use_T_repaint=True, 
                    test_model_kwargs=test_model_kwargs, 
                    **test_model_kwargs
                )
                
                sampling_time = time.time() - start_time
                print(f"✅ Sampling completed in {sampling_time:.2f} seconds")
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
                print(f"DEBUG: samples_ddim shape: {samples_ddim.shape}")
                x_samples_ddim = model.vae.decode(samples_ddim, processed_intermediate_features, int_layers).sample
                print(f"DEBUG: VAE decoded shape: {x_samples_ddim.shape}")
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                x_checked_image = x_samples_ddim
                x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                x_result = x_checked_image_torch
                print(f"DEBUG: Final result shape: {x_result.shape}")
                
                for i, x_sample in enumerate(x_result):
                    save_x = resize(x_sample)
                    save_x = 255. * rearrange(save_x.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(save_x.astype(np.uint8))
                    print(f"DEBUG: Saving image with shape {save_x.shape} to {output_path}")
                    img.save(output_path)
                    print(f"DEBUG: Successfully saved result image to {output_path}")

        except Exception as e:
            print(f"ERROR in inference loop: {e}")
            import traceback
            traceback.print_exc()
            raise
    
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