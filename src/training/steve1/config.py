import torch
import cv2
import os

# Project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

MINECLIP_CONFIG = {
    'arch': "vit_base_p16_fz.v2.t2",
    'hidden_dim': 512,
    'image_feature_dim': 512,
    'mlp_adapter_spec': 'v0-2.t0',
    'pool_type': "attn.d2.nh8.glusw",
    'resolution': [160, 256],
    'ckpt': {
        'path': os.path.join(DATA_DIR, "weights/mineclip/attn.pth"),
        'checksum': 'b5ece9198337cfd117a3bfbd921e56da'
    }
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PRIOR_INFO = {
    'mineclip_dim': 512,
    'latent_dim': 512,
    'hidden_dim': 512,
    'model_path': os.path.join(DATA_DIR, 'weights/steve1/steve1_prior.pt'),
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
