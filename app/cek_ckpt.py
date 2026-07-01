import torch
import sys

ckpt = torch.load(r'D:\seagrass-detection-system\SeagrassLive-Detect\app\models\effdet-d0_0.5366.ckpt', map_location='cpu')
raw_sd = ckpt['state_dict']

new_sd = {k[len('predictor.model.'):]: v 
          for k, v in raw_sd.items() 
          if k.startswith('predictor.model.')}

from effdet import get_efficientdet_config, EfficientDet
from effdet.efficientdet import HeadNet

config = get_efficientdet_config('tf_efficientdet_d0')
config.num_classes = 7
config.image_size = [512, 512]

net = EfficientDet(config, pretrained_backbone=False)
net.class_net = HeadNet(config, num_outputs=7)

model_keys = set(net.state_dict().keys())
ckpt_keys  = set(new_sd.keys())

missing_in_ckpt = model_keys - ckpt_keys
print(f"Missing ({len(missing_in_ckpt)}):")
for k in sorted(missing_in_ckpt)[:10]:
    print(f"  MODEL: {k}")
    base = k.replace('.conv.conv.', '.conv.').replace('.conv.conv_pw.', '.conv.')
    if base in ckpt_keys:
        print(f"    → ADA di ckpt sebagai: {base}")
    else:
        parts = k.split('.')
        candidates = [ck for ck in ckpt_keys if parts[-2] in ck and parts[-1] in ck]
        if candidates:
            print(f"    → Kandidat: {candidates[0]}")
        else:
            print(f"    → TIDAK ADA kandidat")