import hydra
from omegaconf import DictConfig, OmegaConf
from src.models.architectures.hrnet import get_pose_net
import sys

def to_hrnet_cfg(cfg: DictConfig):
    return OmegaConf.create({
        'MODEL': {
            'NUM_JOINTS': cfg.num_joints,
            'INIT_WEIGHTS': False,
            'EXTRA': {
                'FINAL_CONV_KERNEL': cfg.final_conv_kernel,
                'STAGE2': {k.upper(): v for k, v in cfg.stage2.items()},
                'STAGE3': {k.upper(): v for k, v in cfg.stage3.items()},
                'STAGE4': {k.upper(): v for k, v in cfg.stage4.items()}
            }
        } 
    })

@hydra.main(version_base=None, config_path="configs/model/hrnet", config_name="hrnet_w48")
def main(cfg: DictConfig):
    hrnet_cfg = to_hrnet_cfg(cfg)
    model = get_pose_net(hrnet_cfg, is_train=False)
    
    total_params = sum(p.numel() for p in model.parameters())
    model_name = "HRNet-W48"
    
    print(f"{model_name}:")
    print(f"Parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    print("=== Parameters ===")
    for name, param in model.named_parameters():
        print(name, param.shape)
if __name__ == "__main__":
    main()