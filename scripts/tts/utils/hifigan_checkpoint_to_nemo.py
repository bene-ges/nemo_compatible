"""
This script converts HifiGan intermediate .ckpt checkpoint to .nemo file.

Please set config path and name via command line arguments by `--config-path=CONFIG_FILE_PATH', `--config-name=CONFIG_FILE_NAME'.
Usage example:
python hifigan_ckpt_to_nemo.py \
  --config-path ${NEMO_PATH}/examples/tts/conf/hifigan \
  --config-name hifigan.yaml \
  +checkpoint_path=HifiGan76.ckpt \
  +target_nemo_path=HifiGan.nemo

"""

from omegaconf import DictConfig, OmegaConf

from nemo.collections.tts.models import HifiGanModel
from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path="conf/hifigan", config_name="hifigan.yaml")
def main(cfg: DictConfig) -> None:
    logging.debug(f'Config Params: {OmegaConf.to_yaml(cfg)}')
    HifiGanModel.load_from_checkpoint(cfg.checkpoint_path).save_to(cfg.target_nemo_path)


if __name__ == "__main__":
    main()
