"""
This script converts FastPitch intermediate .ckpt checkpoint to .nemo file.

Please set config path and name via command line arguments by `--config-path=CONFIG_FILE_PATH', `--config-name=CONFIG_FILE_NAME'.
Usage example:
python fastpitch_ckpt_to_nemo.py \
  --config-path ../ru_ipa_fastpitch_hifigan/conf \
  --config-name fastpitch_align_22050_grapheme.yaml \
  +checkpoint_path=FastPitch599.ckpt \
  +target_nemo_path=FastPitch.nemo
"""

from omegaconf import DictConfig, OmegaConf

from nemo.collections.tts.models import FastPitchModel
from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path="conf/ru", config_name="fastpitch_align_22050_mix")
def main(cfg: DictConfig) -> None:
    logging.debug(f'Config Params: {OmegaConf.to_yaml(cfg)}')
    FastPitchModel.load_from_checkpoint(cfg.checkpoint_path).save_to(cfg.target_nemo_path)


if __name__ == "__main__":
    main()
