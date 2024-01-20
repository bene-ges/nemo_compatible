# Useful things that work with NVIDIA NeMo library

## Tutorials
- `notebooks/Russian_TTS_with_IPA_G2P_FastPitch_and_HifiGAN.ipynb` - inference pipeline for Russian TTS (G2P + FastPitch + HifiGAN) loading pretrained models from HuggingFace

## Recipes
- `scripts/tts/ru_ipa_fastpitch_hifigan/train.sh` - recipe for data preparation and training of Russian TTS (FastPitch + HifiGAN), using external G2P model.
- `scripts/nlp/en_spellmapper` - Spellchecking model for English ASR Customization.

## Utilities
- `scripts/tts/en_g2p_cmu/infer.sh` - example on how to run inference with English G2P model (converts to phonemes in CMU format).
- `scripts/tts/ru_g2p_ipa/infer.sh` - example on how to run inference with Russian G2P model (converts to IPA-like phonemes).
