#!/bin/bash -l
conda activate nemo

## This is a full recipe for training Russian TTS on RUSLAN corpus.
## It uses a g2p model to converts text to IPA-like phonetic representation (some words stay as graphemes).
## We train FastPitch as spectrogram generator and HifiGAN as vocoder.
## Note that RUSLAN corpus (https://ruslan-corpus.github.io/) is available under CC BY-NC-SA 4.0 license (non-commercial).

## path to your local NeMo repository (https://github.com/NVIDIA/NeMo)
NEMO_PATH=NeMo
## path to your local nemo_compatible repository (https://github.com/bene-ges/nemo_compatible)
NEMO_COMPATIBLE_PATH=nemo_compatible

## Download corpus from http://dataset.sova.ai/SOVA-TTS/ruslan/ruslan_dataset.tar

## Download g2p model
git clone https://huggingface.co/bene-ges/ru_g2p_ipa_bert_large

## Extract only plain text. Delete information about stress(+), keep case and punctuation.
awk 'BEGIN {FS="|"}{print $2}' < marks.txt | tr -d "+" > marks.plain.txt

## Extract only paths to .wav files. They will be used later, coupled with preprocessed texts, to create a manifest.
awk 'BEGIN {FS="|"}{print $1}' < marks.txt > marks.audio.txt

## Extract all unique words in lowercase, split to separate letters. This will be input to g2p.
python ${NEMO_COMPATIBLE_PATH}/scripts/tts/ru_g2p_ipa/extract_unique_words_from_text_for_tts.py \
  --input_name marks.plain.txt \
  --output_name all_words.txt \
  --split_to_letters

## Run g2p model
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/normalization_as_tagging_infer.py \
  pretrained_model=ru_g2p_ipa_bert_large/ru_g2p.nemo \
  inference.from_file=all_words.txt \
  inference.out_file=all_words.g2p.txt \
  model.max_sequence_len=64 \
  inference.batch_size=512 \
  lang=ru

## Substitute words in input text with their phonematic representations (output of g2p).
## Known representations (g2p_correct_vocab.txt) override predictions od g2p model.
## If word is ambiguous (heteronym), it is left as is (graphemes). Random word ocurrences are also kept as is with keep_grapheme_ratio.
python ${NEMO_COMPATIBLE_PATH}/scripts/tts/ru_g2p_ipa/preprocess_text_before_tts.py \
  --input_name marks.plain.txt \
  --output_name marks.g2p.txt \
  --g2p_name all_words.g2p.txt \
  --g2p_correct_name ru_g2p_ipa_bert_large/g2p_correct_vocab.txt \
  --heteronyms_name ru_g2p_ipa_bert_large/heteronyms.txt \
  --keep_grapheme_ratio 0.15

## Combine audio paths, preprocessed and original text into a manifest
python ${NEMO_COMPATIBLE_PATH}/scripts/tts/utils/create_manifest_for_tts.py \
  --orig_text_name marks.plain.txt \
  --preprocessed_text_name marks.g2p.txt \
  --audio_paths_name marks.audio.txt \
  --output_name manifest.json

## This scripts loops through the dataset, generates folder sup_data and extracts pitch parameters. Those will be passed to training.
## In yaml config we override alphabet for BaseCharsTokenizer
## Note that this scripts works slow (don't know why)
python ${NEMO_PATH}/scripts/dataset_processing/tts/extract_sup_data.py \
  --config-path ${NEMO_COMPATIBLE_PATH}/scripts/tts/ru_ipa_fastpitch_hifigan/ds_conf \
  --config-name ds_for_fastpitch_align.yaml \
  manifest_filepath=manifest.json \
  sup_data_path=sup_data \
  ++dataloader_params.num_workers=8

## Gives
## PITCH_MEAN=120.87886047363281, PITCH_STD=43.99850845336914
## PITCH_MIN=65.4063949584961, PITCH_MAX=2080.94970703125

## Split to train and validation
head -n 21000 manifest.json > train_manifest.json
tail -n 460 manifest.json > val_manifest.json

## Train FastPitch (mel spectrogram generator model)
## In yaml config we override alphabet for BaseCharTokenizer in the same way as we did in the previous step.
## Note that we insert pitch_mean and pitch_std that we got on previous step, and sup_data_path.
python ${NEMO_PATH}/examples/tts/fastpitch.py \
  --config-path ${NEMO_COMPATIBLE_PATH}/scripts/tts/ru_ipa_fastpitch_hifigan/conf \
  --config-name fastpitch_align_22050_grapheme.yaml \
  model.train_ds.dataloader_params.batch_size=16 \
  model.validation_ds.dataloader_params.batch_size=16 \
  train_dataset=train_manifest.json \
  validation_datasets=val_manifest.json \
  sup_data_path=sup_data \
  exp_manager.exp_dir=experiments \
  trainer.devices=8 \
  trainer.max_epochs=2000 \
  trainer.check_val_every_n_epoch=50 \
  pitch_mean=120.88 \
  pitch_std=44.0 \
  exp_manager.resume_if_exists=false

## !!! Attention: your final FastPitch checkpoint will be somewhere in nemo_experiments/FastPitch/<datetime>/checkpoints/FastPitch.nemo
## Or you can take any intermediate checkpoint with a name like "FastPitch--val_loss=0.7476-epoch=599.ckpt" and convert it to .nemo
## using
## python ${NEMO_COMPATIBLE_PATH}/scripts/tts/utils/fastpitch_ckpt_to_nemo.py \
##  --config-path ${NEMO_COMPATIBLE_PATH}/scripts/tts/ru_ipa_fastpitch_hifigan/conf \
##  --config-name fastpitch_align_22050_grapheme.yaml \
##  +checkpoint_path=FastPitch599.ckpt \
##  +target_nemo_path=FastPitch.nemo

## Fill path to the FastPitch.nemo model, you got
FASTPITCH_MODEL=""

## Generate mel spectrograms from training data using our new fastpitch checkpoint.
## This is recommended by recipes for HifiGAN training 
python ${NEMO_PATH}/scripts/dataset_processing/tts/generate_mels.py \
    --cpu \
    --fastpitch-model-ckpt ${FASTPITCH_MODEL} \
    --input-json-manifests train_manifest.json val_manifest.json \
    --output-json-manifest-root ./

## Finetune HifiGAN model on our mel spectrograms, starting from English checkpoint (tts_en_hifigan).
## Note, that we switch off the scheduler and set learning rate to a small value.
## Note that validation loss can increase during training, you need to listen to checkpoints periodically (once in a day or so) 
python ${NEMO_PATH}/examples/tts/hifigan_finetune.py \
    --config-path ${NEMO_PATH}/examples/tts/conf/hifigan \
    --config-name hifigan.yaml \
    model.max_steps=200000 \
    model.optim.lr=0.00001 \
    ~model.optim.sched \
    train_dataset=train_manifest_mel.json \
    validation_datasets=val_manifest_mel.json \
    exp_manager.exp_dir=experimentsHifigan \
    +init_from_pretrained_model=tts_en_hifigan \
    +trainer.val_check_interval=100 \
    trainer.check_val_every_n_epoch=null \
    model/train_ds=train_ds_finetune \
    model/validation_ds=val_ds_finetune \
    exp_manager.resume_if_exists=false

## !!! Attention: your final HifiGAN checkpoint will be somewhere in nemo_experiments/HifiGan/<datetime>/checkpoints/HifiGan.nemo
## Or you can take any intermediate checkpoint with a name like "HifiGan--val_loss=0.86-epoch=76.ckpt" and convert it to .nemo
## using
## python ${NEMO_COMPATIBLE_PATH}/scripts/tts/utils/hifigan_ckpt_to_nemo.py \
##  --config-path ${NEMO_PATH}/examples/tts/conf/hifigan \
##  --config-name hifigan.yaml \
##  +checkpoint_path=HifiGan76.ckpt \
##  +target_nemo_path=HifiGan.nemo

## That's it! Now you can test your TTS engine (g2p + FastPitch + HifiGan), see test.sh
