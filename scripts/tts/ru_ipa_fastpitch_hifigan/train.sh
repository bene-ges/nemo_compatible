NEMO_PATH=NeMo

##reuse config from thorsten_neutral, override tokenizer
python ${NEMO_PATH}/scripts/dataset_processing/tts/extract_sup_data.py \
  --config-path ${NEMO_PATH}/scripts/dataset_processing/tts/thorsten_neutral/ds_conf \
  --config-name ds_for_fastpitch_align.yaml \
  manifest_filepath=manifest.json \
  sup_data_path=sup_data \
  ~dataset.text_normalizer \
  ~dataset.text_normalizer_call_kwargs \
  dataset.text_tokenizer._target_=nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.RussianCharsTokenizer \
  ++dataloader_params.num_workers=8

## Gives
## PITCH_MEAN=120.87886047363281, PITCH_STD=43.99850845336914
## PITCH_MIN=65.4063949584961, PITCH_MAX=2080.94970703125

head -n 21000 manifest.json > train_manifest.json
tail -n 460 manifest.json > val_manifest.json

##reuse config from deutsch, override tokenizer
python ${NEMO_PATH}/examples/tts/fastpitch.py \
    --config-path ${NEMO_PATH}/examples/tts/conf/de \
    --config-name fastpitch_align_22050_grapheme.yaml \
    model.train_ds.dataloader_params.batch_size=16 \
    model.validation_ds.dataloader_params.batch_size=16 \
    ~model.text_normalizer \
    ~model.text_normalizer_call_kwargs \
    model.text_tokenizer._target_=nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers.RussianCharsTokenizer \
    train_dataset=train_manifest.json \
    validation_datasets=val_manifest.json \
    sup_data_path=sup_data \
    exp_manager.exp_dir=experiments \
    trainer.devices=4 \
    trainer.max_epochs=2000 \
    trainer.check_val_every_n_epoch=25 \
    pitch_mean=120.88 \
    pitch_std=44.0

python ${NEMO_PATH}/scripts/dataset_processing/tts/generate_mels.py \
    --cpu \
    --fastpitch-model-ckpt fastpitch.nemo \
    --input-json-manifests train_manifest.json val_manifest.json \
    --output-json-manifest-root ./

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
    model/validation_ds=val_ds_finetune
