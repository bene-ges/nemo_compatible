#!/bin/bash -l
conda activate nemo

## path to your local NeMo repository (https://github.com/NVIDIA/NeMo)
NEMO_PATH=NeMo
## path to your local nemo_compatible repository (https://github.com/bene-ges/nemo_compatible)
NEMO_COMPATIBLE_PATH=nemo_compatible

# download models from HuggingFace
git clone https://huggingface.co/bene-ges/ru_g2p_ipa_bert_large
git clone https://huggingface.co/bene-ges/tts_ru_ipa_fastpitch_ruslan
git clone https://huggingface.co/bene-ges/tts_ru_hifigan_ruslan

# paths to checkpoints that we downloaded
G2P_MODEL=ru_g2p_ipa_bert_large/ru_g2p.nemo
FASTPITCH_MODEL=tts_ru_ipa_fastpitch_ruslan/tts_ru_ipa_fastpitch_ruslan.nemo
HIFIGAN_MODEL=tts_ru_hifigan_ruslan/tts_ru_hifigan_ruslan.nemo

# define input and output names
INPUT_NAME=test_input.txt
OUTPUT_DIR=out
OUTPUT_MANIFEST=out_manifest.json

# recreate output folder where generated .wav-files will be stored
rm -r ${OUTPUT_DIR}
mkdir ${OUTPUT_DIR}

# prepare input for g2p
python ${NEMO_COMPATIBLE_PATH}/scripts/tts/ru_g2p_ipa/extract_unique_words_from_text_for_tts.py \
  --input_name ${INPUT_NAME} \
  --output_name ${INPUT_NAME}.words \
  --split_to_letters

# run g2p
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/normalization_as_tagging_infer.py \
  pretrained_model=${G2P_MODEL} \
  inference.from_file=${INPUT_NAME}.words \
  inference.out_file=${INPUT_NAME}.words.g2p \
  model.max_sequence_len=512 \
  inference.batch_size=128 \
  lang=ru

# prepare phonematic input for tts, using g2p results
python ${NEMO_COMPATIBLE_PATH}/scripts/tts/ru_g2p_ipa/preprocess_text_before_tts.py \
  --input_name ${INPUT_NAME} \
  --output_name ${INPUT_NAME}.phonematic \
  --g2p_name ${INPUT_NAME}.words.g2p \
  --g2p_correct_name ru_g2p_ipa_bert_large/g2p_correct_vocab.txt \
  --heteronyms_name ru_g2p_ipa_bert_large/heteronyms.txt

# run tts
python ${NEMO_COMPATIBLE_PATH}/scripts/tts/tts_infer.py \
  --input_name=${INPUT_NAME}.phonematic \
  --output_dir=${OUTPUT_DIR} \
  --output_manifest=${OUTPUT_MANIFEST} \
  --spec_generator=${FASTPITCH_MODEL} \
  --vocoder=${HIFIGAN_MODEL}

# That's it! The generated .wav files are in the output folder, paths and input text are stored in output manifest.
