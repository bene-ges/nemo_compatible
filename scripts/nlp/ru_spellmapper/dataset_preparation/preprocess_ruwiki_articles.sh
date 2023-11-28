#!/bin/bash -l
conda activate nemo

NEMO_PATH=NeMo
NEMO_COMPATIBLE_LAB_PATH=nemo_compatible_lab

## Download kaikki.org-dictionary-Russian.json from here
##  https://kaikki.org/dictionary/Russian/index.html

## Preprocessing
python ${NEMO_COMPATIBLE_LAB_PATH}/scripts/tts/ru_g2p_ipa/preprocess_kaikki.py \
  --input_name kaikki.org-dictionary-Russian.json \
  --output_name kaikki.txt

## LEMMATIZATION
python ${NEMO_COMPATIBLE_LAB_PATH}/scripts/nlp/ru_spellmapper/dataset_preparation/get_word_endings.py \
  --input_name kaikki.txt \
  --output_name endings.txt

awk 'BEGIN {FS="\t"}($3 >= 10){print $0}' < endings.txt > endings10.txt

