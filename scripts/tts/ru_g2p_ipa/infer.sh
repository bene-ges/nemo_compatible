#!/bin/bash

## path to your local NeMo repository (https://github.com/NVIDIA/NeMo)
NEMO_PATH=NeMo

## download model from HuggingFace
git clone https://huggingface.co/bene-ges/ru_g2p_ipa_bert_large

## path to checkpoints that we downloaded
G2P_MODEL=ru_g2p_ipa_bert_large/ru_g2p.nemo

INPUT_FILE=input.txt
OUTPUT_FILE=output.txt

## create sample input file. Note that letters should be separated by space. 
echo "и с х о д" > ${INPUT_FILE}
echo "т р а н с н е п т у н о в ы х" >> ${INPUT_FILE}
echo "т е л я т н и к о в с к о е" >> ${INPUT_FILE}
echo "ц а р с к о г о" >> ${INPUT_FILE}
echo "к р о с х о ф" >> ${INPUT_FILE}
echo "г а н с - ю р г е н" >> ${INPUT_FILE}
echo "д а р д а н е л л" >> ${INPUT_FILE}

## run g2p
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/normalization_as_tagging_infer.py \
  pretrained_model=${G2P_MODEL} \
  inference.from_file=${INPUT_FILE} \
  inference.out_file=${OUTPUT_FILE} \
  model.max_sequence_len=128 \
  inference.batch_size=128 \
  lang=ru
