#!/bin/bash

## path to your local NeMo repository (https://github.com/NVIDIA/NeMo)
NEMO_PATH=NeMo

## download model from HuggingFace
git clone https://huggingface.co/bene-ges/en_g2p_cmu_bert_large

## path to checkpoints that we downloaded
G2P_MODEL=en_g2p_cmu_bert_large/en_g2p.nemo

INPUT_FILE=input.txt
OUTPUT_FILE=output.txt

## create sample input file. Note that letters should be separated by space. 
echo "g e f f e r t" > ${INPUT_FILE}
echo "p r o s c r i b e d" >> ${INPUT_FILE}
echo "p r o m i n e n t l y" >> ${INPUT_FILE}
echo "j o c e l y n" >> ${INPUT_FILE}
echo "m a r c e c a ' s" >> ${INPUT_FILE}
echo "s t a n k o w s k i" >> ${INPUT_FILE}
echo "m u f f l e" >> ${INPUT_FILE}

## run g2p
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/normalization_as_tagging_infer.py \
  pretrained_model=${G2P_MODEL} \
  inference.from_file=${INPUT_FILE} \
  inference.out_file=${OUTPUT_FILE} \
  model.max_sequence_len=128 \
  inference.batch_size=128 \
  lang=en
