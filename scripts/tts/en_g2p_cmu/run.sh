#!/bin/bash

## paths to repositories
NEMO_PATH=NeMo
NEMO_COMPATIBLE_PATH=nemo_compatible

ALIGNMENT_DIR=align
## path to GIZA++ and mkcls binaries
GIZA_BIN_DIR=giza-pp/GIZA++-v2
MCKLS_BINARY=giza-pp/mkcls-v2/mkcls

## Download cmu.txt from here
## https://github.com/NVIDIA/NeMo/blob/main/scripts/tts_dataset_files/cmudict-0.7b_nv22.07

mkdir ${ALIGNMENT_DIR}
python ${NEMO_COMPATIBLE_PATH}/scripts/tts/en_g2p_cmu/preprocess_cmu.py \
  --input_name cmu.txt \
  --output_name cmu.clean \
  --out_dir=${ALIGNMENT_DIR} \
  --giza_dir=${GIZA_BIN_DIR} \
  --mckls_binary=${MCKLS_BINARY}

awk 'BEGIN {FS="\t"} {print $1}' < cmu.clean > ${ALIGNMENT_DIR}/src
awk 'BEGIN {FS="\t"} {print $2}' < cmu.clean > ${ALIGNMENT_DIR}/dst
chmod +x ${ALIGNMENT_DIR}/run.sh

## Run Giza++ alignment
cd ${ALIGNMENT_DIR}
./run.sh
cd ..

python ${NEMO_COMPATIBLE_PATH}/scripts/tts/en_g2p_cmu/extract_giza_alignments.py \
  --giza_dir=${ALIGNMENT_DIR} \
  --out_filename=align.out \
  --giza_suffix=A3.final

python ${NEMO_COMPATIBLE_PATH}/scripts/tts/en_g2p_cmu/prepare_corpora_after_alignment.py \
  --mode=get_replacement_vocab \
  --alignment_filename=${ALIGNMENT_DIR}/align.out \
  --vocab_filename=replacement_vocab_full.txt \
  --out_filename=""

## Restrict vocabulary of tags (tag is a phoneme or a sequence of phonemes)
head -n 170 replacement_vocab_full.txt > replacement_vocab.txt

python ${NEMO_COMPATIBLE_PATH}/scripts/tts/en_g2p_cmu/prepare_corpora_after_alignment.py \
  --mode=filter_by_vocab \
  --alignment_filename=${ALIGNMENT_DIR}/align.out \
  --vocab_filename=replacement_vocab.txt \
  --out_filename=${ALIGNMENT_DIR}/align.out2

## Prepare training data for BERT model whose task is to do G2P conversion in a non-autoregressive manner (as a tagger)
mkdir datasets
mkdir datasets/all
awk 'BEGIN {FS="\t"}{print $1 "\t" $3 "\t"}' < ${ALIGNMENT_DIR}/align.out2 | sort -R > datasets/all.txt
head -n 130000 datasets/all.txt > datasets/all/train.tsv
tail -n 3426 datasets/all.txt > datasets/all/test.tsv
cp datasets/all/test.tsv datasets/all/valid.tsv
awk 'BEGIN {FS="\t"} {print $1}' < datasets/all/valid.tsv > datasets/test.txt

echo "KEEP" > datasets/label_map.txt
echo "DELETE" >> datasets/label_map.txt
awk 'BEGIN {FS="\t"}($1 != "<DELETE>"){print "DELETE|" $1}' < replacement_vocab.txt >> datasets/label_map.txt

echo "PLAIN" > datasets/semiotic_classes.txt
cp datasets/label_map.txt datasets/all/label_map.txt
cp datasets/semiotic_classes.txt datasets/all/semiotic_classes.txt

python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/normalization_as_tagging_train.py \
lang="en" \
data.validation_ds.data_path=datasets/all/valid.tsv \
data.train_ds.data_path=datasets/all/train.tsv \
data.train_ds.batch_size=128 \
data.train_ds.num_workers=8 \
model.language_model.pretrained_model_name=bert-base-uncased \
model.label_map=datasets/all/label_map.txt \
model.semiotic_classes=datasets/all/semiotic_classes.txt \
model.optim.lr=3e-5 \
trainer.devices=[0] \
trainer.num_nodes=1 \
trainer.accelerator=gpu \
trainer.strategy=ddp \
trainer.max_epochs=20 \
