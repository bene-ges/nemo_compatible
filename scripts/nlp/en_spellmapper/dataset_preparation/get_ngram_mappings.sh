NEMO_PATH=NeMo

ALIGNMENT_DIR=align
## path to GIZA++ and mkcls binaries
GIZA_BIN_DIR=giza-pp/GIZA++-v2
MCKLS_BINARY=giza-pp/mkcls-v2/mkcls

## Install GIZA++ if you don't have it
# git clone https://github.com/moses-smt/giza-pp.git giza-pp
# cd giza-pp
# make
# cd ..

mkdir ${ALIGNMENT_DIR}
python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/dataset_preparation/prepare_input_for_giza.py \
  --input_manifest pred_ctc.all.json \
  --output_name giza_input.txt \
  --out_dir=${ALIGNMENT_DIR} \
  --giza_dir=${GIZA_BIN_DIR} \
  --mckls_binary=${MCKLS_BINARY}

awk 'BEGIN {FS="\t"} {print $1}' < giza_input.txt > ${ALIGNMENT_DIR}/src
awk 'BEGIN {FS="\t"} {print $2}' < giza_input.txt > ${ALIGNMENT_DIR}/dst
chmod +x ${ALIGNMENT_DIR}/run.sh

## Run Giza++ alignment
cd ${ALIGNMENT_DIR}
./run.sh
cd ..

python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/dataset_preparation/prepare_corpora_after_alignment.py \
  --mode=extract_alignments \
  --input_name=${ALIGNMENT_DIR} \
  --output_name=${ALIGNMENT_DIR}/align.out

python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/dataset_preparation/prepare_corpora_after_alignment.py \
  --mode=get_replacement_vocab \
  --input_name=${ALIGNMENT_DIR}/align.out \
  --output_name=replacement_vocab_full.txt

awk 'BEGIN {FS="\t"}($3 / $4 > 0.005){print $0}' < replacement_vocab_full.txt > replacement_vocab_filt.txt

## Now we have a vocabulary of n-gram mappings that looks like this.
## Format: original n-gram, misspelled n-gram, joint frequency, frequency of original n-gram, frequency of misspelled n-gram
## Using these frequencies we can get "translation probability", for example, p("autu"|"auto") = 69/3790 
## a u t o    o+t <DELETE> t o        49      3790    60
## a u t o    a u t o 2103    3790    2264
## a u t o    a u t u 69      3790    304
## a u t o    a u t o+_       172     3790    177
## a u t o    o <DELETE> t o  67      3790    1406


python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/dataset_preparation/prepare_corpora_after_alignment.py \
  --mode=get_sub_misspells \
  --input_name=${ALIGNMENT_DIR}/align.out \
  --output_name=sub_misspells.txt

## This gives a vocabulary of aligned subphrases (consisting of adjacent words from original phrases)
## Format: original subphrase, misspelled subphrase, joint frequency, frequency of original subphrase, frequency of misspelled subphrase
## domchor dahmer  2       6       6
## domchor dummer  2       6       23
## domchor dammer  1       6       20
## domchor domer   1       6       9

## This file will be used later during synthetic data generation to use not only Wikipedia titles as whole phrases, but also their parts.

awk 'BEGIN {FS="\t"}{print $1 "\t" $4}' < sub_misspells.txt > big_sample.txt

