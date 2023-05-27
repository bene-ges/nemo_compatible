#!/bin/bash

## git clone https://github.com/NVIDIA/NeMo NeMo
NEMO_PATH=NeMo
## git clone https://github.com/NVIDIA/NeMo-text-processing NeMo-text-processing
NEMO_TEXT_PROCESSING_PATH=NeMo-text-processing
## git clone https://github.com/bene-ges/nemo_compatible nemo_compatible
NEMO_COMPATIBLE_PATH=nemo_compatible

git clone https://huggingface.co/datasets/bene-ges/spellmapper_en_evaluation
git clone https://huggingface.co/bene-ges/spellmapper_asr_customization_en

## Files in model repo  
PRETRAINED_MODEL=spellmapper_asr_customization_en/training_10m_5ep.nemo
NGRAM_MAPPINGS=spellmapper_asr_customization_en/replacement_vocab_filt.txt
BIG_SAMPLE=spellmapper_asr_customization_en/big_sample.txt

## File with IDF (inverse document frequencies) for words and short phrases.
## It is generated during dataset_preparation/build_training_data.sh
IDF=spellmapper_en_evaluation/idf.txt

WORKDIR=`pwd`

## To get the dataset, you need to fill the form at
##     https://datasets.kensho.com/datasets/spgispeech
## You will get an email with download links, from which you need only validation part
##    https://datasets.kensho.com/api/v1/download/val.tar.bz2?key=... -o val.tar.bz2
##    https://datasets.kensho.com/api/v1/download/val.csv.bz2?key=... -o val.csv.bz2

## At this moment your ${WORKDIR} folder structure should look like this:
##  data_folder_example
##   ├── spgispeech
##   │   └── val
##   |      ├── 0018ad922e541b415ae60e175160b976
##   |      |    ├── 118.wav
##   |      |    ├── 120.wav
##   |      |    ├── 148.wav
##   |      |    ├── 21.wav
##   |      |    ├── 34.wav
##   |      |    ├── 3.wav
##   |      |    ├── 72.wav
##   |      |    └── 92.wav
##   |      ├── ...
##   |      └── 002d12258ff802d65f79ae2eef99e4ab
##   └── val.csv

## val.csv contains paths to audio and transcripts
##   wav_filename|wav_filesize|transcript
##   13aa6c0669adb5544a0d62beef677189/12.wav|333164|Daniel, how do we think about the importance of those assets in the total remedy package? I mean, if you look at the last book value of those -- of the remedy assets and then in payments you've taken,
##   13aa6c0669adb5544a0d62beef677189/22.wav|323564|it should be clear that the intention is that the investments in Uttam Galva will be a part of the joint venture and funded by the joint venture.
##   13aa6c0669adb5544a0d62beef677189/75.wav|198764|next year. If you think back to our Q2 results, we did announce at that point that

## Extract text trascripts
awk 'BEGIN {FS="|"} (NR > 1){print $3}' < ${WORKDIR}/val.csv > ${WORKDIR}/text.txt

## Convert digits to words etc, but keep original case and punctuation
python ${NEMO_TEXT_PROCESSING_PATH}/nemo_text_processing/text_normalization/normalize.py \
  --input_file ${WORKDIR}/text.txt \
  --output_file ${WORKDIR}/norm.txt

## Make manifest file in NeMo format and imitate custom vocabularies.
mkdir ${WORKDIR}/vocabs
mkdir ${WORKDIR}/manifests
python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/evaluation/preprocess_kensho_and_create_vocabs.py \
  --input_folder ${WORKDIR}/spgispeech/val \
  --destination_folder ${WORKDIR}/vocabs \
  --transcription_file ${WORKDIR}/val.csv \
  --normalized_file ${WORKDIR}/norm.txt \
  --output_manifest ${WORKDIR}/manifests/manifest.json \
  --idf_file ${IDF} \
  --min_idf_uppercase 5.0 \
  --min_idf_lowercase 8.0 \
  --min_len 6

## Transcribe data by NeMo model Conformer-CTC Large
python ${NEMO_PATH}/examples/asr/transcribe_speech.py \
  pretrained_name="stt_en_conformer_ctc_large" \
  dataset_manifest=${WORKDIR}/manifests/manifest.json \
  output_filename=${WORKDIR}/manifests/manifest_ctc.json \
  batch_size=16

## Transcribe data by NeMo model Conformer-Transducer Large 
python ${NEMO_PATH}/examples/asr/transcribe_speech.py \
  pretrained_name="stt_en_conformer_transducer_large" \
  dataset_manifest=${WORKDIR}/manifests/manifest.json \
  output_filename=${WORKDIR}/manifests/manifest_transducer.json \
  batch_size=16

## Merge multiple spaces (only occur in Conformer CTC).
python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/evaluation/combine_multiple_space_in_manifest.py \
  --input_manifest ${WORKDIR}/manifests/manifest_ctc.json \
  --output_manifest ${WORKDIR}/manifests/manifest_ctc2.json

## Remove all occurences of "um" and "uh" from transcriptions, because they are absent in the Kensho reference text.
python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/evaluation/remove_uh_um_from_manifest.py \
  --input_manifest ${WORKDIR}/manifests/manifest_ctc2.json \
  --output_manifest ${WORKDIR}/manifests/manifest_ctc_without_uh_um.json

python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/evaluation/remove_uh_um_from_manifest.py \
  --input_manifest ${WORKDIR}/manifests/manifest_transducer.json \
  --output_manifest ${WORKDIR}/manifests/manifest_transducer_without_uh_um.json

## Get WER of baseline CTC model
python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=${WORKDIR}/manifests/manifest_ctc_without_uh_um.json \
  use_cer=False \
  only_score_manifest=True

## Get WER of baseline Transducer model
python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=${WORKDIR}/manifests/manifest_transducer_without_uh_um.json \
  use_cer=False \
  only_score_manifest=True

for ASRTYPE in "ctc_without_uh_um" "transducer_without_uh_um"
do

    ## Split ASR output transcriptions into shorter fragments to serve as ASR hypotheses for spellchecking model
    mkdir ${WORKDIR}/hypotheses_${ASRTYPE}
    python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/evaluation/extract_asr_hypotheses.py \
      --manifest ${WORKDIR}/manifests/manifest_${ASRTYPE}.json \
      --folder ${WORKDIR}/hypotheses_${ASRTYPE}

    ## Prepare inputs for inference of neural customization spellchecking model
    mkdir ${WORKDIR}/spellchecker_input_${ASRTYPE}
    mkdir ${WORKDIR}/spellchecker_output_${ASRTYPE}
    python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/evaluation/prepare_input_for_spellchecker_inference.py \
      --hypotheses_folder ${WORKDIR}/hypotheses_${ASRTYPE} \
      --vocabs_folder ${WORKDIR}/vocabs \
      --output_folder ${WORKDIR}/spellchecker_input_${ASRTYPE} \
      --ngram_mappings ${NGRAM_MAPPINGS} \
      --big_sample_file ${BIG_SAMPLE}

    ## Create filelist with input filenames
    find ${WORKDIR}/spellchecker_input_${ASRTYPE}/*.txt | grep -v "info.txt" > ${WORKDIR}/filelist.txt

    ## Run inference with neural customization spellchecking model
    python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/spellchecking_asr_customization_infer.py \
      pretrained_model=${PRETRAINED_MODEL} \
      model.max_sequence_len=512 \
      +inference.from_filelist=${WORKDIR}/filelist.txt \
      +inference.output_folder=${WORKDIR}/spellchecker_output_${ASRTYPE} \
      inference.batch_size=16 \
      lang=en

    ## Postprocess and combine spellchecker results into a single manifest
    python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/evaluation/update_transcription_with_spellchecker_results.py \
      --asr_hypotheses_folder ${WORKDIR}/hypotheses_${ASRTYPE} \
      --spellchecker_results_folder ${WORKDIR}/spellchecker_output_${ASRTYPE} \
      --input_manifest ${WORKDIR}/manifests/manifest_${ASRTYPE}.json \
      --output_manifest ${WORKDIR}/manifests/manifest_${ASRTYPE}_corrected.json \
      --ngram_mappings ${NGRAM_MAPPINGS} \
      --min_dp_score_per_symbol -1.5

    ## Check WER of spellchecker results
    python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
      dataset_manifest=${WORKDIR}/manifests/manifest_${ASRTYPE}_corrected.json \
      use_cer=False \
      only_score_manifest=True

    ## Perform error analysis and create "ideal" spellchecker results for comparison
    python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/evaluation/analyze_custom_ref_vs_asr.py \
      --manifest ${WORKDIR}/manifests/manifest_${ASRTYPE}_corrected.json \
      --vocab_dir ${WORKDIR}/vocabs \
      --input_dir ${WORKDIR}/spellchecker_input_${ASRTYPE} \
      --ngram_mappings ${NGRAM_MAPPINGS} \
      --output_name ${WORKDIR}/${ASRTYPE}_analysis_ref_vs_asr.txt

    ## Check WER of "ideal" spellcheck results
    python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
      dataset_manifest=${WORKDIR}/${ASRTYPE}_analysis_ref_vs_asr.txt.ideal_spellcheck \
      use_cer=False \
      only_score_manifest=True
done
