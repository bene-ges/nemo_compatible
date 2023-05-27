#!/bin/bash
NEMO_PATH=NeMo
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

## Download dataset from https://www.kaggle.com/datasets/google/userlibri
## Note that initial UserLibri audio files are in .flac format. 
## Use this command to convert them to .wav:
##    for i in */*.flac; do ffmpeg -i "$i" -ac 1 -ar 16000 "${i%.*}.wav"; done

## At this moment your ${WORKDIR} folder structure should look like this:
## data_folder_example
##   corpus
##    ├── audio_data
##    |    ├── test-clean
##    |    |    ├── speaker-1089-book-4217
##    |    |    |    ├── 1089-134686-0000.wav
##    |    |    |    |   ...
##    |    |    |    ├── 1089-134686-0037.wav
##    |    |    |    ├── 1089-134686.trans.txt      # 1089-134686-0001 STUFF IT INTO YOU HIS BELLY COUNSELLED HIM
##    |    |    |    ├── 1089-134691-0000.wav
##    |    |    |    |   ...
##    |    |    |    ├── 1089-134691-0025.wav
##    |    |    |    └── 1089-134691.trans.txt      
##    |    |    |   ...
##    |    |    └── speaker-908-book-574       
##    |    ├── test-other
##    |    └── metadata.tsv      # User ID, Split, Num Audio Examples, Average Words Per Example
##    └── lm_data
##        ├── 10136_lm_train.txt         # CALF'S HEAD A LA MAITRE D'HOTEL
##        ├── 1041_lm_train.txt
##        |   ...
##        └── metadata.tsv    # Book ID, Num Text Examples, Average Words Per Example

## Make manifest file in NeMo format and imitate custom vocabularies.
mkdir ${WORKDIR}/vocabs
mkdir ${WORKDIR}/manifests
python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/evaluation/preprocess_userlibri_and_create_vocabs.py \
  --input_folder ${WORKDIR}/corpus \
  --destination_folder ${WORKDIR}/vocabs \
  --output_manifest ${WORKDIR}/manifests/manifest.json \
  --idf_file ${IDF} \
  --min_idf 8.0 \
  --min_len 6

## Transcribe data by NeMo model Conformer-CTC Large
python ${NEMO_PATH}/examples/asr/transcribe_speech.py \
  pretrained_name="stt_en_conformer_ctc_large" \
  dataset_manifest=${WORKDIR}/manifests/manifest.json \
  output_filename=${WORKDIR}/manifests/manifest_ctc_tmp.json \
  batch_size=16

## Merge multiple spaces (only occur in Conformer CTC).
python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/evaluation/combine_multiple_space_in_manifest.py \
  --input_manifest ${WORKDIR}/manifests/manifest_ctc_tmp.json \
  --output_manifest ${WORKDIR}/manifests/manifest_ctc.json

## Transcribe data by NeMo model Conformer-Transducer Large
python ${NEMO_PATH}/examples/asr/transcribe_speech.py \
  pretrained_name="stt_en_conformer_transducer_large" \
  dataset_manifest=${WORKDIR}/manifests/manifest.json \
  output_filename=${WORKDIR}/manifests/manifest_transducer.json \
  batch_size=16

## Get WER of baseline CTC model
python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=${WORKDIR}/manifests/manifest_ctc.json \
  use_cer=False \
  only_score_manifest=True

## Get WER of baseline Transducer model
python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=${WORKDIR}/manifests/manifest_transducer.json \
  use_cer=False \
  only_score_manifest=True

for ASRTYPE in "ctc" "transducer"
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

## If you want to test separately on test-clean and test-other you can grep corresponding lines from manifest, e.g.
##     grep "/test-clean/" ${WORKDIR}/manifests/manifest_ctc.json > ${WORKDIR}/manifests/manifest_ctc_test_clean.json
##     grep "/test-other/" ${WORKDIR}/manifests/manifest_ctc.json > ${WORKDIR}/manifests/manifest_ctc_test_other.json
