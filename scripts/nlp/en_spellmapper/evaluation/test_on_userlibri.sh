#!/bin/bash
NEMO_PATH=NeMo

## Spellchecking model in nemo format, that you get after training. See run_training.sh or run_training_tarred.sh  
PRETRAINED_MODEL=training.nemo

## These two files are generated by dataset_preparation/get_ngram_mappings.sh 
NGRAM_MAPPINGS=replacement_vocab_filt.txt
SUB_MISSPELLS=sub_misspells.txt

## File with IDF (inverse document frequencies) for words and short phrases.
## It is generated during dataset_preparation/build_training_data.sh
IDF=idf.txt

DATA_DIR="data/userlibri"

## Download dataset from https://www.kaggle.com/datasets/google/userlibri
## Note that initial UserLibri audio files are in .flac format. 
## Use this command to convert them to .wav:
##    for i in */*.flac; do ffmpeg -i "$i" -ac 1 -ar 16000 "${i%.*}.wav"; done

## At this moment your ${DATA_DIR} folder structure should look like this:
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
mkdir ${DATA_DIR}/vocabs
mkdir ${DATA_DIR}/manifests
python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/evaluation/preprocess_userlibri_and_create_vocabs.py \
  --input_folder ${DATA_DIR}/corpus \
  --destination_folder ${DATA_DIR}/vocabs \
  --output_manifest ${DATA_DIR}/manifests/manifest.json \
  --idf_file ${IDF} \
  --min_idf 8.0 \
  --min_len 6

## Transcribe data by NeMo model Conformer-CTC Large
python ${NEMO_PATH}/examples/asr/transcribe_speech.py \
  pretrained_name="stt_en_conformer_ctc_large" \
  dataset_manifest=${DATA_DIR}/manifests/manifest.json \
  output_filename=${DATA_DIR}/manifests/manifest_ctc.json \
  batch_size=16

## Transcribe data by NeMo model Conformer-Transducer Large
python ${NEMO_PATH}/examples/asr/transcribe_speech.py \
  pretrained_name="stt_en_conformer_transducer_large" \
  dataset_manifest=${DATA_DIR}/manifests/manifest.json \
  output_filename=${DATA_DIR}/manifests/manifest_transducer.json \
  batch_size=16

## Get CER of baseline CTC model
python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=${DATA_DIR}/manifests/manifest_ctc.json \
  use_cer=True \
  only_score_manifest=True

## Get WER of baseline CTC model
python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=${DATA_DIR}/manifests/manifest_ctc.json \
  use_cer=False \
  only_score_manifest=True

## Get CER of baseline Transducer model
python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=${DATA_DIR}/manifests/manifest_transducer.json \
  use_cer=True \
  only_score_manifest=True

## Get WER of baseline Transducer model
python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=${DATA_DIR}/manifests/manifest_transducer.json \
  use_cer=False \
  only_score_manifest=True

for ASRTYPE in "ctc" "transducer"
do
    ## Split ASR output transcriptions into shorter fragments to serve as ASR hypotheses for spellchecking model
    mkdir ${DATA_DIR}/hypotheses_${ASRTYPE}
    python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/evaluation/extract_asr_hypotheses.py \
      --manifest ${DATA_DIR}/manifests/manifest_${ASRTYPE}.json \
      --folder ${DATA_DIR}/hypotheses_${ASRTYPE}

    ## Prepare inputs for inference of neural customization spellchecking model
    mkdir ${DATA_DIR}/spellchecker_input_${ASRTYPE}
    mkdir ${DATA_DIR}/spellchecker_output_${ASRTYPE}
    python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/evaluation/prepare_input_for_spellchecker_inference.py \
      --hypotheses_folder ${DATA_DIR}/hypotheses_${ASRTYPE} \
      --vocabs_folder ${DATA_DIR}/vocabs \
      --output_folder ${DATA_DIR}/spellchecker_input_${ASRTYPE} \
      --ngram_mappings ${NGRAM_MAPPINGS} \
      --sub_misspells_file ${SUB_MISSPELLS}

    ## Create filelist with input filenames
    find ${DATA_DIR}/spellchecker_input_${ASRTYPE}/*.txt | grep -v "info.txt" > ${DATA_DIR}/filelist.txt

    ## Run inference with neural customization spellchecking model
    python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/spellchecking_asr_customization_infer.py \
      pretrained_model=${PRETRAINED_MODEL} \
      model.max_sequence_len=512 \
      +inference.from_filelist=${DATA_DIR}/filelist.txt \
      +inference.output_folder=${DATA_DIR}/spellchecker_output_${ASRTYPE} \
      inference.batch_size=16 \
      lang=en

    ## Postprocess and combine spellchecker results into a single manifest
    python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/evaluation/update_transcription_with_spellchecker_results.py \
      --asr_hypotheses_folder ${DATA_DIR}/hypotheses_${ASRTYPE} \
      --spellchecker_inputs_folder ${DATA_DIR}/spellchecker_input_${ASRTYPE} \
      --spellchecker_results_folder ${DATA_DIR}/spellchecker_output_${ASRTYPE} \
      --input_manifest ${DATA_DIR}/manifests/manifest_${ASRTYPE}.json \
      --output_manifest ${DATA_DIR}/manifests/manifest_${ASRTYPE}_corrected.json \
      --min_cov 0.4 \
      --min_real_cov 0.8 \
      --min_dp_score_per_symbol -1.5 \
      --ngram_mappings ${NGRAM_MAPPINGS}

    ## Check CER of spellchecker results
    python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
      dataset_manifest=${DATA_DIR}/manifests/manifest_${ASRTYPE}_corrected.json \
      use_cer=True \
      only_score_manifest=True

    ## Check WER of spellchecker results
    python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
      dataset_manifest=${DATA_DIR}/manifests/manifest_${ASRTYPE}_corrected.json \
      use_cer=False \
      only_score_manifest=True

    ## Perform error analysis and create "ideal" spellchecker results for comparison
    python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/evaluation/analyze_custom_ref_vs_asr.py \
      --manifest ${DATA_DIR}/manifests/manifest_${ASRTYPE}_corrected.json \
      --vocab_dir ${DATA_DIR}/vocabs \
      --input_dir ${DATA_DIR}/spellchecker_input_${ASRTYPE} \
      --ngram_mappings ${NGRAM_MAPPINGS} \
      --output_name ${DATA_DIR}/${ASRTYPE}_analysis_ref_vs_asr.txt

    ## Check CER of "ideal" spellcheck results
    python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
      dataset_manifest=${DATA_DIR}/${ASRTYPE}_analysis_ref_vs_asr.txt.ideal_spellcheck \
      use_cer=True \
      only_score_manifest=True

    ## Check WER of "ideal" spellcheck results
    python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
      dataset_manifest=${DATA_DIR}/${ASRTYPE}_analysis_ref_vs_asr.txt.ideal_spellcheck \
      use_cer=False \
      only_score_manifest=True
done

## If you want to test separately on test-clean and test-other you can grep corresponding lines from manifest, e.g.
##     grep "/test-clean/" ${DATA_DIR}/manifests/manifest_ctc.json > ${DATA_DIR}/manifests/manifest_ctc_test_clean.json
##     grep "/test-other/" ${DATA_DIR}/manifests/manifest_ctc.json > ${DATA_DIR}/manifests/manifest_ctc_test_other.json
