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

INPUT_DIR=${WORKDIR}/english_prepared
OUTPUT_DIR=${WORKDIR}/english_result


## Download the Spoken Wikipedia corpus for English
## Note, that there are some other languages available
## @InProceedings{KHN16.518,
##  author = {Arne K{\"o}hn and Florian Stegen and Timo Baumann},
##  title = {Mining the Spoken Wikipedia for Speech Data and Beyond},
##  booktitle = {Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC 2016)},
##  year = {2016},
##  month = {may},
##  date = {23-28},
##  location = {Portorož, Slovenia},
##  editor = {Nicoletta Calzolari (Conference Chair) and Khalid Choukri and Thierry Declerck and Marko Grobelnik and Bente Maegaard and Joseph Mariani and Asuncion Moreno and Jan Odijk and Stelios Piperidis},
##  publisher = {European Language Resources Association (ELRA)},
##  address = {Paris, France},
##  isbn = {978-2-9517408-9-1},
##  islrn = {684-927-624-257-3/},
##  language = {english}
## }

wget https://corpora.uni-hamburg.de/hzsk/de/islandora/object/file:swc-2.0_en-with-audio/datastream/TAR/en-with-audio.tar .
tar -xvf en-with-audio.tar

##  We get a folder English with 1339 subfolders, each subfolder corresponds to a Wikipedia article. Example:
##  ├── Universal_suffrage
##  │   ├── aligned.swc
##  │   ├── audiometa.txt
##  │   ├── audio.ogg
##  │   ├── info.json
##  │   ├── wiki.html
##  │   ├── wiki.txt
##  │   └── wiki.xml

##  We will use two files: audio.ogg and wiki.txt

## Some folders have multiple .ogg files, this will be handled during preprocess.py. Example:
##  |── Universe
##  │   ├── aligned.swc
##  │   ├── audio1.ogg
##  │   ├── audio2.ogg
##  │   ├── audio3.ogg
##  │   ├── audio4.ogg
##  │   ├── audiometa.txt
##  │   ├── info.json
##  │   ├── wiki.html
##  │   ├── wiki.txt
##  │   └── wiki.xml

## Some rare folders are incomplete, these will be skipped during preprocessing.

## Rename some folders with special symbols because they cause problems to ffmpeg when concatening multiple .ogg files
mv "english/The_Hitchhiker%27s_Guide_to_the_Galaxy" "english/The_Hitchhikers_guide_to_the_Galaxy"
mv "english/SummerSlam_(2003)" "english/SummerSlam_2003"
mv "english/Over_the_Edge_(1999)" "english/Over_the_Edge_1999"
mv "english/Lost_(TV_series)" "english/Lost_TV_series"
mv "english/S._A._Andr%c3%a9e%27s_Arctic_Balloon_Expedition_of_1897" "english/S_A_Andres_Arctic_Balloon_Expedition_of_1897"

INPUT_DIR="english"
OUTPUT_DIR=${INPUT_DIR}_result

rm -rf $OUTPUT_DIR
rm -rf ${INPUT_DIR}_prepared
mkdir ${INPUT_DIR}_prepared
mkdir ${INPUT_DIR}_prepared/audio
mkdir ${INPUT_DIR}_prepared/text
mkdir ${INPUT_DIR}_prepared/vocabs
mkdir ${OUTPUT_DIR}
mkdir ${OUTPUT_DIR}/manifests

## Preprocess and collect first part of custom vocabularies - extract headings of referenced articles.
python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/evaluation/preprocess_spoken_wikipedia_and_create_vocabs.py --input_folder ${INPUT_DIR} --destination_folder ${INPUT_DIR}_prepared

## Run ctc-segmentation to extract shorter fragments like sentences and corresponding audio fragments.
## Note that this procedure extracts only parts of text that it manages to align, and substantial parts of article can be lost.
MODEL_FOR_SEGMENTATION="QuartzNet15x5Base-En" 
## We set this threshold as very permissive, later we will use other metrics for filtering
THRESHOLD=-10

${NEMO_PATH}/tools/ctc_segmentation/run_segmentation.sh \
--SCRIPTS_DIR=${NEMO_PATH}/tools/ctc_segmentation/scripts \
--MODEL_NAME_OR_PATH=${MODEL_FOR_SEGMENTATION} \
--DATA_DIR=${INPUT_DIR}_prepared \
--OUTPUT_DIR=${OUTPUT_DIR} \
--MIN_SCORE=${THRESHOLD}

# Thresholds for filtering
CER_THRESHOLD=20
WER_THRESHOLD=30
CER_EDGE_THRESHOLD=30
LEN_DIFF_RATIO_THRESHOLD=0.15
EDGE_LEN=25
BATCH_SIZE=1

## Run baseline ASR on the generated manifest, apply additional filtering based on its results
## Note that the final output file has name "manifest_transcribed_metrics_filtered.json"
MODEL_FOR_RECOGNITION="stt_en_conformer_ctc_large"
${NEMO_PATH}/tools/ctc_segmentation/run_filter.sh \
--SCRIPTS_DIR=${NEMO_PATH}/tools/ctc_segmentation/scripts \
--MODEL_NAME_OR_PATH=${MODEL_FOR_RECOGNITION} \
--BATCH_SIZE=${BATCH_SIZE} \
--MANIFEST=$OUTPUT_DIR/manifests/manifest.json \
--INPUT_AUDIO_DIR=${INPUT_DIR}_prepared/audio/ \
--EDGE_LEN=${EDGE_LEN} \
--CER_THRESHOLD=${CER_THRESHOLD} \
--WER_THRESHOLD=${WER_THRESHOLD} \
--CER_EDGE_THRESHOLD=${CER_EDGE_THRESHOLD} \
--LEN_DIFF_RATIO_THRESHOLD=${LEN_DIFF_RATIO_THRESHOLD}

## This is end of segmentation. We already have ctc baseline as its by-product.

## Merge multiple spaces (only occur in Conformer CTC).
python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/evaluation/combine_multiple_space_in_manifest.py \
  --input_manifest ${WORKDIR}/manifests/manifest_transcribed_metrics_filtered.json \
  --output_manifest ${WORKDIR}/manifests/manifest_ctc.json

## Transcribe data by NeMo model Conformer-Transducer Large (this is the second baseline model)
python ${NEMO_PATH}/examples/asr/transcribe_speech.py \
  pretrained_name="stt_en_conformer_transducer_large" \
  dataset_manifest=${WORKDIR}/manifests/manifest_ctc.json \
  output_filename=${WORKDIR}/manifests/manifest_transducer.json \
  batch_size=16

## Check WER of ASR results on the resulting dataset.
python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=${OUTPUT_DIR}/manifests/manifest_ctc.json \
  use_cer=False \
  only_score_manifest=True

python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=${OUTPUT_DIR}/manifests/manifest_transducer.json \
  use_cer=False \
  only_score_manifest=True

## At this moment your ${WORKDIR} folder structure should look like this:
##  data_folder_example
##   ├── english
##   |   ├── (I_Can%27t_Get_No)_Satisfaction
##   |   ├── ...
##   │   ├── Zinc
##   │   └── Zorbing
##   ├── english_prepared
##   │   ├── audio
##   |   |   ├── 1.ogg
##   |   |   ├── ...
##   |   |   └── 1340.ogg
##   │   ├── text
##   |   |   ├── 1.txt
##   |   |   ├── ...
##   |   |   └── 1340.txt
##   │   └── vocabs
##   |       ├── 1.headings.txt
##   |       ├── ...
##   |       ├── 1340.headings.txt
##   |       └── idf.txt
##   └── english_result
##       ├── clips
##       ├── manifests
##       |   ├── manifest.json
##       |   ├── manifest_ctc.json
##       |   ├── manifest_transducer.json
##       |   └── manifest_transcribed_metrics_filtered.json
##       ├── processed
##       |   ├── ...
##       |   ├── 1000.txt
##       |   ├── 1000.wav
##       |   ├── 1000_with_punct.txt
##       |   ├── 1000_with_punct_normalized.txt
##       |   ├── ...
##       |   └── en_grammars
##       ├── segments
##       └── verified_segments

## Note that some output files can be missing or empty, because of unsuccessful ctc-segmentation - this is ok.

## Create custom vocabularies in ${INPUT_DIR}/vocabs/{1..1340}.custom.txt
## It will use *.headings.txt and rare words/phrases from article text.
python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/evaluation/create_custom_vocabs_for_spoken_wikipedia.py \
  --folder ${INPUT_DIR} 
  --processed_folder ${OUTPUT_DIR}/processed
  --min_len 6

for ASRTYPE in "ctc" "transducer"
do
    ## Split ASR output transcriptions into shorter fragments to serve as ASR hypotheses for spellchecking model
    mkdir ${OUTPUT_DIR}/hypotheses_${ASRTYPE}
    python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/evaluation/extract_asr_hypotheses.py \
      --manifest ${OUTPUT_DIR}/manifests/manifest_${ASRTYPE}.json \
      --folder ${OUTPUT_DIR}/hypotheses_${ASRTYPE}

    ## Prepare inputs for inference of neural customization spellchecking model
    mkdir ${OUTPUT_DIR}/spellchecker_input_${ASRTYPE}
    mkdir ${OUTPUT_DIR}/spellchecker_output_${ASRTYPE}
    python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/evaluation/prepare_input_for_spellchecker_inference.py \
      --hypotheses_folder ${OUTPUT_DIR}/hypotheses_${ASRTYPE} \
      --vocabs_folder ${INPUT_DIR}/vocabs \
      --output_folder ${OUTPUT_DIR}/spellchecker_input_${ASRTYPE} \
      --ngram_mappings ${NGRAM_MAPPINGS} \
      --big_sample_file ${BIG_SAMPLE}

    ## Create filelist with input filenames
    rm ${WORKDIR}/filelist.txt
    for i in {1..1341}
    do
        echo ${OUTPUT_DIR}/spellchecker_input/${i}.txt >> ${WORKDIR}/filelist.txt
    done

    ## Run inference with neural customization spellchecking model
    python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/spellchecking_asr_customization_infer.py \
      pretrained_model=${PRETRAINED_MODEL} \
      model.max_sequence_len=512 \
      +inference.from_filelist=${WORKDIR}/filelist.txt \
      +inference.output_folder=${OUTPUT_DIR}/spellchecker_output_${ASRTYPE} \
      inference.batch_size=16 \
      lang=en

    ## Postprocess and combine spellchecker results into a single manifest
    python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/evaluation/update_transcription_with_spellchecker_results.py \
      --asr_hypotheses_folder ${OUTPUT_DIR}/hypotheses_${ASRTYPE} \
      --spellchecker_results_folder ${OUTPUT_DIR}/spellchecker_output_${ASRTYPE} \
      --input_manifest ${OUTPUT_DIR}/manifests/manifest_${ASRTYPE}.json \
      --output_manifest ${OUTPUT_DIR}/manifests/manifest_${ASRTYPE}_corrected.json \
      --ngram_mappings ${NGRAM_MAPPINGS} \
      --min_dp_score_per_symbol -1.5

    ## Check WER of spellchecker results
    python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
      dataset_manifest=${OUTPUT_DIR}/manifests/manifest_${ASRTYPE}_corrected.json \
      use_cer=False \
      only_score_manifest=True

    ## Perform error analysis and create "ideal" spellchecker results for comparison
    python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/evaluation/analyze_custom_ref_vs_asr.py \
      --manifest ${OUTPUT_DIR}/manifests/manifest_${ASRTYPE}_corrected.json \
      --vocab_dir ${INPUT_DIR}/vocabs \
      --input_dir ${OUTPUT_DIR}/spellchecker_input_${ASRTYPE} \
      --ngram_mappings ${NGRAM_MAPPINGS} \
      --output_name ${WORKDIR}/${ASRTYPE}_analysis_ref_vs_asr.txt

    ## Check WER of "ideal" spellcheck results
    python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
      dataset_manifest=${WORKDIR}/${ASRTYPE}_analysis_ref_vs_asr.txt.ideal_spellcheck \
      use_cer=False \
      only_score_manifest=True

done
