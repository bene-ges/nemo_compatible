NEMO_PATH=NeMo
NEMO_COMPATIBLE_PATH=nemo_compatible

## tts_input.txt is just all entries from Yago corpus passed through a G2P model.
## It should have the following format (space is also a phoneme)
## aadityana       AA0,AA2,D,AH0,T,Y,AE1,N,AH0
## aadivaram aadavallaku selavu    AA2,D,IH1,V,ER0,AE2,M, ,AA2,AA0,D,AH0,V,AA1,L,AA1,K,UW2, ,S,EH1,L,AH0,V,UW0
## aa divasam      EY1,EY1, ,D,IH0,V,AH0,S,AA1,M
## aadi velli      AA1,D,IY0, ,V,EH1,L,IY0

mkdir tts
mkdir tts_resample

split -n 26 tts_input.txt
for part in "xaa" "xab" "xac" "xad" "xae" "xaf" "xag" "xah" "xai" "xaj" "xak" "xal" "xam" "xan" "xao" "xap" "xaq" "xar" "xas" "xat" "xau" "xav" "xaw" "xax" "xay" "xaz"
do
    python ${NEMO_COMPATIBLE_PATH}/scripts/tts/tts_en_infer_from_cmu_phonemes.py --input_name $part --output_dir tts --output_manifest $part.json --sample_rate 16000
    python ${NEMO_PATH}/examples/asr/transcribe_speech.py \
      pretrained_name="stt_en_conformer_ctc_large" \
      dataset_manifest=${part}.json \
      output_filename=./pred_ctc.$part.json \
      batch_size=256 \
      cuda=0 \
      amp=True
done

cat pred_ctc.x*.json > pred_ctc.all.json

## Our final output file pred_ctc.all.json is a NeMo manifest, where each line is a json,
## containing fields
##  "text" - reference text
##  "pred_text" - predicted text
## For example:
## {... "text": "zyxomma elgneri", "pred_text": "six summer alnery"}
## its WER is expected to be above 80%

python ${NEMO_PATH}/examples/asr/speech_to_text_eval.py \
  dataset_manifest=pred_ctc.all.json \
  use_cer=False \
  only_score_manifest=True
