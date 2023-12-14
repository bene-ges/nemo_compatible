## This script shows how to use WIKI-EN-ASR-ADAPT dataset to generate training data for SpellMapper model.

NEMO_COMPATIBLE_PATH=nemo_compatible

git clone https://huggingface.co/datasets/bene-ges/wiki-en-asr-adapt

## Vocabulary of aligned YAGO subphrases, allows to use not only Wikipedia titles as whole phrases, but also their parts.
## Preparation of this file is described in get_ngram_mappings.sh.
SUBMISSPELLS=wiki-en-asr-adapt/Keys2Corruptions.txt   #sub_misspells.txtx

## Vocabulary of n-gram mappings
## Preparation of this file is described in get_ngram_mappings.sh.
NGRAM_MAPPINGS=wiki-en-asr-adapt/NgramMappings.txt    #replacement_vocab_filt.txt

FALSE_POSITIVES=wiki-en-asr-adapt/FalsePositives.txt    #false_positives.txt

RELATED_PHRASES=wiki-en-asr-adapt/Keys2Related.txt    #related_phrases.txt

YAGO_WIKI=wiki-en-asr-adapt/Keys2Paragraphs.txt    #yago_wiki.txt

## Take a sample from  yago_wiki.txt file.
## Sampling is controlled by parameters --each_n_line (skip other) and --max_count (skips paragraph if all its phrases already occured at least as many times)
## Phrase lists and paragraphs are written to separate files with equal number of lines 
python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/dataset_preparation/sample_by_max_count.py \
  --input_name ${YAGO_WIKI} \
  --max_count 5 \
  --output_name yago_wiki_sample.txt

awk 'BEGIN {FS="\t"}{print $1}' < yago_wiki_sample.txt > yago_wiki_sample.phrases
awk 'BEGIN {FS="\t"}{print $2}' < yago_wiki_sample.txt > yago_wiki_sample.paragraphs

## Vocabulary from Google Text Normalization Dataset.
## It is used here to perform a simple fast text normalization by substitution.
## To generate this file use ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/evaluation/get_multi_reference_vocab.py
## Format: semiotic class, spoken, written, frequency
## CARDINAL        seventeen       17      103679
## CARDINAL        seventeen       xvii    2212
## DATE    nineteen eighties       1980s   57236
## DATE    nineteen eighties       1980's  546
## DATE    nineteen eighties       nineteen eighties       1
## CARDINAL        four hundred    400     28999
## CARDINAL        four hundred    400,    19
git clone https://huggingface.co/datasets/bene-ges/en_gtn_vocab
GTN_REFERENCE_VOCAB=en_gtn_vocab/en_gtn_vocab.txt

## Normalize paragraphs using substitution by GTN vocabulary (fast and simple).
python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/dataset_preparation/normalize_by_gtn_vocab.py \
  --input_file yago_wiki_sample.paragraphs \
  --output_file yago_wiki_sample.paragraphs.norm \
  --tn_vocab ${GTN_REFERENCE_VOCAB}

## Search phrases in the paragraph, cut spans containing some phrase(s) and some surrounding context.
## Two outputs: 1) examples with at least 1 correct candidate, 2) examples with no correct candidates.
python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/dataset_preparation/get_fragments_from_yago_wiki.py \
  --input_phrases_file yago_wiki_sample.phrases \
  --input_paragraphs_file yago_wiki_sample.paragraphs.norm \
  --output_file_non_empty fragments_non_empty.txt \
  --output_file_empty fragments_empty.txt \
  --step_in_words 5

## File fragments_non_empty.txt:
##   programming block known as cdis curriculum development institute of singapore began     1 2     [cdis] 27 31;[curriculum development] 32 54
##   tibensky january nineteen eighty four matej bel zivot a 1 2 3 4 [tibensky] 0 8;[matej] 38 43;[bel] 44 47;[zivot] 48 53
##   in response to malaysia's tv pendidikan 1 2 3   [malaysia's] 15 25;[tv pendidikan] 26 39;[pendidikan] 29 39

## File fragments_empty.txt:
##   on the first of february of that year   0
##   first of february of that year  0
##   celebrated its thirty years of television       0
##   its thirty years of television broadcasting on  0

## Add misspells and 10 candidates via different sampling
python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/dataset_preparation/construct_positive_and_negative_examples.py \
  --non_empty_fragments_file fragments_non_empty.txt \
  --empty_fragments_file fragments_empty.txt \
  --related_phrases_file ${RELATED_PHRASES} \
  --sub_misspells_file ${SUBMISSPELLS} \
  --false_positives_file ${FALSE_POSITIVES} \
  --output_file_positive positive.txt \
  --output_file_negative negative.txt

## File positive.txt:
## Format: text fragment, candidates (# - similar to correct candidate, * - false positive, & - random, no mark - correct), indices of correct targets in candidates, positions of correct targets in text,  misspelled targets
##   twenty thirteen top marques monaco      #marquet;monaco;marques;*mark oaten;*tpr;#aldo bonacossa;*mark kligman;*patwin;#alexandre buonacorsi;*theo martins  2 3      28 34;20 27     monoco;marx
##   the collection of the museum of tuscan  #escanlar;tuscan;*over the hump;*todd santos;&samtiden;*glenbow museum;*mimara museum;#puscanski;#hugh scanlon;*museum attendant     2       32 38   tusk can

## File negative.txt:
## Format: text fragment, candidates (* - false positive, & - random)
##   tom abbott played from nineteen seventy five seventy eight      *framke;*planeten;*nineta gusti;*pikfyve;*platycerioideae;*five guys;*live seventy nine;*japanese night heron;*ivantsovi;*abbott drive
##   assistant offensive coach for penn      *vasile stanescu;*jessica stern;*tet offensive;*pend;*dunsbach ferry;*hambach forest;*poochhe;*may offensive;*sven oftedal;*jessie stanton

## Here we want to take equally many examples containing false positives and not.
## This is done because examples with false positives are somewhat scarce.
grep "*" negative.txt > negative_with_fp.txt
grep -v "*" negative.txt > negative_no_fp.txt
grep "*" positive.txt > positive_with_fp.txt
grep -v "*" positive.txt > positive_no_fp.txt

cat negative_with_fp.txt negative_no_fp.txt | head -n 17000000 > negative2.txt
cat positive_with_fp.txt positive_no_fp.txt | head -n 17000000 > positive2.txt

shuf negative2.txt > negative3.txt
shuf positive2.txt > positive3.txt

mkdir data_tsv

## Make training examples as expected by the neural model.
python ${NEMO_COMPATIBLE_PATH}/scripts/nlp/en_spellmapper/dataset_preparation/make_final_training_examples.py \
  --positive_file positive3.txt \
  --negative_file negative3.txt \
  --output_folder data_tsv \
  --lines_in_portion 110000

## All .tsv files contain training examples - one example per line.
## Format: asr-hypothesis split by characters, 10 candidates, indices of correct candidates, positions of misspeled fragments in asr-hypothesis
## t _ v _ o r _ e y e _ o n _ t h e _ f i r s t _ o f _ f e b r u a r y _ o f _ t h a t _ y e a r _ s _ b _ c _ c e l e b r a t e d       t v r _ t u s c a n;t h e _ f i r s t _ e d e n;w s p c;s b c _ p u b l i s h i n g _ b u i l d i n g;t v r _ g r a n t u r a;t v r i;s b c;s o a e b _ t a i;d u y a r;f i n t h e n     6 7     CUSTOM 0 10;CUSTOM 49 54
## d i s t r i c t _ o f _ a f g h a n i s t a n   d i s t r i c t _ o f _ p r i s t i n a;f e n i _ d i s t r i c t;d i s t r i c t _ o f _ l o u i s i a n a;m o r g a n _ s t a n l e y;f l a g _ o f _ a f g h a n i s t a n;u n a _ d i s t r i c t;z i r c _ d i s t r i c t;f i n n _ f u g l e s t a d;p h l o _ f i n i s t e r;g u s t a f _ a u l e n     0

## To generate files config.json, label_map.txt, semiotic_classes.txt, run generate_configs.sh

## HOW TO TRAIN WITH NON-TARRED DATA 
## You can take any subsets of all.tsv to use directly as training and validation datasets.
## Example of all files needed to run training with non-tarred data:
## data_folder_example
##   ├── config.json
##   ├── label_map.txt
##   ├── semiotic_classes.txt
##   ├── test.tsv
##   └── train.tsv

## To run training with non-tarred data, use ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/run_training.sh
## Note that training with non-tarred data only works on single gpu. It makes sense if you use 1-2 million examples or less.

## HOW TO TRAIN WITH TARRED DATA
## To convert data to tarred format, split all.tsv to pieces of 110'000 examples and use ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/convert_data_to_tarred.sh
## To run training with tarred data, use ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/run_training_tarred.sh
## data_folder_example
##   ├── train_tarred
##   |   ├── part1.tar
##   |   ├── ...
##   |   └── part200.tar
##   ├── config.json
##   ├── label_map.txt
##   ├── semiotic_classes.txt
##   └── test.tsv
