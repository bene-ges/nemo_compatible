#!/bin/bash -l
conda activate nemo

NEMO_PATH=NeMo

## download https://dumps.wikimedia.org/ruwiki/20230301/ruwiki-20230301-all-titles.gz

awk 'BEGIN {FS="\t"} ($1 == "0"){print $2}($1 == "1"){print $2}' < ruwiki-20230301-all-titles | sort -u > ruwiki_titles.uniq
python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/ru/preprocess_ruwiki_titles.py --input_name ruwiki_titles.uniq --output_name ruwiki_titles.uniq2

## ruwiki_titles.uniq2 example content:

## Ящерицы ящерицы__1
## Ящерицы_безногие        ящерицы_безногие__1
## Ящерицы_Волгоградской_области   ящерицы_волгоградской_области__1
## Ящерицын,_Петр_Иванович ящерицын_петр_иванович__1
## Ящерицын_Петр_Иванович  ящерицын_петр_иванович__2
## Ящерицын,_Пётр_Иванович ящерицын_петр_иванович__3
## Ящерицын_Пётр_Иванович  ящерицын_петр_иванович__4
## Ящеричная_змея  ящеричная_змея__1
## Ящеричные       ящеричные__1
## Ящеричные_змеи  ящеричные_змеи__1
## Ящерка_Билль    ящерка_билль__1
## Ящерка_(приток_Бурначки)        ящерка__2
## Ящер_(мифология)        ящер__7
## Ящер_(млекопитающее)    ящер__8
## Ящерощука       ящерощука__1
## Ящер-тиран      ящер-тиран__1
## Ящик    ящик__1
## Ящик_для_бутылок        ящик_для_бутылок__1
## Ящик_для_голосования    ящик_для_голосования__1
## Ящик_для_письменных_принадлежностей_(Ямницер)   ящик_для_письменных_принадлежностей__1
## Ящик_для_пожертвований  ящик_для_пожертвований__1
## Ящик_(значения) ящик__2

## We want to download all Wikipedia articles with titles from yago.uniq2
## Example of download command
## wget "https://ru.wikipedia.org/w/api.php?format=xml&action=query&prop=extracts&titles=Ящерка_Билль&redirects=true&format=json&explaintext=1&exsectionformat=plain" -O ящерка_билль__1.txt

## To use downloaded artiles in later scripts, you need to create a folder with following structure:
## wikipedia_ru
##  ├── part_xaa.tar.gz
##  ├── ...
##  └── part_xeс.tar.gz
## Names do not matter, each tar.gz contains multiple downloaded articles, each in a separate json file 
## Example of a downloaded json
## {"batchcomplete":"","query":{"normalized":[{"from":"\u042d\u0440\u0433\u0443\u043d_\u0413\u044e\u043c\u0440\u044e\u043a\u0447\u044e\u043e\u0433\u043b\u0443","to":"\u042d\u0440\u0433\u0443\u043d \u0413\u044e\u043c\u0440\u044e\u043a\u0447\u044e\u043e\u0433\u043b\u0443"}],"redirects":[{"from":"\u042d\u0440\u0433\u0443\u043d \u0413\u044e\u043c\u0440\u044e\u043a\u0447\u044e\u043e\u0433\u043b\u0443","to":"\u0413\u044e\u043c\u0440\u044e\u043a\u0447\u044e\u043e\u0433\u043b\u0443, \u042d\u0440\u0433\u0443\u043d"}],"pages":{"8007751":{"pageid":8007751,"ns":0,"title":"\u0413\u044e\u043c\u0440\u044e\u043a\u0447\u044e\u043e\u0433\u043b\u0443, \u042d\u0440\u0433\u0443\u043d","extract":"\u042d\u0440\u0433\u0443\u043d \u0413\u044e\u043c\u0440\u044e\u043a\u0447\u044e\u043e\u0433\u043b\u0443 (\u0442\u0443\u0440. Ergun G\u00fcmr\u00fck\u00e7\u00fco\u011flu) \u2014 \u0442\u0443\u0440\u0435\u0446\u043a\u0438\u0439 \u0448\u0430\u0445\u043c\u0430\u0442\u0438\u0441\u0442, \u043d\u0430\u0446\u0438\u043e\u043d\u0430\u043b\u044c\u043d\u044b\u0439 \u043c\u0430\u0441\u0442\u0435\u0440.\n\u0414\u0432\u0443\u043a\u0440\u0430\u0442\u043d\u044b\u0439 \u0447\u0435\u043c\u043f\u0438\u043e\u043d \u0422\u0443\u0440\u0446\u0438\u0438 (1977 \u0438 1980 \u0433\u0433.).\n\u0412 \u0441\u043e\u0441\u0442\u0430\u0432\u0435 \u043d\u0430\u0446\u0438\u043e\u043d\u0430\u043b\u044c\u043d\u043e\u0439 \u0441\u0431\u043e\u0440\u043d\u043e\u0439 \u0422\u0443\u0440\u0446\u0438\u0438 \u0443\u0447\u0430\u0441\u0442\u043d\u0438\u043a \u0448\u0430\u0445\u043c\u0430\u0442\u043d\u043e\u0439 \u043e\u043b\u0438\u043c\u043f\u0438\u0430\u0434\u044b 1980 \u0433. \u0438 \u0434\u0432\u0443\u0445 \u0411\u0430\u043b\u043a\u0430\u043d\u0438\u0430\u0434.\n\u0423\u0447\u0430\u0441\u0442\u0432\u043e\u0432\u0430\u043b \u0432 \u0448\u0430\u0445\u043c\u0430\u0442\u043d\u044b\u0445 \u0441\u043e\u0440\u0435\u0432\u043d\u043e\u0432\u0430\u043d\u0438\u044f\u0445 \u0434\u043e \u043d\u0430\u0447\u0430\u043b\u0430 1990-\u0445 \u0433\u0433.\n\u0421\u0432\u0435\u0434\u0435\u043d\u0438\u044f \u043e \u0436\u0438\u0437\u043d\u0438 \u0448\u0430\u0445\u043c\u0430\u0442\u0438\u0441\u0442\u0430 \u043a\u0440\u0430\u0439\u043d\u0435 \u0441\u043a\u0443\u0434\u043d\u044b.\n\n\n\u0421\u043f\u043e\u0440\u0442\u0438\u0432\u043d\u044b\u0435 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u044b\n\n\n\u041f\u0440\u0438\u043c\u0435\u0447\u0430\u043d\u0438\u044f\n\n\n\u0421\u0441\u044b\u043b\u043a\u0438\n\u041f\u0430\u0440\u0442\u0438\u0438 \u042d. \u0413\u044e\u043c\u0440\u044e\u043a\u0447\u044e\u043e\u0433\u043b\u0443 \u0432 \u0431\u0430\u0437\u0435 Chessgames\n\u041b\u0438\u0447\u043d\u0430\u044f \u043a\u0430\u0440\u0442\u043e\u0447\u043a\u0430 \u042d. \u0413\u044e\u043c\u0440\u044e\u043a\u0447\u044e\u043e\u0433\u043b\u0443 \u043d\u0430 \u0441\u0430\u0439\u0442\u0435 365Chess"}}}}

## Example of script looping through such folder with .tar.gz files (for English)
## https://github.com/bene-ges/nemo_compatible/blob/main/scripts/nlp/en_spellmapper/dataset_preparation/prepare_sentences_from_yago_wiki.py

