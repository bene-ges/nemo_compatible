# Spellchecking model for ASR Customization (Russian language)

Pipeline for this model differs from English (see "../README.md") because Russian has rich morphology (same words can have various endings).

As initial data we use about 4 mln entities from [Wikipedia dump](https://dumps.wikimedia.org/ruwiki/20230301/ruwiki-20230301-all-titles.gz). 

The pipeline consists of multiple steps:

1. Download and preprocess Yago corpus and instruction on how to download full Wikipedia articles
   `dataset_preparation/preprocess_ruwiki_titles.sh`

2. Preprocess full Wikipedia articles
   `dataset_preparation/preprocess_ruwiki_articles.sh`

Здесь отличие от того как происходит в английском.
В английском достаточно было пропустить через TTS+G2P+ASR только названия статей, потому что в английском мало словоизменения.
В русском так не годится, потому что все примеры будут в начальной форме.
Нужно собрать примеры выражений в разных формах из текста статей.

Названия статей
    https://huggingface.co/datasets/bene-ges/wikipedia_ru_titles

Примеры запакованных файлов со скачанными статьями (пока не все, только на букву -я)
    https://huggingface.co/datasets/bene-ges/wikipedia_ru

Пример скрипта, как итерироваться по скачанным статьям (английским)
    https://github.com/bene-ges/nemo_compatible/blob/main/scripts/nlp/en_spellmapper/dataset_preparation/prepare_sentences_from_yago_wiki.py

Нужно написать такой же скрипт, но чтобы прикладывал находил употребления в разных формах.

Для прикладывания окончаний можно использовать список пар окончаний из файла endings_10.txt
 
3. TTS+G2P+ASR

4. Выравнивание, сбор словаря н-грамм

5. Подготовка обучающих данных. Нужно извлечь из текстов статей occurrences + окружающий контекст, добавить искусственные ошибки,
    добавить искусственные списки кандидатов.
   Опционально. Конвертация в tarred dataset.

6. Изменить код модели (добавить дополнительную голову и т п) и код загрузки bert_example

Как я собирась исправлять окончания при помощи берта:
```
input:        признаки досиминированныго туберкулеса       <кандидат_1>; диссеминированный туберкулез ; ... ; <кандидат_10>
segment_ids:  00000000000000000000000000000000000000       111111111111  2222222222222222222222222222   333
targets:      00000000022222222222222***22222222222#       ...           1111111111111110011111111111
* - 234 (окончание -ого)
# - 15 (окончание -а)
окончания можно нумеровать метками начиная с 11.

В таргете для входного текста символы окончания которое нужно удалить помечаются лейблом окончания, на которое нужно заменить. 
В таргет для самих кандидатов добавляется маска, которая предсказывает нули на месте символов окончания, которые надо отрезать.
Либо можно вместо нулей предсказывать тоже лейбл для окончания, на которое заменить.
В остальном все похоже на то как работает для английского.
```
 
7. Обучение

8. Тестирование. Нужны тестовые наборы для русского.
