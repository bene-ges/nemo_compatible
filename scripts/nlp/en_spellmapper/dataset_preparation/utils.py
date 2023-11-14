"""Utility functions for English SpellMapper(Spellchecking ASR Customization) data preparation"""

import json
import re
from heapq import heappush, heapreplace
from typing import Dict, List, Set, Tuple

import numpy as np
from nemo.collections.nlp.data.spellchecking_asr_customization.utils import (
    replace_diacritics,
    search_in_index,
)

# ATTENTION: do not delete hyphen and apostrophe
CHARS_TO_IGNORE_REGEX = re.compile(r"[\.\,\?\:!;()«»…\]\[/\*–‽+&_\\½√>€™$•¼}{~—=“\"”″‟„]")
OOV_REGEX = "[^ '\-aiuenrbomkygwthszdcjfvplxq]"

SPACE_REGEX = re.compile(r"[\u2000-\u200F]", re.UNICODE)
APOSTROPHES_REGEX = re.compile(r"[’'‘`ʽ']")

def preprocess_apostrophes_space_diacritics(text):
    text = APOSTROPHES_REGEX.sub("'", text)  # replace different apostrophes by one
    text = re.sub(r"'+", "'", text)  # merge multiple apostrophes
    text = SPACE_REGEX.sub(" ", text)  # replace different spaces by one
    text = replace_diacritics(text)

    text = re.sub(r" '", " ", text)  # delete apostrophes at the beginning of word
    text = re.sub(r"' ", " ", text)  # delete apostrophes at the end of word
    text = re.sub(r" +", " ", text)  # merge multiple spaces
    return text


def get_title_and_text_from_json(content: str, exclude_titles: Set[str]) -> Tuple[str, str, str]:
    # Example of file content
    #   {"query":
    #     {"normalized":[{"from":"O'_Coffee_Club","to":"O' Coffee Club"}],
    #      "pages":
    #       {"49174116":
    #         {"pageid":49174116,
    #          "ns":0,
    #          "title":"O' Coffee Club",
    #          "extract":"O' Coffee Club (commonly known as Coffee Club) is a Singaporean coffee house..."
    #         }
    #       }
    #     }
    #   }
    try:
        js = json.loads(content.strip())
    except:
        print("cannot load json from text")
        return (None, None, None)
    if "query" not in js or "pages" not in js["query"]:
        print("no query[\"pages\"] in " + content)
        return (None, None, None)
    for page_key in js["query"]["pages"]:
        if page_key == "-1":
            continue
        page = js["query"]["pages"][page_key]
        if "title" not in page:
            continue
        title = page["title"]
        if title in exclude_titles:
            return (None, None, None)
        if "extract" not in page:
            continue
        text = page["extract"]
        title_clean = preprocess_apostrophes_space_diacritics(title)
        # number of characters is the same in p and p_clean
        title_clean = CHARS_TO_IGNORE_REGEX.sub(" ", title_clean).lower()
        return text, title, title_clean
    return (None, None, None)


def get_paragraphs_from_text(text):
    paragraphs = text.split("\n")
    for paragraph in paragraphs:
        if paragraph == "":
            continue
        p = preprocess_apostrophes_space_diacritics(paragraph)
        # number of characters is the same in p and p_clean
        p_clean = CHARS_TO_IGNORE_REGEX.sub(" ", p).lower()
        yield p, p_clean


def get_paragraphs_from_json(text, exclude_titles):
    # Example of file content
    #   {"query":
    #     {"normalized":[{"from":"O'_Coffee_Club","to":"O' Coffee Club"}],
    #      "pages":
    #       {"49174116":
    #         {"pageid":49174116,
    #          "ns":0,
    #          "title":"O' Coffee Club",
    #          "extract":"O' Coffee Club (commonly known as Coffee Club) is a Singaporean coffee house..."
    #         }
    #       }
    #     }
    #   }
    try:
        js = json.loads(text.strip())
    except:
        print("cannot load json from text")
        return
    if "query" not in js or "pages" not in js["query"]:
        print("no query[\"pages\"] in " + text)
        return
    for page_key in js["query"]["pages"]:
        if page_key == "-1":
            continue
        page = js["query"]["pages"][page_key]
        if "title" not in page:
            continue
        title = page["title"]
        if title in exclude_titles:
            continue
        if "extract" not in page:
            continue
        text = page["extract"]
        paragraphs = text.split("\n")
        for paragraph in paragraphs:
            if paragraph == "":
                continue
            p = preprocess_apostrophes_space_diacritics(paragraph)
            # number of characters is the same in p and p_clean
            p_clean = CHARS_TO_IGNORE_REGEX.sub(" ", p).lower()
            yield p, p_clean


def load_yago_entities(input_name: str, exclude_titles: Set[str]) -> Set[str]:
    yago_entities = set()
    with open(input_name, "r", encoding="utf-8") as f:
        for line in f:
            title_orig, title_clean = line.strip().split("\t")
            # meta-information is divided by __ from the title content, remove it
            if "__" in title_clean:
                title_clean, _ = title_clean.split("__")
            title_clean = title_clean.replace("_", " ")
            title_orig = title_orig.replace("_", " ")
            if title_orig in exclude_titles:
                print("skip: ", title_orig)
                continue
            yago_entities.add(title_clean)
    return yago_entities


def read_custom_phrases(filename: str, max_lines: int = -1, portion_size: int = -1) -> List[str]:
    """Reads custom phrases from input file.
    If input file contains multiple columns, only first column is used.
    """
    phrases = set()
    with open(filename, "r", encoding="utf-8") as f:
        n = 0
        n_for_portion = 0
        for line in f:
            parts = line.strip().split("\t")
            phrases.add(" ".join(list(parts[0].casefold().replace(" ", "_"))))
            if portion_size > 0 and n_for_portion >= portion_size:
                yield list(phrases)
                phrases = set()
                n_for_portion = 0
            if max_lines > 0 and n >= max_lines:
                yield list(phrases)
                return
            n += 1
            n_for_portion += 1
    yield list(phrases)


def get_candidates_with_most_coverage(
    phrases2positions: np.ndarray, phrase_lengths: List[int], max_candidates: int
) -> List[Tuple[float, int, int]]:
    """Returns k candidates whose ngrams cover most of the input text (compared to the candidate length).
       Args:
           phrases2positions: matrix where rows are phrases columns are letters of input sentence. Value is 1 on intersection of letter ngrams that were found in index leading to corresponding phrase. 
           phrase_lengths: list of phrase lengths (to avoid recalculation)
           max_candidates: required number of candidates
       Returns:
           List of tuples:
               coverage,
               approximate beginning position of the phrase
               phrase id
    """
    top = []
    for i in range(max_candidates):  # add placeholders for best candidates
        heappush(top, (0.0, -1, -1))

    for i in range(len(phrase_lengths)):
        phrase_length = phrase_lengths[i]
        all_coverage = np.sum(phrases2positions[i]) / phrase_length
        if all_coverage < 0.4:
            continue
        moving_sum = np.sum(phrases2positions[i, 0:phrase_length])
        max_sum = moving_sum
        best_pos = 0
        for pos in range(1, phrases2positions.shape[1] - phrase_length):
            moving_sum -= phrases2positions[i, pos - 1]
            moving_sum += phrases2positions[i, pos + phrase_length - 1]
            if moving_sum > max_sum:
                max_sum = moving_sum
                best_pos = pos

        coverage = max_sum / (phrase_length + 2)  # smoothing
        if coverage > top[0][0]:  # top[0] is the smallest element in the heap, top[0][0] - smallest coverage
            heapreplace(top, (coverage, best_pos, i))
    return top


def get_candidates_with_most_coverage_on_whole_input(
    phrases2positions: np.ndarray, max_candidates: int
) -> List[Tuple[float, int, int]]:
    """Returns k candidates whose ngrams cover most of the input text (compared to the candidate length).
       Args:
           phrases2positions: matrix where rows are phrases columns are letters of input sentence. Value is 1 on intersection of letter ngrams that were found in index leading to corresponding phrase. 
           max_candidates: required number of candidates
       Returns:
           List of tuples:
               coverage,
               approximate beginning position of the phrase (in case of this function always 0)
               phrase id
    """
    top = []
    for i in range(max_candidates):  # add placeholders for best candidates
        heappush(top, (0.0, -1, -1))

    coverage = np.sum(phrases2positions, axis=1) / (2 + phrases2positions.shape[1])
    indices = np.argpartition(coverage, -max_candidates)[-max_candidates:]

    for i in range(max_candidates):
        if coverage[indices[i]] >= 0.4:
            heapreplace(top, (coverage[indices[i]], 0, indices[i]))
    return top


def get_candidates(
    ngram2phrases: Dict[str, int],
    phrases: List[str],
    phrase_lengths: List[int],
    letters: List[str],
    max_candidates: int = 10,
    min_real_coverage: float = 0.8,
    match_whole_input: bool = False,
) -> List[str]:
    phrases2positions, position2ngrams = search_in_index(ngram2phrases, phrases, letters)
    if match_whole_input:
        top = get_candidates_with_most_coverage_on_whole_input(phrases2positions, 3 * max_candidates)
    else:
        top = get_candidates_with_most_coverage(phrases2positions, phrase_lengths, 3 * max_candidates)

    top_sorted = sorted(top, key=lambda item: item[0], reverse=True)
    # mask for each custom phrase, how many which symbols are covered by input ngrams
    phrases2coveredsymbols = [[0 for x in phrases[top_sorted[i][2]].split(" ")] for i in range(len(top_sorted))]
    candidates = []
    i = -1
    for coverage, begin, idx in top_sorted:
        i += 1
        phrase_length = phrase_lengths[idx]
        for pos in range(begin, begin + phrase_length):
            # we do not know exact end of custom phrase in text, it can be different from phrase length
            if pos >= len(position2ngrams):
                break
            for ngram in position2ngrams[pos]:
                for phrase_id, b, size, lp in ngram2phrases[ngram]:
                    if phrase_id != idx:
                        continue
                    for ppos in range(b, b + size):
                        if ppos >= phrase_length:
                            break
                        phrases2coveredsymbols[i][ppos] = 1

        real_coverage = sum(phrases2coveredsymbols[i]) / len(phrases2coveredsymbols[i])
        if real_coverage < min_real_coverage:
            continue
        candidates.append(phrases[idx])
        if len(candidates) >= max_candidates:
            break

    return candidates
