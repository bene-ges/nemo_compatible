import re

"""Utility functions for Russian grapheme-to-phoneme conversion using IPA-like alphabet"""

def clean_russian_g2p_trascription(text: str) -> str:
    result = text
    result = result.replace("<DELETE>", " ").replace("+", "").replace("~", "")
    result = result.replace("ʑ", "ɕ:").replace("ɣ", "x")
    result = result.replace(":", "ː").replace("'", "`")
    result = "".join(result.split())
    result = result.replace("_", " ")
    return result


def clean_russian_text_for_tts(text: str) -> str:
    result = text
    result = result.replace("+", "")  # remove stress
    result = result.casefold()  # lowercase
    result = result.replace("ё", "е")
    result = result.replace("\u2011", "-")  # non-breaking hyphen
    result = result.replace("\u2013", "-")  # en dash
    result = result.replace("\u2014", "-")  # em dash
    result = result.replace("\u2026", ".")  # horizontal ellipsis
    result = result.replace("\u00ab", "\"")  # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
    result = result.replace("\u00bb", "\"")  # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    result = result.replace("\u2019", "'")  # ’ Right Single Quotation Mark
    result = result.replace("\u201c", "\"")  # “ Left Double Quotation Mark
    result = result.replace("\u201d", "\"")  # ” Right Double Quotation Mark
    result = result.replace("\u201e", "\"")  # „ Double Low-9 Quotation Mark
    result = result.replace("\u201f", "\"")  # ‟ Double High-reversed-9 Quotation Mark
    return result

