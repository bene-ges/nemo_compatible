"""
This script can be used to preprocess Spoken Wikipedia corpus before running ctc-segmentation.
The input folder consists of subfolders with following stricture
  ├── <Name of Wikipedia article>
  │   ├── aligned.swc
  │   ├── audiometa.txt
  │   ├── audio.ogg
  │   ├── info.json
  │   ├── wiki.html
  │   ├── wiki.txt
  │   └── wiki.xml


## The destination folder will contain look enumerated .ogg and .txt files like this:
  ├── audio
  |   ├── 1.ogg
  |   ├── 2.ogg
  |   ...
  └── text
      ├── 1.txt     
      ├── 2.txt     
      ...
"""

import argparse
import os
import re
from collections import defaultdict

from nemo.collections.nlp.data.spellchecking_asr_customization.utils import (
    replace_diacritics,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_folder", required=True, type=str, help="Input folder in which each subfolder contains an article"
)
parser.add_argument(
    "--destination_folder", required=True, type=str, help="Destination folder with audio and text subfolder"
)
args = parser.parse_args()


def get_audio(name, n):
    """
    Copies .ogg file. If there are several .ogg files, concatenates them.

    Args:
        name - name of folder within Spoken Wikipedia
        n - integer that will serve as output file name, e.g. if n=1, file 1.ogg will be created  
    """
    audio_path = os.path.join(args.input_folder, name, "audio.ogg")
    if not os.path.exists(audio_path):
        ##  Some folders have multiple .ogg files, so we need to first combine them into one file. Example:
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

        multiple_ogg_files = []
        for i in range(1, 5):
            path = os.path.join(args.input_folder, name, "audio" + str(i) + ".ogg")
            if os.path.exists(path):
                multiple_ogg_files.append(path)
            else:
                break
        if len(multiple_ogg_files) == 0:
            return
        elif len(multiple_ogg_files) == 1:
            os.system("cp \"" + multiple_ogg_files[0] + "\" \"" + audio_path + "\"")
        else:
            tmp_file_name = "ffmeg_inputs.txt"
            print("tmp_file_name=", tmp_file_name)
            with open(tmp_file_name, "w", encoding="utf-8") as tmp_file:
                for path in multiple_ogg_files:
                    tmp_file.write("file '" + path + "'\n")
            cmd = "ffmpeg -f concat -i \"" + tmp_file_name + "\" -c copy \"" + audio_path + "\""
            print(cmd)
            os.system(cmd)

    output_audio_path = args.destination_folder + "/audio/" + str(n) + ".ogg"
    os.system("cp \"" + audio_path + "\" " + output_audio_path)


def get_text(name, n, word2doc):
    """
    Cleans wiki.txt.

    Args:
        name - name of folder within Spoken Wikipedia
        n - integer that will serve as output file name, e.g. if n=1, file 1.txt will be created  
    """

    # Then we need to clean the text
    out_text = open(args.destination_folder + "/text/" + str(n) + ".txt", "w", encoding="utf-8")
    headings = set()
    with open(args.input_folder + "/" + name + "/wiki.txt", "r", encoding="utf-8") as f:
        for line in f:
            do_break = False
            line2 = line.strip()
            ref_parts = line2.split("<ref")
            for idx, s in enumerate(ref_parts):
                if idx != 0:
                    s = "<ref" + s
                if s.startswith("[[Image") and s.endswith("]]"):
                    continue
                if s.startswith("[[File") and s.endswith("]]"):
                    continue
                if s.startswith(":"):
                    continue
                if s.startswith("{|") or s.startswith("|}") or s.startswith("|") or s.startswith("!"):
                    continue
                if s.startswith("{{") and (s.endswith("}}") or "}}" not in s):
                    continue
                if s.startswith("{{pp-move"):
                    continue
                s = re.sub(r"\[\[Image\:[^\]]+\]\]", r"", s)
                s = re.sub(r"\[\[File\:[^\]]+\]\]", r"", s)
                s = re.sub(r"\[http[^\]]+\]", r"", s)
                s = re.sub(r"<math>[^<>]+</math>", r"", s)
                s = re.sub(r"<!\-\-.+\-\->", r"", s)  # <!--DashBot-->    can be inside <ref>
                s = re.sub(r"<ref>.+</ref>", r"", s)
                s = re.sub(r"<ref .+</ref>", r"", s)
                s = re.sub(r"<ref[^<>]+/>", r"", s)
                s = re.sub(r"<[^ <>]+>", r"", s)  # <sub>, <sup>, </u>
                if (
                    re.match(r"== *Notes *==", s)
                    or re.match(r"== *References *==", s)
                    or re.match(r"== *External links *==", s)
                    or re.match(r"== *See also *==", s)
                ):
                    do_break = True
                    break
                s = re.sub(r"{{convert\|(\d+)\|(\w+)\|[^}]+}}", r"\g<1> \g<2>", s)  # {{convert|7600|lb|kg}}
                s = re.sub(r"{{cquote\|", r"", s)
                s = re.sub(r"{{[^{}]+}}", r"", s)
                s = s.replace("{{", "").replace("}}", "")
                s = re.sub(r"(lang[^()]+)", r"", s)  # (lang-bn...)
                s = re.sub(r"==+", r"", s)
                s = re.sub(r"''+", r" ", s)  # remove multiple quotes
                s = re.sub(r" '", r" ", s)  # remove quote at the beginning
                s = re.sub(r"' ", r" ", s)  # remove quote at the end
                s = re.sub(r"[…\*]", r" ", s)
                s = re.sub(r"\\u....", r" ", s)  # remove unicode
                s = re.sub(r"&[^ ;&]+;", r"", s)  # &nbsp; &mdash;

                s = replace_diacritics(s)

                for h in re.findall(r"\[\[[^\]]+\|([^\]]+)\]\]", s):
                    headings.add(h.casefold())
                s = re.sub(r"\[\[[^\]]+\|([^\]]+)\]\]", r"\g<1>", s)  # if several variants, take the last one

                for h in re.findall(r"\[\[([^\]]+)\]\]", s):
                    headings.add(h.casefold())
                s = re.sub(r"\[\[([^\]]+)\]\]", r"\g<1>", s)

                out_text.write(s + "\n")
                for w in re.findall(r"[\w']+", s):  # find all words
                    word2doc[w.casefold()].add(n)

            if do_break:
                break
    with open(args.destination_folder + "/vocabs/" + str(n) + ".headings.txt", "w", encoding="utf-8") as out:
        for h in sorted(headings):
            if len(h) < 3:
                continue
            h = re.sub(r"[\.\,\-]", r" ", h)
            h = re.sub(r"&", r" and ", h)
            h = " ".join(h.split())
            out.write(h + "\n")
    out_text.close()


if __name__ == "__main__":
    n = 0
    word2doc = defaultdict(set)
    for name in os.listdir(args.input_folder):
        n += 1
        print(name)
        if not os.path.exists(args.input_folder + "/" + name + "/wiki.txt"):
            print("wiki.txt does not exist in " + name)
            continue
        get_audio(name, n)
        get_text(name, n, word2doc)
    with open(args.destination_folder + "/vocabs/idf.txt", "w", encoding="utf-8") as out:
        for w, d in sorted(word2doc.items(), key=lambda item: len(item[1]), reverse=True):
            out.write(w + "\t" + str(len(word2doc[w])) + "\n")
