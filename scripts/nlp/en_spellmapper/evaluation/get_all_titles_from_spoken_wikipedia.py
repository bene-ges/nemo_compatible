"""
This script can be used to extract article titles from Spoken Wikipedia corpus.
The input folder consists of subfolders with following stricture
  ├── <Name of Wikipedia article>
  │   ├── aligned.swc
  │   ├── audiometa.txt
  │   ├── audio.ogg
  │   ├── info.json
  │   ├── wiki.html
  │   ├── wiki.txt
  │   └── wiki.xml
"""

import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_folder", required=True, type=str, help="Input folder in which each subfolder contains an article"
)
parser.add_argument("--output_file", required=True, type=str, help="Output file")
args = parser.parse_args()


if __name__ == "__main__":
    n = 0
    titles = []
    for name in os.listdir(args.input_folder):
        n += 1
        input_name = args.input_folder + "/" + name + "/info.json"
        if not os.path.exists(input_name):
            print("info.json does not exist in " + name)
            continue
        with open(input_name, "r", encoding="utf-8") as f:
            for line in f:
                js = json.loads(line.strip())
                if "article" not in js or "title" not in js["article"]:
                    print("no title in info.json in " + name)
                    continue
                titles.append(js["article"]["title"])

    with open(args.output_file, "w", encoding="utf-8") as out:
        for t in titles:
            out.write(t + "\n")
