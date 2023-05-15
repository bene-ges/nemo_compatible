"""Script for evaluation of G2P for conversion of English to CMU phonemes.
   It cleans auxiliary tags and symbols from the phoneme sequence and then calculates WER between predicted and reference sequences. 

   Command example:
       python eval.py --g2p_output_name test.output.txt --reference_name test.reference.txt 

   Lines in g2p_output_name and reference_name correspond to each other.
   Lines in g2p_output_name consist of 5 columns, the input is in second column, the prediction is in the third column.
   Lines in reference_name consist of 2 columns: input and reference.

"""

from argparse import ArgumentParser

from nemo.collections.asr.metrics.wer import word_error_rate

parser = ArgumentParser(description="Evaluate WER of G2P conversion of English letters to CMU phonemes")
parser.add_argument("--g2p_output_name", type=str, required=True, help="Input file with g2p results")
parser.add_argument("--reference_name", type=str, required=True, help="Input file with reference")

args = parser.parse_args()

refs_for_wer = []
preds_for_wer = []

with open(args.reference_name, "r", encoding="utf-8") as f:
    for line in f:
        _, ref = line.strip().split("\t")
        ref = ref.replace("<DELETE>", "").replace("_", " ")
        ref = " ".join(ref.split())
        refs_for_wer.append(ref)

with open(args.g2p_output_name, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        pred = parts[2]
        pred = pred.replace("<DELETE>", "").replace("_", " ")
        pred = " ".join(pred.split())
        preds_for_wer.append(pred)

wer = word_error_rate(refs_for_wer, preds_for_wer, use_cer=False)

for i in range(len(preds_for_wer)):
    if preds_for_wer[i] != refs_for_wer[i]:
        print("[" + preds_for_wer[i] + "]")
        print("[" + refs_for_wer[i] + "]")
        print()

print("WER: ", wer)
