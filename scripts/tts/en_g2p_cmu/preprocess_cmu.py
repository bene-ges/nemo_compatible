# This Python file uses the following encoding: utf-8

import re
from os.path import join

from argparse import ArgumentParser

parser = ArgumentParser(description="Preprocess CMUdict and prepare input for GIZA++")
parser.add_argument("--input_name", type=str, required=True, help="Input file")
parser.add_argument("--output_name", type=str, required=True, help="Output file")
parser.add_argument("--out_dir", type=str, required=True, help="Path to output folder")
parser.add_argument("--giza_dir", type=str, required=True, help="Path to folder with GIZA++ binaries")
parser.add_argument("--mckls_binary", type=str, required=True, help="Path to mckls binary")

args = parser.parse_args()

with open(join(args.out_dir, "run.sh"), "w") as out:
    out.write("GIZA_PATH=\"" + args.giza_dir + "\"\n")
    out.write("MKCLS=\"" + args.mckls_binary + "\"\n")
    out.write("\n")
    out.write("${GIZA_PATH}/plain2snt.out src dst\n")
    out.write("${MKCLS} -m2 -psrc -c4 -Vsrc.classes opt >& mkcls1.log\n")
    out.write("${MKCLS} -m2 -pdst -c4 -Vdst.classes opt >& mkcls2.log\n")
    out.write("${GIZA_PATH}/snt2cooc.out src.vcb dst.vcb src_dst.snt > src_dst.cooc\n")
    out.write(
        "${GIZA_PATH}/GIZA++ -S src.vcb -T dst.vcb -C src_dst.snt -coocurrencefile src_dst.cooc -p0 0.98 -o GIZA++ >& GIZA++.log\n"
    )
    out.write("##reverse direction\n")
    out.write("${GIZA_PATH}/snt2cooc.out dst.vcb src.vcb dst_src.snt > dst_src.cooc\n")
    out.write(
        "${GIZA_PATH}/GIZA++ -S dst.vcb -T src.vcb -C dst_src.snt -coocurrencefile dst_src.cooc -p0 0.98 -o GIZA++reverse >& GIZA++reverse.log\n"
    )

out = open(args.output_name, "w", encoding="utf-8")

with open(args.input_name, "r", encoding="utf-8") as inp:
    for line in inp:
        s = line.strip()
        if line.startswith(";;;"):
            continue

        s = re.sub(r"^[^\d\w]+", r"", s)   #delete punctuation at beginning
        parts = s.split("  ")
        assert(len(parts) == 2)
        g, p = parts    #g - graphematic, p - phonetic
        g = re.sub(r"\(\d+\)$", r"", g)   #delete brackets at end
        g = g.casefold()
        g = g.replace("-", "")
        if re.match(".*\d", g):
            continue
        if re.match(".*[\.,!?:;]", g):
            continue
        out.write(" ".join(list(g)) + "\t" + p + "\n")

out.close()
