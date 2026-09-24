"""Compare cited keys in the manuscript against the entries in the .bib.

Two defects are possible and both matter for submission:
  * a key cited in the text with no bib entry  -> undefined citation;
  * a bib entry never cited                    -> an uncited reference.

Usage: python -u audit_cites.py [manuscript.tex] [references.bib]
"""
import os
import re
import sys

TEX = sys.argv[1] if len(sys.argv) > 1 else "manuscript_p2_optical_engineering.tex"
BIB = sys.argv[2] if len(sys.argv) > 2 else "references_p2_oe.bib"

CITE = re.compile(r"\\cite[a-zA-Z]*\{([^}]+)\}")
ENTRY = re.compile(r"(?m)^@\w+\{([^,]+),")


def main():
    tex = open(TEX, encoding="utf-8").read()
    bib = open(BIB, encoding="utf-8").read()

    cited = set()
    for m in CITE.finditer(tex):
        cited |= {k.strip() for k in m.group(1).split(",") if k.strip()}
    defined = {m.group(1).strip() for m in ENTRY.finditer(bib)}

    print("%s: %d distinct keys cited" % (os.path.basename(TEX), len(cited)))
    print("%s: %d entries defined" % (os.path.basename(BIB), len(defined)))

    undef = sorted(cited - defined)
    uncited = sorted(defined - cited)
    print("\ncited but NOT in the bib: %s" % (undef or "none"))
    print("in the bib but NEVER cited: %s" % (uncited or "none"))


if __name__ == "__main__":
    main()
