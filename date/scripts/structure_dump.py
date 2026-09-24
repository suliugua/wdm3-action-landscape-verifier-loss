"""Dump section/subsection headings with approximate body word counts.

Read-only. Used to judge the balance of the manuscript: which sections carry the
argument and which are appendices-in-disguise.

Run: python -u structure_dump.py [file.tex]
"""
import re
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "manuscript_p2_optical_engineering.tex"
HEAD = re.compile(r"\\(section|appendix|subsection)\*?\{")
MACRO = re.compile(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?")


def main():
    text = open(PATH, encoding="utf-8").read()
    lines = text.split("\n")
    heads = [(i, l) for i, l in enumerate(lines) if HEAD.match(l)]
    total = 0
    for k, (i, l) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        body = MACRO.sub(" ", "\n".join(lines[i:end]))
        words = len(re.findall(r"[A-Za-z][A-Za-z'-]+", body))
        total += words
        lvl = "SEC" if l.startswith("\\section") or l.startswith("\\appendix") else "  sub"
        print("%s %5dw  L%-5d %s" % (lvl, words, i + 1, l[:86]))
    print("\ntotal words counted: %d" % total)


if __name__ == "__main__":
    main()
