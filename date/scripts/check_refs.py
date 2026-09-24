"""Report undefined \\ref targets and brace balance for the active manuscript tree.

Read-only: parses the .tex sources and prints, per file, any \\ref whose label is
not defined in the same file, plus a brace-balance count that catches the class of
typo that previously produced a fatal \\Hy@tempa error.

Run: python -u check_refs.py
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

FILES = [
    "manuscript_p2_optical_engineering.tex",
]

LABEL = re.compile(r"\\label\{([^}]+)\}")
REF = re.compile(r"\\(?:page|eq|auto)?ref\{([^}]+)\}")
CITE = re.compile(r"\\cite[a-zA-Z]*\{([^}]+)\}")


def check(name):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        print("%-56s MISSING" % name)
        return
    s = open(path, encoding="utf-8").read()
    labels = set(LABEL.findall(s))
    refs = REF.findall(s)
    cites = CITE.findall(s)
    undefined = sorted({r for r in refs if r not in labels})
    # Brace balance: a raw "{" count equal to "}" is necessary but not
    # sufficient; a mismatch is always a defect.
    open_b, close_b = s.count("{"), s.count("}")
    print("%-56s labels=%-4d refs=%-4d cites=%-3d braces %d/%d %s"
          % (name, len(labels), len(refs), len(cites), open_b, close_b,
             "OK" if open_b == close_b else "*** UNBALANCED"))
    if undefined:
        print("    UNDEFINED REFS: %s" % undefined)
    else:
        print("    all \\ref targets resolve")


def main():
    for f in FILES:
        check(f)


if __name__ == "__main__":
    main()
