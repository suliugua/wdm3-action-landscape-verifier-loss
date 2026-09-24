"""Compare the numeric content of two .tex files.

A restructure moves thousands of words; the risk it carries is not losing prose
but silently losing or altering a number. This extracts every numeric literal
from both files and reports what disappeared and what appeared, so a dropped
statistic is visible rather than discovered by a reviewer.

Usage: python -u audit_numbers.py <before.tex> <after.tex>
"""
import collections
import re
import sys

# Numeric literals as they appear in this manuscript: decimals, percentages,
# scientific notation, and integers that are not part of a macro name.
NUM = re.compile(r"[-+]?\d+\.\d+(?:\\times10\^\{[-+]?\d+\})?"
                 r"|[-+]?\d+\s*\\%"
                 r"|[-+]?\d+(?:\\times10\^\{[-+]?\d+\})"
                 r"|\b\d{1,3}\b")


def counts(path):
    text = open(path, encoding="utf-8").read()
    # Drop comments, but NOT \%, which is an escaped percent in LaTeX text.
    # A naive r"%.*" eats every line from its first \% onward and silently
    # hides most of the numbers on that line.
    text = re.sub(r"(?<!\\)%.*", "", text)
    return collections.Counter(m.group(0).strip() for m in NUM.finditer(text))


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a, b = counts(sys.argv[1]), counts(sys.argv[2])
    gone = a - b
    new = b - a
    print("distinct numeric tokens: before=%d after=%d" % (len(a), len(b)))
    print("\n-- present before, reduced or absent after (top 40) --")
    for tok, n in gone.most_common(40):
        print("   %-28s %3d -> %d" % (tok, n, b.get(tok, 0)))
    print("\n-- new after (top 40) --")
    for tok, n in new.most_common(40):
        print("   %-28s %3d" % (tok, n))


if __name__ == "__main__":
    main()
