"""Four post-restructure audits for the no-SI submission.

Read-only. With no supplementary file and no appendix, four things must hold:

  A. No pointer to a supplementary or appendix object survives anywhere.
  B. Every figure and table defined is referred to at least once.
  C. Every top-level section reaches at least one figure or table, so no
     claim-bearing section is text-only.
  D. No sentence has drifted back to claiming a controlled execution-stage
     verifier effect, which the protocol does not support.

Run: python -u audit_no_si.py [file.tex]
"""
import re
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "manuscript_p2_optical_engineering.tex"

FORBIDDEN = [
    "Supplementary Table", "Supplementary Tables", "Supplementary Figure",
    "Supplementary Note", "supplementary material", "see the SI",
    "\\appendix", "Appendix~", "Appendix ",
    "pure verifier swap", "same null-to-sign swap",
    "isolates the execution", "isolate the execution",
    "controlled execution-stage verifier", "execution-only causal",
    "the verifier effect alone",
]


def main():
    text = open(PATH, encoding="utf-8").read()

    print("=" * 68)
    print("A. pointers to supplementary / appendix objects")
    print("=" * 68)
    bad = 0
    for tok in FORBIDDEN:
        n = len(re.findall(re.escape(tok), text))
        if n:
            bad += n
            print("   %-38s %d" % (tok, n))
    if not bad:
        print("   none found")
    print("   (note: 'execution-stage verifier swap' is expected and correct;")
    print("    only the *positive* claim that such a swap is controlled is a defect)")

    print()
    print("=" * 68)
    print("B. defined figures/tables and how often they are referenced")
    print("=" * 68)
    figs = re.findall(r"\\label\{(fig:[^}]+)\}", text)
    tabs = re.findall(r"\\label\{(tab:[^}]+)\}", text)
    for kind, names in (("figure", figs), ("table", tabs)):
        for nm in names:
            n = len(re.findall(r"\\ref\{%s\}" % re.escape(nm), text))
            flag = "  <-- NEVER REFERENCED" if n == 0 else ""
            print("   %-8s %-26s refs=%d%s" % (kind, nm, n, flag))
    print("   figures: %d   tables: %d" % (len(figs), len(tabs)))

    print()
    print("=" * 68)
    print("C. figure/table support per top-level section")
    print("=" * 68)
    heads = [(m.start(), m.group(1))
             for m in re.finditer(r"\\section\{([^}]+)\}", text)]
    for k, (pos, name) in enumerate(heads):
        end = heads[k + 1][0] if k + 1 < len(heads) else len(text)
        seg = text[pos:end]
        nf = len(re.findall(r"\\ref\{fig:[^}]+\}", seg))
        nt = len(re.findall(r"\\ref\{tab:[^}]+\}", seg))
        flag = "  <-- TEXT ONLY" if nf + nt == 0 else ""
        print("   %-46s fig=%d tab=%d%s" % (name[:46], nf, nt, flag))

    print()
    print("=" * 68)
    print("D. protocol-level vs execution-stage phrasing")
    print("=" * 68)
    for tok in ("protocol-level", "execution-stage", "warm-up", "warmup"):
        print("   %-22s %d" % (tok, len(re.findall(re.escape(tok), text))))


if __name__ == "__main__":
    main()
