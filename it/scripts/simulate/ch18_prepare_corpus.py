"""Prepare the Chapter 18 corpus: Florence Nightingale, *Notes on Nursing: What It Is, and What It
Is Not* (1859; this transcription follows a New York reprint of 1898), a public-domain text.

Input : the plain-text eBook #12439 as distributed at https://www.gutenberg.org/ebooks/12439
        (transcribed by Distributed Proofreaders). Download it and pass its path, e.g.
            python3 scripts/simulate/ch18_prepare_corpus.py notes_on_nursing_pg12439.txt
Output: data/notes_on_nursing.txt  (UTF-8, one paragraph per line)

What the script does, and nothing else:
  * keeps only the book text between the START and END markers of the eBook, so the distributor's
    header, footer and licence (and its name) are not shipped;
  * drops the publisher's imprint, the transcriber's notes, the footnote markers ("[1]") and the
    "FOOTNOTES:" headings (the footnote texts themselves are kept), the "* * *" separators, and the
    two statistical tables of the appendix (columns of figures that do not survive as running text);
  * turns each marginal note "[Sidenote: ...]" into a short paragraph of its own;
  * removes the underscores that mark italics;
  * joins the lines of each paragraph (a line ending in "--" or "-" is joined without a space);
    headings and the table of contents, written in capitals, keep one line each.
The author's words are otherwise unchanged.
"""
import re, sys, os

src = sys.argv[1] if len(sys.argv) > 1 else "notes_on_nursing_pg12439.txt"
out = sys.argv[2] if len(sys.argv) > 2 else "data/notes_on_nursing.txt"
raw = open(src, encoding="utf-8-sig").read().replace("\r\n", "\n")

start = raw.index("\n", raw.index("*** START OF")) + 1
end = raw.index("*** END OF")
text = raw[start:end]
text = re.sub(r"\n\s*End of .*$", "\n", text.rstrip(), flags=re.S)       # closing line of the eBook

text = re.sub(r"\[Transcriber's note:.*?\]", "", text, flags=re.S)        # transcriber's notes
text = re.sub(r"NEW YORK:\s*\nD\. APPLETON AND COMPANY\s*\n72 FIFTH AVENUE\s*\n1898\.\s*\n", "", text)
text = re.sub(r"\nTABLE A\..*?(?=\nNOTE AS TO THE NUMBER)", "\n", text, flags=re.S)   # appendix tables
text = re.sub(r"^FOOTNOTES:\s*$", "", text, flags=re.M)
text = re.sub(r"^\[\d+\]\s*$", "", text, flags=re.M)                     # "[1]" on a line of its own
text = re.sub(r"^\[\d+\]\s+", "", text, flags=re.M)                      # "[1] " opening a footnote
text = re.sub(r"\[\d+\]", "", text)                                      # inline markers
text = re.sub(r"^\s*\* \* \* \* \*\s*$", "", text, flags=re.M)
text = re.sub(r"\[Sidenote:\s*(.*?)\]", lambda m: "\n\n" + " ".join(m.group(1).split()) + "\n\n", text, flags=re.S)
text = text.replace("_", "")

paras = []
for block in re.split(r"\n\s*\n", text):
    lines = [l.strip() for l in block.split("\n") if l.strip()]
    if not lines:
        continue
    if not re.search(r"[a-z]", block):          # headings and the table of contents keep their lines
        paras.extend(re.sub(r" {2,}", " ", l) for l in lines)
        continue
    p = lines[0]
    for l in lines[1:]:
        p = p + l if p.endswith("-") else p + " " + l
    paras.append(re.sub(r" {2,}", " ", p))

result = "\n".join(paras) + "\n"
assert "gutenberg" not in result.lower(), "distributor's name left in the text"
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
open(out, "w", encoding="utf-8").write(result)
print(f"{len(paras)} paragraphs, {len(result)} characters, {len(set(result))} distinct characters -> {out}")
