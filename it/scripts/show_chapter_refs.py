"""Quarto books hide each chapter's bibliography and link citations to the book's References page.
This book shows each chapter's own reference list: un-hide div#refs in chapter pages and point
citation links to the local list. Usage: python3 scripts/show_chapter_refs.py _book/en"""
import sys, glob, re, os
pages = [p for p in glob.glob(os.path.join(sys.argv[1], "*.html")) if not p.endswith(("references.html", "index.html"))]
for f in pages:
    s = open(f, encoding="utf8").read()
    s2 = s.replace('<div id="refs" class="references csl-bib-body hanging-indent" data-entry-spacing="0" role="list" style="display: none">',
                   '<div id="refs" class="references csl-bib-body hanging-indent" data-entry-spacing="0" role="list">')
    s2 = re.sub(r'href="(?:\./)?references\.html(#ref-[^"]+)"', r'href="\1"', s2)
    if s2 != s:
        open(f, "w", encoding="utf8").write(s2)
