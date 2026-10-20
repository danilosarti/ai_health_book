"""Structural check: a translated chapter must keep the same ids, cross-reference keys,
citations, code chunks (count and engine) and exercise labels as the English source.
Usage: python3 scripts/tests/check_translation.py ch01.qmd ch01.pt.qmd"""
import re, sys
a, b = open(sys.argv[1]).read(), open(sys.argv[2]).read()
# the translated pages carry one extra setup chunk that prints numbers with a decimal comma
b = re.sub(r"```\{r\}\n#\| include: false\n# decimal comma for numbers printed in the text\n.*?```\n", "", b, flags=re.S)
def keys(s):
    return {
        "ids": sorted(set(re.findall(r"#((?:sec|def|fig|tbl|eq|ch01|ch\d\d)-[A-Za-z0-9_-]+)", s))),
        "refs": sorted(set(re.findall(r"@((?:sec|def|fig|tbl|eq)-[A-Za-z0-9_-]+)", s))),
        "cites": sorted(set(re.findall(r"@([A-Za-z][A-Za-z0-9_]+\d{2,4}[a-z_]*|Bell88|White1970|dq|sarti2013|Howard\w+)", s)) - set()),
        "chunks": re.findall(r"```\{(webr|pyodide|ojs|r)\}", s),
        "exercises": sorted(set(re.findall(r"exercise: (\w+)", s))),
        "classes": sorted(re.findall(r"\{\.(quick-check|challenge|mission|tryit|gate|gated-content|progress-panel|starting-here|panel-tabset)", s)),
    }
ka, kb = keys(a), keys(b)
ok = True
for k in ka:
    if ka[k] != kb[k]:
        ok = False
        print(f"MISMATCH {k}:\n  only in source: {sorted(set(ka[k]) - set(kb[k]))}\n  only in translation: {sorted(set(kb[k]) - set(ka[k]))}\n  counts {len(ka[k])} vs {len(kb[k])}")
links = re.findall(r"\]\(((?!https?:|#|data/|scripts/|images/)[^)#]+\.qmd)", b)
bad = [l for l in links if not re.search(r"\.(pt|it)\.qmd$", l)]
if bad: ok = False; print("Links still pointing to English files:", sorted(set(bad)))
print("OK" if ok else "FAILED")
