#!/usr/bin/env bash
# Render the book in English, Portuguese and Italian, then add the language chooser at _book/.
set -euo pipefail
cd "$(dirname "$0")/.."
export LC_ALL="${LC_ALL:-C.UTF-8}" LANG="${LANG:-C.UTF-8}"   # accented text in R-generated tables
rm -rf .site _book
for lang in en pt it; do
  quarto render --profile "$lang"
  python3 scripts/show_chapter_refs.py "_book/$lang"   # show each chapter's own reference list
  mkdir -p .site && mv "_book/$lang" ".site/$lang"   # keep finished languages out of the next render
done
rm -rf _book && mv .site _book
cat > _book/index.html <<'HTML'
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI and Machine Learning for Health Sciences</title>
<style>
:root{--bg:#f4f7f8;--fg:#13262e;--accent:#1f6f8b;--card:#fff;--line:#cfdde2}
@media (prefers-color-scheme:dark){:root{--bg:#0e1a1f;--fg:#e3eef2;--accent:#6cc3e0;--card:#15262d;--line:#2a434c;color-scheme:dark}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;padding:48px 16px}
main{max-width:30rem;margin:0 auto;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:28px}
h1{font-size:1.35rem;margin:0 0 1rem}ul{list-style:none;padding:0;margin:0;display:grid;gap:.6rem}
a{display:block;padding:.6rem .9rem;border:1px solid var(--line);border-radius:8px;color:var(--accent);text-decoration:none;font-weight:600}
a:hover,a:focus-visible{border-color:var(--accent)}
</style></head>
<body><main>
<h1>AI and Machine Learning for Health Sciences</h1>
<ul>
<li><a href="en/index.html" lang="en">English</a></li>
<li><a href="pt/index.html" lang="pt">Português</a></li>
<li><a href="it/index.html" lang="it">Italiano</a></li>
</ul></main>
<script>
(function(){try{var saved=localStorage.getItem("aihs:lang");var l=saved||(navigator.language||"en").slice(0,2);
if(["en","pt","it"].indexOf(l)>=0&&!/[?&]choose/.test(location.search)){location.replace(l+"/index.html");}}catch(e){}})();
</script></body></html>
HTML
echo "Built _book/{en,pt,it} and _book/index.html"
