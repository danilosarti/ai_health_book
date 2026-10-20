#!/usr/bin/env bash
# Render the book in English, Portuguese and Italian, then add the language chooser at _book/.
set -euo pipefail
cd "$(dirname "$0")/.."
export LC_ALL="${LC_ALL:-C.UTF-8}" LANG="${LANG:-C.UTF-8}"   # accented text in R-generated tables
rm -rf .site _book
for lang in en pt es it; do
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
:root{--bg:#f4f7f8;--fg:#13262e;--accent:#1f6f8b;--card:#fff;--line:#cfdde2;--muted:#4a5d65}
@media (prefers-color-scheme:dark){:root{--bg:#0e1a1f;--fg:#e3eef2;--accent:#6cc3e0;--card:#15262d;--line:#2a434c;--muted:#a9bcc4;color-scheme:dark}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;padding:48px 16px}
main{max-width:32rem;margin:0 auto;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:28px}
h1{font-size:1.35rem;margin:0 0 .4rem}p{margin:.2rem 0 1rem;color:var(--muted)}
ul{list-style:none;padding:0;margin:0;display:grid;gap:.6rem}
a.lang{display:flex;justify-content:space-between;align-items:baseline;padding:.7rem .9rem;border:1px solid var(--line);border-radius:8px;color:var(--accent);text-decoration:none;font-weight:600}
a.lang small{font-weight:400;color:var(--muted)}
a.lang:hover,a.lang:focus-visible,a.lang.last{border-color:var(--accent)}
.foot{margin-top:1.2rem;font-size:.9rem}.foot a{color:var(--accent)}
</style></head>
<body><main>
<h1>AI and Machine Learning for Health Sciences</h1>
<p>Choose a language · Escolha o idioma · Elige el idioma · Scegli la lingua</p>
<ul>
<li><a class="lang" href="en/index.html" lang="en" data-l="en">English <small>Interactive edition</small></a></li>
<li><a class="lang" href="pt/index.html" lang="pt" data-l="pt">Português <small>Edição interativa</small></a></li>
<li><a class="lang" href="es/index.html" lang="es" data-l="es">Español <small>Edición interactiva</small></a></li>
<li><a class="lang" href="it/index.html" lang="it" data-l="it">Italiano <small>Edizione interattiva</small></a></li>
</ul>
<p class="foot">You can switch language on any page (EN · PT · ES · IT, top right). · Danilo Augusto Sarti<br>
<a href="v1/">First edition</a></p>
</main>
<script>
(function(){try{var l=localStorage.getItem("aihs:lang");if(l){var a=document.querySelector('a[data-l="'+l+'"]');if(a)a.classList.add("last");}}catch(e){}})();
</script></body></html>
HTML
echo "Built _book/{en,pt,es,it} and _book/index.html"
