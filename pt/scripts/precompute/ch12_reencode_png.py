"""Re-encode the figures made by scripts/precompute/ch12_vivid_bartman.R with PIL:
RGB, longest side <= 900 px, optimised PNG, and no embedded metadata (text chunks, ICC, dpi).
Usage (from the book folder):  python3 scripts/precompute/ch12_reencode_png.py"""
from PIL import Image

FILES = ["ch12_vivid_heatmap.png", "ch12_vivid_network.png", "ch12_vivid_pdppairs.png",
         "ch12_bart_vsup.png"]
for f in FILES:
    path = "images/" + f
    im = Image.open(path).convert("RGB")
    s = 900 / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    clean = Image.frombytes("RGB", im.size, im.tobytes())   # pixels only: no chunks, no info
    clean.save(path, format="PNG", optimize=True)
    print(path, clean.size)
