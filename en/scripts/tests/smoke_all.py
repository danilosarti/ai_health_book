"""Load every chapter in every language; report page errors, OJS errors and horizontal overflow.
Needs a server on localhost:8765 serving _book/. Usage: python3 scripts/tests/smoke_all.py"""
import asyncio, re
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        for w in (1280, 390):
            pg = await b.new_page(viewport={"width": w, "height": 900})
            for lang in ["en", "pt", "it"]:
                for i in range(1, 20):
                    errs = []
                    pg.on("pageerror", lambda e: errs.append(str(e)[:120]))
                    url = f"http://localhost:8765/{lang}/ch{i:02d}.html"
                    await pg.goto(url, wait_until="networkidle", timeout=60000)
                    await pg.wait_for_timeout(1500)
                    ojs = await pg.eval_on_selector_all(".observablehq--error", "els=>els.map(e=>e.textContent.slice(0,120))")
                    over = await pg.evaluate("document.documentElement.scrollWidth - window.innerWidth")
                    qc = await pg.eval_on_selector_all(".quick-check", "e=>e.length")
                    pg.remove_listener("pageerror", pg.listeners("pageerror")[0]) if hasattr(pg,"listeners") else None
                    net = lambda x: re.search(r"webr|pyodide|Worker loading|Failed to fetch|OJS Runtime Error$", x)
                    errs = [e for e in errs if not net(e)]; ojs = [o for o in ojs if not net(o)]
                    bad = errs or ojs or over > 2
                    print(("BAD " if bad else "ok  "), w, lang, i, "qc", qc, "overflow", over, errs[:2], ojs[:2])
            await pg.close()
        await b.close()
asyncio.run(main())
