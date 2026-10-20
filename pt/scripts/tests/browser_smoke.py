"""Browser smoke test of the rendered book (all languages). Needs a server on localhost:8765
serving _book/ and Playwright with Chromium. Usage: python3 scripts/tests/browser_smoke.py"""
import asyncio, re
from playwright.async_api import async_playwright

ANS = {"pt": ["dados", "dados", "modelo", "informação", "decisão"],
       "it": ["dati", "dati", "modello", "informazione", "decisione"],
       "en": ["data", "data", "model", "information", "decision"]}

async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1280, "height": 900}, locale="pt-BR")
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto("http://localhost:8765/index.html")
        await pg.wait_for_timeout(1500)
        print("root redirect ->", pg.url)
        for lang in ["pt", "it", "en"]:
            await pg.goto(f"http://localhost:8765/{lang}/ch01.html", wait_until="networkidle")
            await pg.wait_for_timeout(2000)
            labels = await pg.eval_on_selector_all(".aihs-label", "els=>[...new Set(els.map(e=>e.textContent))].slice(0,6)")
            sels = await pg.query_selector_all(".gate select")
            for s, a in zip(sels, ANS[lang]):
                await s.select_option(a)
            await pg.click(".gate .aihs-btn.primary")
            fb = await pg.inner_text(".gate .aihs-feedback")
            ch = await pg.query_selector(".challenge")
            opts = await ch.eval_on_selector("select", "s=>[...s.options].map(o=>o.value).slice(1)")
            meta = (await pg.inner_text(".progress-panel .meta")).replace("\n", " | ")
            body = await pg.inner_text("body")
            defs = re.findall(r"(Definição|Definizione|Definition) 1\.1", body)[:1]
            ojs = re.findall(r"(MAE[^\n]{0,60})", body)[:1]
            sh = await pg.query_selector(".starting-here")
            switch = await pg.eval_on_selector_all("a", "els=>els.map(e=>e.getAttribute('href')).filter(h=>h&&/^\\.\\.\\/(en|pt|it)\\//.test(h))")
            print(f"[{lang}] labels={labels}\n  gate={fb}\n  challenge={opts}\n  panel={meta}\n  def={defs} starting-here={'yes' if sh else 'NO'} ojs={ojs}\n  switch={switch}")
        print("page errors:", errs[:5])
        await b.close()

asyncio.run(main())
