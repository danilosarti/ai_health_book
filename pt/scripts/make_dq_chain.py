"""Draw the decision-quality chain (six interlocking links) as inline SVG for each language
and replace the figure block in chNN(.lang).qmd."""
import math, re, sys

L = {
 "": dict(lab=[["1.", "Appropriate", "frame"], ["2.", "Creative, doable", "alternatives"], ["3.", "Meaningful,", "reliable information"],
               ["4.", "Clear values", "and trade-offs"], ["5.", "Logically correct", "reasoning"], ["6.", "Commitment", "to action"]],
          center=["Elements of", "decision", "quality"],
          title="The six elements of decision quality as a chain",
          desc="Six interlocking links form a closed chain around the words elements of decision quality: 1 appropriate frame, 2 creative, doable alternatives, 3 meaningful, reliable information, 4 clear values and trade-offs, 5 logically correct reasoning, 6 commitment to action. Link 3, information, is shaded because it is where models contribute. A decision is only as strong as its weakest link.",
          cap="The decision-quality chain [adapted from @dq]. A decision is only as strong as its weakest link; link 3, where models supply information, is shaded."),
 "pt": dict(lab=[["1.", "Enquadramento", "adequado"], ["2.", "Alternativas criativas", "e viáveis"], ["3.", "Informação relevante", "e confiável"],
                 ["4.", "Valores e", "compensações claros"], ["5.", "Raciocínio", "logicamente correto"], ["6.", "Compromisso", "com a ação"]],
          center=["Elementos da", "qualidade da", "decisão"],
          title="Os seis elementos da qualidade da decisão como uma corrente",
          desc="Seis elos entrelaçados formam uma corrente fechada em torno das palavras elementos da qualidade da decisão: 1 enquadramento adequado, 2 alternativas criativas e viáveis, 3 informação relevante e confiável, 4 valores e compensações claros, 5 raciocínio logicamente correto, 6 compromisso com a ação. O elo 3, informação, aparece sombreado porque é onde os modelos contribuem. Uma decisão é tão forte quanto o seu elo mais fraco.",
          cap="A corrente da qualidade da decisão [adaptado de @dq]. Uma decisão é tão forte quanto o seu elo mais fraco; o elo 3, onde os modelos fornecem informação, aparece sombreado."),
 "it": dict(lab=[["1.", "Inquadramento", "appropriato"], ["2.", "Alternative creative", "e praticabili"], ["3.", "Informazioni pertinenti", "e affidabili"],
                 ["4.", "Valori e", "compromessi chiari"], ["5.", "Ragionamento", "logicamente corretto"], ["6.", "Impegno", "ad agire"]],
          center=["Elementi della", "qualità", "decisionale"],
          title="I sei elementi della qualità decisionale come una catena",
          desc="Sei anelli intrecciati formano una catena chiusa attorno alle parole elementi della qualità decisionale: 1 inquadramento appropriato, 2 alternative creative e praticabili, 3 informazioni pertinenti e affidabili, 4 valori e compromessi chiari, 5 ragionamento logicamente corretto, 6 impegno ad agire. L'anello 3, le informazioni, è ombreggiato perché è lì che contribuiscono i modelli. Una decisione è forte quanto il suo anello più debole.",
          cap="La catena della qualità decisionale [adattato da @dq]. Una decisione è forte quanto il suo anello più debole; l'anello 3, dove i modelli forniscono informazioni, è ombreggiato."),
}

CX, CY, R, A, B = 350, 245, 158, 104, 70

def link_geom(k):
    th = math.radians(150 + 60 * k)
    c = (CX + R * math.cos(th), CY + R * math.sin(th))
    t = (-math.sin(th), math.cos(th))   # tangent
    r = (math.cos(th), math.sin(th))    # radial (outward)
    return c, t, r

def half(k, outer):
    c, t, r = link_geom(k)
    rng = [-0.06 + (math.pi + 0.12) * i / 48 for i in range(49)]
    pts = []
    for phi in rng:
        s = math.sin(phi) if outer else -math.sin(phi)
        x = c[0] + A * math.cos(phi) * t[0] + B * s * r[0]
        y = c[1] + A * math.cos(phi) * t[1] + B * s * r[1]
        pts.append(f"{x:.1f},{y:.1f}")
    return "M" + " L".join(pts)

def svg(d):
    o = ['<figure id="fig-dq-cycle">',
         '<svg class="aihs-svg dq-chain" viewBox="0 0 700 490" role="img" aria-labelledby="dq-title dq-desc">',
         f'  <title id="dq-title">{d["title"]}</title>', f'  <desc id="dq-desc">{d["desc"]}</desc>']
    c, t, r = link_geom(2)  # link 3 shaded
    ang = math.degrees(math.atan2(t[1], t[0]))
    o.append(f'  <ellipse cx="{c[0]:.1f}" cy="{c[1]:.1f}" rx="{A-8}" ry="{B-8}" transform="rotate({ang:.1f} {c[0]:.1f} {c[1]:.1f})" class="dq-fill"/>')
    # interlocking: even links put the inner half behind, odd links the outer half behind
    back, front = [], []
    for k in range(6):
        inner_p, outer_p = half(k, False), half(k, True)
        if k % 2 == 0: back.append(inner_p); front.append(outer_p)
        else: back.append(outer_p); front.append(inner_p)
    for p in back + front:
        o.append(f'  <path d="{p}" class="dq-edge"/>')
        o.append(f'  <path d="{p}" class="dq-link"/>')
    for k in range(6):
        c, _, _ = link_geom(k)
        n, *lines = d["lab"][k]
        words = " ".join(lines).split()
        rows, cur = [], ""
        for w in words:
            if len(cur) + len(w) + 1 > 13 and cur: rows.append(cur); cur = w
            else: cur = (cur + " " + w).strip()
        rows.append(cur)
        y0 = c[1] - 8 * len(rows)
        o.append(f'  <text x="{c[0]:.1f}" y="{y0:.1f}" text-anchor="middle" font-size="13" font-weight="700">{n}</text>')
        for i, row in enumerate(rows):
            o.append(f'  <text x="{c[0]:.1f}" y="{y0 + 16 * (i + 1):.1f}" text-anchor="middle" font-size="12">{row}</text>')
    for i, w in enumerate(d["center"]):
        o.append(f'  <text x="{CX}" y="{CY-22+i*26}" text-anchor="middle" font-size="21" font-weight="700">{w}</text>')
    o += ['</svg>', '</figure>']
    return "```{=html}\n" + "\n".join(o) + "\n```\n\n::: {.aihs-figcap}\n" + d["cap"] + "\n:::\n"

for sfx, d in L.items():
    p = "ch01.qmd" if not sfx else f"ch01.{sfx}.qmd"
    s = open(p).read()
    s2 = re.sub(r"```\{=html\}\n<figure id=\"fig-dq-cycle\">.*?```\n\n::: \{\.aihs-figcap\}\n.*?\n:::\n", lambda m: svg(d), s, count=1, flags=re.S)
    assert s2 != s, p
    open(p, "w").write(s2); print("updated", p)
