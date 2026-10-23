"""Draw the decision-quality chain of Chapter 1 with the role AI can play in each link
(Chapter 19) and write it into ch19.qmd between the markers <!-- dq19:start --> and <!-- dq19:end -->.
Geometry copied from scripts/make_dq_chain.py so that the two figures match."""
import math, re

LAB = [["1.", "Appropriate", "frame"], ["2.", "Creative, doable", "alternatives"], ["3.", "Meaningful,", "reliable information"],
       ["4.", "Clear values", "and trade-offs"], ["5.", "Logically correct", "reasoning"], ["6.", "Commitment", "to action"]]
ROLE = ["prompts", "supports", "strengthens", "prompts", "strengthens", "prompts"]
TAG = {"strengthens": "AI strengthens", "supports": "AI supports", "prompts": "people lead; AI prompts"}
CX, CY, R, A, B = 350, 275, 158, 104, 70


def link_geom(k):
    th = math.radians(150 + 60 * k)
    c = (CX + R * math.cos(th), CY + R * math.sin(th))
    t = (-math.sin(th), math.cos(th))
    r = (math.cos(th), math.sin(th))
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


def svg():
    o = ['<figure id="fig-ch19-dq">',
         '<svg class="aihs-svg dq-chain" viewBox="-30 0 760 560" role="img" aria-labelledby="dq19-title dq19-desc">',
         '  <title id="dq19-title">The decision-quality chain with the role AI can play in each link</title>',
         '  <desc id="dq19-desc">The six interlocking links of decision quality. Link 3, information, and link 5, reasoning, are shaded dark: '
         'AI can strengthen them. Link 2, alternatives, is shaded light: AI can support it. Links 1 frame, 4 values and 6 commitment are '
         'not shaded: people lead, and AI can at most prompt them with questions.</desc>']
    for k in range(6):
        c, t, r = link_geom(k)
        ang = math.degrees(math.atan2(t[1], t[0]))
        if ROLE[k] == "strengthens":
            o.append(f'  <ellipse cx="{c[0]:.1f}" cy="{c[1]:.1f}" rx="{A-8}" ry="{B-8}" transform="rotate({ang:.1f} {c[0]:.1f} {c[1]:.1f})" class="dq-fill"/>')
        elif ROLE[k] == "supports":
            o.append(f'  <ellipse cx="{c[0]:.1f}" cy="{c[1]:.1f}" rx="{A-8}" ry="{B-8}" transform="rotate({ang:.1f} {c[0]:.1f} {c[1]:.1f})" class="dq-fill" style="opacity:0.45"/>')
    back, front = [], []
    for k in range(6):
        inner_p, outer_p = half(k, False), half(k, True)
        if k % 2 == 0:
            back.append(inner_p); front.append(outer_p)
        else:
            back.append(outer_p); front.append(inner_p)
    for p in back + front:
        o.append(f'  <path d="{p}" class="dq-edge"/>')
        o.append(f'  <path d="{p}" class="dq-link"/>')
    for k in range(6):
        c, _, r = link_geom(k)
        n, *lines = LAB[k]
        words = " ".join(lines).split()
        rows, cur = [], ""
        for w in words:
            if len(cur) + len(w) + 1 > 13 and cur:
                rows.append(cur); cur = w
            else:
                cur = (cur + " " + w).strip()
        rows.append(cur)
        y0 = c[1] - 8 * len(rows)
        o.append(f'  <text x="{c[0]:.1f}" y="{y0:.1f}" text-anchor="middle" font-size="13" font-weight="700">{n}</text>')
        for i, row in enumerate(rows):
            o.append(f'  <text x="{c[0]:.1f}" y="{y0 + 16 * (i + 1):.1f}" text-anchor="middle" font-size="12">{row}</text>')
        tx, ty = c[0] + r[0] * (B + 18), c[1] + r[1] * (B + 18) + 4
        anchor = "end" if r[0] < -0.3 else ("start" if r[0] > 0.3 else "middle")
        weight = "700" if ROLE[k] != "prompts" else "400"
        o.append(f'  <text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anchor}" font-size="12" font-style="italic" font-weight="{weight}">{TAG[ROLE[k]]}</text>')
    for i, w in enumerate(["AI as a", "meta-player"]):
        o.append(f'  <text x="{CX}" y="{CY - 8 + i * 26}" text-anchor="middle" font-size="20" font-weight="700">{w}</text>')
    o += ['</svg>', '</figure>']
    return "\n".join(o)


block = ("<!-- dq19:start -->\n```{=html}\n" + svg() + "\n```\n\n::: {.aihs-figcap}\n"
         "The decision-quality chain of Chapter 1 [adapted from @dq], with the role AI can play in each link today. "
         "Dark: AI can strengthen the link; light: AI can support it; unshaded: people lead and AI can at most prompt them.\n:::\n<!-- dq19:end -->")
p = "ch19.qmd"
s = open(p, encoding="utf-8").read()
s2 = re.sub(r"<!-- dq19:start -->.*?<!-- dq19:end -->", lambda m: block, s, flags=re.S)
assert s2 != s or block in s, "markers not found"
open(p, "w", encoding="utf-8").write(s2)
print("dq chain written")
