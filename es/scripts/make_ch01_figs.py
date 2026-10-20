"""Theme-aware SVG versions of three v1 figures (ML families, AI/ML/predictive-generative,
anatomy of a method), in English, Portuguese and Italian. Replaces the placeholder lines
<!-- FIG:name --> in ch01*.qmd with a Quarto figure div."""
import re, sys

T = {
 "": dict(
   ml_title="Machine learning methods", ml=[("Supervised", ["trained with", "labelled outcomes"]), ("Unsupervised", ["discovering structure", "without labels,", "e.g. clustering"]), ("Reinforcement", ["learning by", "trial and reward"])],
   ml_cap="The three classic families of machine-learning methods.",
   ml_alt="A box labelled machine learning methods branches into three boxes: supervised, unsupervised and reinforcement learning.",
   ai=["Artificial", "intelligence"], ml_box=["Machine", "learning"], pred=["Predictive", "AI"], gen=["Generative", "AI"],
   ai_cap="Machine learning is one part of artificial intelligence; its applications are often grouped into predictive and generative AI.",
   ai_alt="A large box labelled artificial intelligence contains a smaller box labelled machine learning, from which arrows lead to two boxes: predictive AI and generative AI.",
   an=["Data", "Algorithm", "Performance"], an_sub=["Supervised", "Unsupervised", "Reinforcement", "Generative"],
   an_cap="The anatomy shared by every method in this book: data, an algorithm that depends on the kind of task, and a measure of performance.",
   an_alt="Boxes labelled data, algorithm and performance joined by arrows; four arrows lead down from algorithm to supervised, unsupervised, reinforcement and generative."),
 "pt": dict(
   ml_title="Métodos de aprendizado de máquina", ml=[("Supervisionado", ["treinado com", "desfechos rotulados"]), ("Não supervisionado", ["descobre estrutura", "sem rótulos,", "p. ex. agrupamento"]), ("Por reforço", ["aprende por", "tentativa e recompensa"])],
   ml_cap="As três famílias clássicas de métodos de aprendizado de máquina.",
   ml_alt="Uma caixa com o rótulo métodos de aprendizado de máquina se ramifica em três caixas: aprendizado supervisionado, não supervisionado e por reforço.",
   ai=["Inteligência", "artificial"], ml_box=["Aprendizado", "de máquina"], pred=["IA", "preditiva"], gen=["IA", "generativa"],
   ai_cap="O aprendizado de máquina é uma parte da inteligência artificial; suas aplicações costumam ser agrupadas em IA preditiva e IA generativa.",
   ai_alt="Uma caixa grande com o rótulo inteligência artificial contém uma caixa menor, aprendizado de máquina, da qual saem setas para duas caixas: IA preditiva e IA generativa.",
   an=["Dados", "Algoritmo", "Desempenho"], an_sub=["Supervisionado", "Não supervisionado", "Por reforço", "Generativo"],
   an_cap="A anatomia comum a todos os métodos deste livro: dados, um algoritmo que depende do tipo de tarefa e uma medida de desempenho.",
   an_alt="Caixas com os rótulos dados, algoritmo e desempenho ligadas por setas; quatro setas descem de algoritmo para supervisionado, não supervisionado, por reforço e generativo."),
 "it": dict(
   ml_title="Metodi di apprendimento automatico", ml=[("Supervisionato", ["addestrato con", "esiti etichettati"]), ("Non supervisionato", ["scopre strutture", "senza etichette,", "es. clustering"]), ("Per rinforzo", ["impara per", "tentativi e ricompense"])],
   ml_cap="Le tre famiglie classiche di metodi di apprendimento automatico.",
   ml_alt="Un riquadro con l'etichetta metodi di apprendimento automatico si dirama in tre riquadri: apprendimento supervisionato, non supervisionato e per rinforzo.",
   ai=["Intelligenza", "artificiale"], ml_box=["Apprendimento", "automatico"], pred=["IA", "predittiva"], gen=["IA", "generativa"],
   ai_cap="L'apprendimento automatico è una parte dell'intelligenza artificiale; le sue applicazioni si raggruppano spesso in IA predittiva e IA generativa.",
   ai_alt="Un grande riquadro con l'etichetta intelligenza artificiale contiene un riquadro più piccolo, apprendimento automatico, da cui partono frecce verso due riquadri: IA predittiva e IA generativa.",
   an=["Dati", "Algoritmo", "Prestazioni"], an_sub=["Supervisionato", "Non supervisionato", "Per rinforzo", "Generativo"],
   an_cap="L'anatomia comune a tutti i metodi di questo libro: dati, un algoritmo che dipende dal tipo di compito e una misura delle prestazioni.",
   an_alt="Riquadri con le etichette dati, algoritmo e prestazioni collegati da frecce; quattro frecce scendono da algoritmo verso supervisionato, non supervisionato, per rinforzo e generativo."),
}

def lines(x, y, rows, size=14, bold_first=False, gap=19):
    out = []
    for i, r in enumerate(rows):
        w = ' font-weight="700"' if (bold_first and i == 0) else ''
        out.append(f'<text x="{x}" y="{y + i*gap}" text-anchor="middle" font-size="{size}"{w}>{r}</text>')
    return "\n".join(out)

def wrap(name, svg, cap, alt):
    return (f"::: {{#{name}}}\n```{{=html}}\n<div role=\"img\" aria-label=\"{alt}\">\n{svg}\n</div>\n```\n\n{cap}\n:::\n")

def fig_ml(d):
    s = ['<svg class="aihs-svg" viewBox="0 0 720 300" aria-hidden="true" style="max-width:680px">']
    s.append('<rect x="160" y="15" width="400" height="62" rx="12" class="box-accent"/>')
    s.append(lines(360, 53, [d["ml_title"]], size=18, bold_first=True))
    s.append('<path d="M360,77 L360,112" class="arrow" style="marker-end:none"/>')
    s.append('<path d="M120,112 L600,112" class="arrow" style="marker-end:none"/>')
    xs = [120, 360, 600]
    for x in xs:
        s.append(f'<path d="M{x},112 L{x},146" class="arrow"/>')
    for (t, sub), x in zip(d["ml"], xs):
        s.append(f'<rect x="{x-110}" y="150" width="220" height="130" rx="12" class="box"/>')
        s.append(lines(x, 185, [t], size=17, bold_first=True))
        s.append(lines(x, 214, sub, size=14))
    s.append('</svg>')
    return wrap("fig-ml-classes", "\n".join(s), d["ml_cap"], d["ml_alt"])

def fig_ai(d):
    s = ['<svg class="aihs-svg" viewBox="0 0 640 360" aria-hidden="true" style="max-width:600px">']
    s.append('<rect x="20" y="20" width="300" height="320" rx="22" class="box"/>')
    s.append(lines(170, 70, d["ai"], size=24, bold_first=False, gap=30).replace('font-size="24"', 'font-size="24" font-weight="700"'))
    s.append('<rect x="60" y="150" width="220" height="160" rx="20" class="box-accent"/>')
    s.append(lines(170, 222, d["ml_box"], size=21, gap=28).replace('font-size="21"', 'font-size="21" font-weight="700"'))
    s.append('<path d="M280,230 L380,230" class="arrow" style="marker-end:none"/>')
    s.append('<path d="M380,230 L380,105 L425,105" class="arrow"/>')
    s.append('<path d="M380,230 L380,265 L425,265" class="arrow"/>')
    for y, lab in [(60, d["pred"]), (220, d["gen"])]:
        s.append(f'<rect x="430" y="{y}" width="190" height="90" rx="12" class="box"/>')
        s.append(lines(525, y + 38, lab, size=18, gap=24).replace('font-size="18"', 'font-size="18" font-weight="700"'))
    s.append('</svg>')
    return wrap("fig-ai-ml", "\n".join(s), d["ai_cap"], d["ai_alt"])

def fig_anat(d):
    s = ['<svg class="aihs-svg" viewBox="0 0 760 260" aria-hidden="true">']
    boxes = [(20, d["an"][0]), (290, d["an"][1]), (560, d["an"][2])]
    for x, lab in boxes:
        s.append(f'<rect x="{x}" y="20" width="180" height="70" rx="6" class="box{"-accent" if x == 290 else ""}" style="stroke-width:2.2"/>')
        s.append(lines(x + 90, 63, [lab], size=22, bold_first=True))
    s.append('<path d="M200,55 L287,55" class="arrow"/>')
    s.append('<path d="M470,55 L557,55" class="arrow"/>')
    xs = [100, 290, 470, 660]
    for x, lab in zip(xs, d["an_sub"]):
        s.append(f'<path d="M380,90 L{x},195" class="arrow"/>')
        s.append(lines(x, 222, [lab], size=16, bold_first=True))
    s.append('</svg>')
    return wrap("fig-anatomy", "\n".join(s), d["an_cap"], d["an_alt"])

for sfx, d in T.items():
    p = "ch01.qmd" if not sfx else f"ch01.{sfx}.qmd"
    s = open(p).read()
    for key, fn in [("ml-classes", fig_ml), ("ai-ml", fig_ai), ("anatomy", fig_anat)]:
        tag = f"<!-- FIG:{key} -->"
        if tag in s:
            s = s.replace(tag, fn(d))
        else:
            print(f"warning: {tag} not found in {p}")
    open(p, "w").write(s)
    print("figures written into", p)
