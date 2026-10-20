"""Mind map of the whole book (inline SVG, theme-aware, three languages).
Writes _includes/mindmap.<lang>.qmd; chapters listed in READY become links."""
import html
READY = {f"ch{i:02d}" for i in range(1, 20)}
B = {
 "en": dict(center=["AI and machine learning", "for health sciences"], tag="the decision loop",
   parts=[("I · Foundations", [("ch01","1 Data, models and decisions"),("ch02","2 Uncertainty and probability"),("ch03","3 Learning from data: the workflow")]),
          ("II · Predicting outcomes", [("ch04","4 Linear models, regularisation"),("ch05","5 Classification and thresholds"),("ch06","6 Trees and ensembles"),("ch07","7 Neural networks, deep learning"),("ch08","8 Bayesian learning"),("ch09","9 When data are missing")]),
          ("III · Finding structure", [("ch10","10 High-dimensional data"),("ch11","11 Clustering patients")]),
          ("IV · Trusting models", [("ch12","12 Interpretable AI"),("ch13","13 Prediction vs causation"),("ch14","14 Reinforcement learning")]),
          ("V · Language and generative AI", [("ch15","15 Text as data"),("ch16","16 Language models, embeddings"),("ch17","17 Attention, transformers"),("ch18","18 From LLM to clinical assistant"),("ch19","19 Generative models beyond text")]),
          ("Appendices", [("setup","A Setup: R, Python, Git, GitHub"),("data-dictionary","B Data"),("glossary","C Glossary"),("notation","D Notation")])],
   title="Mind map of the book", desc="The book's central theme, AI and machine learning for health sciences seen through the decision loop, branches into five parts and the appendices, each listing its chapters."),
 "pt": dict(center=["IA e aprendizado de máquina", "para as ciências da saúde"], tag="o ciclo de decisão",
   parts=[("I · Fundamentos", [("ch01","1 Dados, modelos e decisões"),("ch02","2 Incerteza e probabilidade"),("ch03","3 Aprender com dados")]),
          ("II · Prever desfechos", [("ch04","4 Modelos lineares, regularização"),("ch05","5 Classificação e limiares"),("ch06","6 Árvores e comitês"),("ch07","7 Redes neurais, deep learning"),("ch08","8 Aprendizado bayesiano"),("ch09","9 Quando faltam dados")]),
          ("III · Encontrar estrutura", [("ch10","10 Dados de alta dimensão"),("ch11","11 Agrupamento de pacientes")]),
          ("IV · Confiar nos modelos", [("ch12","12 IA interpretável"),("ch13","13 Predição × causalidade"),("ch14","14 Aprendizado por reforço")]),
          ("V · Linguagem e IA generativa", [("ch15","15 Texto como dado"),("ch16","16 Modelos de linguagem"),("ch17","17 Atenção e transformers"),("ch18","18 Do LLM ao assistente clínico"),("ch19","19 Generativos além do texto")]),
          ("Apêndices", [("setup","A Configuração: R, Python, Git"),("data-dictionary","B Dados"),("glossary","C Glossário"),("notation","D Notação")])],
   title="Mapa mental do livro", desc="O tema central do livro, IA e aprendizado de máquina para as ciências da saúde vistos pelo ciclo de decisão, se ramifica em cinco partes e nos apêndices, cada um com os seus capítulos."),
 "it": dict(center=["IA e apprendimento automatico", "per le scienze della salute"], tag="il ciclo decisionale",
   parts=[("I · Fondamenti", [("ch01","1 Dati, modelli e decisioni"),("ch02","2 Incertezza e probabilità"),("ch03","3 Imparare dai dati")]),
          ("II · Prevedere gli esiti", [("ch04","4 Modelli lineari, regolarizzazione"),("ch05","5 Classificazione e soglie"),("ch06","6 Alberi e metodi d'insieme"),("ch07","7 Reti neurali, deep learning"),("ch08","8 Apprendimento bayesiano"),("ch09","9 Quando mancano i dati")]),
          ("III · Trovare strutture", [("ch10","10 Dati ad alta dimensionalità"),("ch11","11 Raggruppare i pazienti")]),
          ("IV · Fidarsi dei modelli", [("ch12","12 IA interpretabile"),("ch13","13 Predizione e causalità"),("ch14","14 Apprendimento per rinforzo")]),
          ("V · Linguaggio e IA generativa", [("ch15","15 Il testo come dato"),("ch16","16 Modelli linguistici"),("ch17","17 Attenzione e transformer"),("ch18","18 Dall'LLM all'assistente clinico"),("ch19","19 Generativi oltre il testo")]),
          ("Appendici", [("setup","A Configurazione: R, Python, Git"),("data-dictionary","B Dati"),("glossary","C Glossario"),("notation","D Notazione")])],
   title="Mappa mentale del libro", desc="Il tema centrale del libro, IA e apprendimento automatico per le scienze della salute visti attraverso il ciclo decisionale, si dirama in cinque parti e nelle appendici, ciascuna con i suoi capitoli."),
 "es": dict(center=["IA y aprendizaje automático", "para las ciencias de la salud"], tag="el ciclo de decisión",
   parts=[("I · Fundamentos", [("ch01","1 Datos, modelos y decisiones"),("ch02","2 Incertidumbre y probabilidad"),("ch03","3 Aprender de los datos")]),
          ("II · Predecir desenlaces", [("ch04","4 Modelos lineales, regularización"),("ch05","5 Clasificación y umbrales"),("ch06","6 Árboles y ensambles"),("ch07","7 Redes neuronales, deep learning"),("ch08","8 Aprendizaje bayesiano"),("ch09","9 Cuando faltan datos")]),
          ("III · Encontrar estructura", [("ch10","10 Datos de alta dimensión"),("ch11","11 Agrupar pacientes")]),
          ("IV · Confiar en los modelos", [("ch12","12 IA interpretable"),("ch13","13 Predicción y causalidad"),("ch14","14 Aprendizaje por refuerzo")]),
          ("V · Lenguaje e IA generativa", [("ch15","15 El texto como dato"),("ch16","16 Modelos de lenguaje"),("ch17","17 Atención y transformers"),("ch18","18 Del LLM al asistente clínico"),("ch19","19 Generativos más allá del texto")]),
          ("Apéndices", [("setup","A Configuración: R, Python, Git"),("data-dictionary","B Datos"),("glossary","C Glosario"),("notation","D Notación")])],
   title="Mapa mental del libro", desc="El tema central del libro, la IA y el aprendizaje automático para las ciencias de la salud vistos a través del ciclo de decisión, se ramifica en cinco partes y en los apéndices, cada una con sus capítulos."),
}
W, H, CX, CY, LH = 1600, 560, 800, 280, 24

def build(d):
    left, right = d["parts"][:3], d["parts"][3:]
    o = [f'<figure class="mindmap-fig"><svg class="aihs-svg mindmap" viewBox="0 0 {W} {H}" role="img" aria-labelledby="mm-title mm-desc">',
         f'<title id="mm-title">{html.escape(d["title"])}</title><desc id="mm-desc">{html.escape(d["desc"])}</desc>']
    def side(parts, sign, start_idx):
        n = sum(len(c) for _, c in parts) + 1.2 * (len(parts) - 1)
        y = CY - n * LH / 2 + LH / 2
        px = CX + sign * 225; lx = CX + sign * 270
        for i, (pname, chs) in enumerate(parts):
            k = start_idx + i
            ys = [y + j * LH for j in range(len(chs))]
            py = sum(ys) / len(ys)
            # centre -> part
            o.append(f'<path d="M{CX + sign*148},{CY} C{CX + sign*180},{CY} {px - sign*60},{py} {px - sign*10},{py}" class="mm-branch mm-c{k}"/>')
            tw = 8 * len(pname) + 24
            bx = px - 10 if sign > 0 else px + 10 - tw
            o.append(f'<rect x="{bx}" y="{py-15}" width="{tw}" height="30" rx="15" class="mm-part mm-c{k}"/>')
            o.append(f'<text x="{bx + tw/2}" y="{py+5}" text-anchor="middle" font-size="13" font-weight="700" class="mm-part-text">{html.escape(pname)}</text>')
            pe = bx + tw if sign > 0 else bx
            for cid, ctitle in chs:
                yy = ys[chs.index((cid, ctitle))]
                tx = pe + sign * 34
                o.append(f'<path d="M{pe},{py} C{pe + sign*18},{py} {tx - sign*18},{yy} {tx - sign*4},{yy}" class="mm-leaf-line mm-c{k}"/>')
                anchor = "start" if sign > 0 else "end"
                label = html.escape(ctitle)
                cls = "mm-leaf ready" if cid in READY else "mm-leaf"
                t = f'<text x="{tx}" y="{yy+4}" text-anchor="{anchor}" font-size="13" class="{cls}">{label}</text>'
                if cid in READY:
                    t = f'<g class="mm-link" data-href="{cid}.html" role="link" tabindex="0">{t}</g>'
                o.append(t)
            y += (len(chs) + 1.2) * LH
    side(left, -1, 0)
    side(right, 1, 3)
    o.append(f'<ellipse cx="{CX}" cy="{CY}" rx="150" ry="60" class="mm-center"/>')
    o.append(f'<text x="{CX}" y="{CY-12}" text-anchor="middle" font-size="15" font-weight="700">{html.escape(d["center"][0])}</text>')
    o.append(f'<text x="{CX}" y="{CY+8}" text-anchor="middle" font-size="15" font-weight="700">{html.escape(d["center"][1])}</text>')
    o.append(f'<text x="{CX}" y="{CY+30}" text-anchor="middle" font-size="12" class="muted">{html.escape(d["tag"])}</text>')
    o.append('</svg></figure>')
    return "```{=html}\n" + "\n".join(o) + "\n```\n"

for lang, d in B.items():
    open(f"_includes/mindmap.{lang}.qmd", "w").write(build(d))
    print("wrote", lang)
