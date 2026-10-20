"""Write the downloadable Lab scripts of a chapter (all languages) from the chapter sources."""
import sys, os
HDR = {"": "# Chapter {n} Lab\n# AI and Machine Learning for Health Sciences\n# Run from the book folder (the one that contains data/), or change the paths.\n",
       "pt": "# Laboratório do capítulo {n}\n# Inteligência Artificial e Aprendizado de Máquina para as Ciências da Saúde\n# Execute a partir da pasta do livro (a que contém data/) ou ajuste os caminhos.\n",
       "it": "# Laboratorio del capitolo {n}\n# Intelligenza Artificiale e Apprendimento Automatico per le Scienze della Salute\n# Esegui dalla cartella del libro (quella che contiene data/) o modifica i percorsi.\n",
       "es": "# Laboratorio del capítulo {n}\n# Inteligencia Artificial y Aprendizaje Automático para las Ciencias de la Salud\n# Ejecútalo desde la carpeta del libro (la que contiene data/) o ajusta las rutas.\n"}
ch = sys.argv[1]  # e.g. ch01
n = int(ch[2:])
for lang in ["", "pt", "it", "es"]:
    import os as _os
    if lang and not _os.path.exists(f"{ch}.{lang}.qmd"): continue
    src = f"{ch}.qmd" if not lang else f"{ch}.{lang}.qmd"
    if not os.path.exists(src): continue
    s = open(src).read()
    lab = s.split('{#sec-lab}')[1].split('{#sec-exercises}')[0]
    r = lab.split('```{webr}')[1].split('```')[0].replace('#| autorun: false\n', '')
    py = lab.split('```{pyodide}')[1].split('```')[0].replace('#| autorun: false\n', '')
    lines = py.rstrip().split('\n')
    if lines[-1] and not lines[-1].startswith((' ', '#', 'print')):
        lines[-1] = 'print(' + lines[-1] + ')'
    py = '\n'.join(lines) + '\n'
    sfx = f".{lang}" if lang else ""
    h = HDR[lang].format(n=n)
    open(f"scripts/labs/{ch}_lab{sfx}.R", "w").write(h + r.lstrip('\n'))
    open(f"scripts/labs/{ch}_lab{sfx}.py", "w").write(h + py.lstrip('\n'))
    print("wrote", f"{ch}_lab{sfx}.R/.py")
