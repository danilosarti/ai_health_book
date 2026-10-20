"""Test every quarto-live exercise in a chapter: the solution must pass its grader,
and each listed wrong answer must fail. R checks run through Rscript, Python checks in-process.
Usage: python3 scripts/tests/test_graders.py ch01.qmd"""
import re, sys, subprocess, json, textwrap, os, ast

src = open(sys.argv[1]).read()
chunks = re.findall(r"```\{(webr|pyodide)\}\n(.*?)```", src, re.S)
ex = {}
for eng, body in chunks:
    opts = dict(re.findall(r"^#\|\s*(\S+):\s*(.+)$", body, re.M))
    code = "\n".join(l for l in body.split("\n") if not l.startswith("#|"))
    if "exercise" not in opts: continue
    e = ex.setdefault(opts["exercise"], {"eng": eng})
    if opts.get("setup") == "true": e["setup"] = code
    elif opts.get("check") == "true": e["check"] = code
    else: e["code"] = code
for m in re.finditer(r'::: \{\.solution exercise="([^"]+)"\}\n```\w+\n(.*?)```', src, re.S):
    ex[m.group(1)]["solution"] = m.group(2)

def fill(code, sol):
    """Build learner code: if solution is a full statement replacing a line with blanks, substitute."""
    lines = code.split("\n"); sol_lines = [l for l in sol.strip().split("\n") if l.strip()]
    if len(sol_lines) == 1 and any("______" in l for l in lines):
        out = []
        for l in lines:
            if "______" in l:
                lhs = l.split("______")[0].split("#")[0]
                out.append(sol_lines[0] if sol_lines[0].split("(")[0].split("=")[0].strip() in l or len(lines) == 1 or "=" in sol_lines[0] or "<-" in sol_lines[0] else sol_lines[0])
            else: out.append(l)
        return "\n".join(out)
    return sol

def run_r(setup, learner, check):
    prog = f"""
.envir_prep <- new.env(); local({{ {setup} }}, envir = .envir_prep)
.envir_result <- new.env(parent = globalenv()); for (n in ls(.envir_prep)) assign(n, get(n, .envir_prep), .envir_result)
exprs <- parse(text = {json.dumps(learner)})
.result <- NULL; for (e in exprs) .result <- eval(e, .envir_result)
fb <- local({{ {check} }})
cat(if (isTRUE(fb$correct)) "PASS" else "FAIL", fb$message, "\\n")
"""
    r = subprocess.run(["Rscript", "-e", prog], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()

def run_py(setup, learner, check):
    env = {}
    exec(setup, env)
    tree = ast.parse(learner); last = tree.body[-1]
    if isinstance(last, ast.Expr):
        exec(compile(ast.Module(tree.body[:-1], []), "<l>", "exec"), env)
        result = eval(compile(ast.Expression(last.value), "<l>", "eval"), env)
    else:
        exec(learner, env); result = None
    cenv = dict(env); cenv["result"] = result
    ctree = ast.parse(check); clast = ctree.body[-1]
    exec(compile(ast.Module(ctree.body[:-1], []), "<c>", "exec"), cenv)
    fb = eval(compile(ast.Expression(clast.value), "<c>", "eval"), cenv)
    return ("PASS " if fb and fb.get("correct") else "FAIL ") + (fb or {}).get("message", "")

ok = True
for name, e in ex.items():
    sol = e.get("solution", "")
    if "______" in e["code"]:
        line = [l for l in e["code"].split("\n") if "______" in l][0]
        s1 = sol.strip().split("\n")[0]
        if s1.split("#")[0].strip().startswith(line.split("______")[0].split("#")[0].strip()[:8]) and ("<-" in s1 or "=" in s1.split("(")[0]):
            learner = e["code"].replace(line, s1)
        else:
            # solution is the whole expression that replaces the blank line
            learner = e["code"].replace(line, s1)
    else:
        learner = sol
    wrong = e["code"].replace("______", "median" if "mean" in sol else "0.5")
    runner = run_r if e["eng"] == "webr" else run_py
    good = runner(e.get("setup", ""), learner, e["check"])
    try: bad = runner(e.get("setup", ""), wrong, e["check"])
    except Exception as err: bad = "FAIL (error: %s)" % err
    flag = good.startswith("PASS") and not bad.startswith("PASS")
    ok &= flag
    print(("OK  " if flag else "XX  ") + name, "| solution:", good[:90], "| wrong:", bad[:70])
sys.exit(0 if ok else 1)
