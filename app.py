import math
import os
import re
import statistics
from tokenize import TokenError

import sympy as sp
from flask import Flask, jsonify, render_template, request, session
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24)
print("STATIC FOLDER:", app.static_folder)
print("EXISTS:", os.path.exists(os.path.join(app.static_folder, "style.css")))

TRANSFORMS = standard_transformations + (implicit_multiplication, convert_xor)
VARS = {n: sp.Symbol(n) for n in "xyzt"}
MAX_LEN = 300
ALLOWED = {
    "sin", "cos", "tan", "asin", "acos", "atan", "log", "ln", "logb",
    "sqrt", "cbrt", "root", "abs", "fact", "npr", "ncr", "conj", "arg",
    "re", "im", "exp", "pi", "e", "i", "x", "y", "z", "t", "ANS", "oo",
}


class CalcError(Exception):
    pass


# ---------------------------------------------------------------- دوال مساعدة
def _fact(n):
    n = sp.sympify(n)
    if not (n.is_integer and 0 <= n <= 500):
        raise CalcError("Math ERROR")
    return sp.factorial(n)


def _npr(n, r):
    n, r = sp.sympify(n), sp.sympify(r)
    if not (n.is_integer and r.is_integer and 0 <= r <= n <= 500):
        raise CalcError("Math ERROR")
    return sp.factorial(n) / sp.factorial(n - r)


def _ncr(n, r):
    n, r = sp.sympify(n), sp.sympify(r)
    if not (n.is_integer and r.is_integer and 0 <= r <= n <= 500):
        raise CalcError("Math ERROR")
    return sp.binomial(n, r)


def build_locals(mode):
    k = sp.pi / 180 if mode == "deg" else sp.Integer(1)
    loc = dict(VARS)
    loc.update({
        "pi": sp.pi, "e": sp.E, "i": sp.I, "oo": sp.oo,
        "sin": lambda a: sp.sin(a * k),
        "cos": lambda a: sp.cos(a * k),
        "tan": lambda a: sp.tan(a * k),
        "asin": lambda a: sp.asin(a) / k,
        "acos": lambda a: sp.acos(a) / k,
        "atan": lambda a: sp.atan(a) / k,
        "ln": sp.log,
        "log": lambda a: sp.log(a, 10),
        "logb": lambda b, a: sp.log(a, b),
        "sqrt": sp.sqrt,
        "cbrt": sp.cbrt,
        "root": lambda n, a: sp.root(a, n),
        "abs": sp.Abs,
        "exp": sp.exp,
        "fact": _fact, "npr": _npr, "ncr": _ncr,
        "conj": sp.conjugate,
        "arg": lambda a: sp.arg(a) / k,
        "re": sp.re, "im": sp.im,
    })
    return loc


def get_ans():
    try:
        return sp.sympify(session.get("ans", "Integer(0)"))
    except Exception:
        return sp.Integer(0)


def normalize(text):
    s = text.strip()
    s = re.sub(r"[Aa][Nn][Ss]", "ANS", s)
    s = re.sub(r"√\s*(\d+(?:\.\d+)?|[xyzt]|π)", r"sqrt(\1)", s)
    for a, b in [
        ("sin⁻¹", "asin"), ("cos⁻¹", "acos"), ("tan⁻¹", "atan"),
        ("×", "*"), ("÷", "/"), ("−", "-"), ("–", "-"), ("·", "*"),
        ("√", "sqrt"), ("∛", "cbrt"), ("π", "pi"), ("∞", "oo"),
        ("%", "/100"), ("[", "("), ("]", ")"), ("{", "("), ("}", ")"),
    ]:
        s = s.replace(a, b)
    s = re.sub(r"[A-Za-z_][A-Za-z_0-9]*",
                lambda m: m.group(0) if m.group(0) == "ANS" else m.group(0).lower(), s)
    s = re.sub(r"\b(x|y|z|t|pi|e|i|ANS)\s*\(", r"\1*(", s)
    return s


def parse(text, mode="deg"):
    s = normalize(text)
    if not s:
        raise CalcError("Syntax ERROR")
    if len(s) > MAX_LEN or not re.fullmatch(r"[0-9A-Za-z_+\-*/^().,\s]*", s):
        raise CalcError("Syntax ERROR")
    for name in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", s):
        if name not in ALLOWED:
            raise CalcError("Syntax ERROR")
    # حماية بسيطة من الأسس الضخمة جدًا
    if re.search(r"\^\s*\(?\s*\d{5,}", s) or s.count("^") > 8:
        raise CalcError("Math ERROR")

    loc = build_locals(mode)
    loc["ANS"] = get_ans()
    try:
        return parse_expr(s, local_dict=loc, transformations=TRANSFORMS)
    except CalcError:
        raise
    except (SyntaxError, TokenError, TypeError, NameError, AttributeError):
        raise CalcError("Syntax ERROR")
    except Exception:
        raise CalcError("Math ERROR")


def pretty(expr):
    s = sp.sstr(expr)
    s = s.replace("sqrt(", "√(")
    s = re.sub(r"\bpi\b", "π", s)
    s = re.sub(r"\bI\b", "i", s)
    return s.replace("**", "^").replace("*", "·")


def fmt_real(v):
    f = float(v)
    if not math.isfinite(f):
        raise ValueError
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    s = f"{f:.10g}"
    if "e" in s:
        m, ex = s.split("e")
        return f"{m}×10^{int(ex)}"
    return s


def decimal_text(expr):
    if not expr.is_number:
        return None
    v = sp.N(expr, 15)
    r, im = v.as_real_imag()
    a, b = float(r), float(im)
    if abs(b) < 1e-12 * max(1.0, abs(a)):
        return fmt_real(a)
    if abs(a) < 1e-12 * max(1.0, abs(b)):
        return f"{fmt_real(b)}i"
    sign = "+" if b >= 0 else "−"
    return f"{fmt_real(a)} {sign} {fmt_real(abs(b))}i"


def item(label, obj, approx=True):
    out = {"label": label, "text": pretty(obj), "latex": sp.latex(obj)}
    if approx and obj.is_number and not obj.is_Integer and not obj.has(sp.Float):
        try:
            out["approx"] = decimal_text(obj)
        except Exception:
            pass
    return out

STUDENT_NAME = "  الدكتوره رحمه أجمل بنوته في الدنيا ( Roka )"   # ← غيّره لاسم بنتك
# ---------------------------------------------------------------- المسارات
@app.get("/")
def index():
    return render_template("index.html", name=STUDENT_NAME)


@app.post("/api/calc")
def api_calc():
    data = request.get_json(silent=True) or {}
    mode = "rad" if data.get("mode") == "rad" else "deg"
    try:
        expr = parse(str(data.get("expr", "")), mode)
        if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            raise CalcError("Math ERROR")
        dec = None
        try:
            dec = decimal_text(expr)
        except Exception:
            raise CalcError("Math ERROR")
        has_float = bool(expr.has(sp.Float))
        differs = dec is not None and not expr.is_Integer and not has_float
        session["ans"] = sp.srepr(expr)
        return jsonify({
            "ok": True,
            "latex": sp.latex(expr),
            "text": pretty(expr),
            "decimal": dec,
            "hasFloat": has_float,
            "differs": differs,
        })
    except CalcError as err:
        return jsonify({"ok": False, "error": str(err)})


def run_tool(d):
    tool = d.get("tool")

    def P(key):
        return parse(str(d.get(key, "")), "rad")  # التفاضل والتكامل بالراديان

    v = VARS.get(str(d.get("var", "x")).strip(), VARS["x"])
    out = []

    if tool == "derivative":
        f = P("expr")
        n = max(1, min(int(d.get("order") or 1), 5))
        res = sp.simplify(sp.diff(f, v, n))
        out.append(item("المشتقة" if n == 1 else f"المشتقة رقم {n}", res))
        if str(d.get("point", "")).strip():
            pt = P("point")
            out.append(item(f"قيمتها عند {v} = {pretty(pt)}", sp.simplify(res.subs(v, pt))))

    elif tool == "integral":
        f = P("expr")
        if str(d.get("a", "")).strip() and str(d.get("b", "")).strip():
            res = sp.integrate(f, (v, P("a"), P("b")))
            if res.has(sp.Integral):
                raise CalcError("Math ERROR")
            out.append(item("التكامل المحدد", sp.simplify(res)))
        else:
            res = sp.integrate(f, v)
            if res.has(sp.Integral):
                raise CalcError("Math ERROR")
            out.append({"label": "التكامل غير المحدد",
                        "latex": sp.latex(res) + " + C",
                        "text": pretty(res) + " + C"})

    elif tool == "limit":
        f, pt = P("expr"), P("point")
        direction = d.get("dir", "+-")
        if direction not in ("+", "-", "+-"):
            direction = "+-"
        out.append(item("النهاية", sp.limit(f, v, pt, direction)))

    elif tool == "solve":
        eqs = []
        for part in re.split(r"[;\n]+", str(d.get("expr", ""))):
            part = part.strip()
            if not part:
                continue
            if "=" in part:
                left, right = part.split("=", 1)
                eqs.append(sp.Eq(parse(left, "rad"), parse(right, "rad")))
            else:
                eqs.append(parse(part, "rad"))
        if not eqs or len(eqs) > 3:
            raise CalcError("Syntax ERROR")
        syms = sorted(set().union(*[sp.sympify(e).free_symbols for e in eqs]), key=str)
        if not syms:
            raise CalcError("Syntax ERROR")
        sols = sp.solve(eqs, syms, dict=True)
        if not sols:
            out.append({"label": "النتيجة", "text": "لا يوجد حل"})
        for idx, sol in enumerate(sols, 1):
            latex = r",\ ".join(f"{sp.latex(k)} = {sp.latex(val)}" for k, val in sol.items())
            text = " , ".join(f"{k} = {pretty(val)}" for k, val in sol.items())
            row = {"label": f"الحل {idx}", "latex": latex, "text": text}
            if len(sol) == 1:
                val = list(sol.values())[0]
                if val.is_number and not val.is_Integer:
                    try:
                        row["approx"] = decimal_text(val)
                    except Exception:
                        pass
            out.append(row)

    elif tool == "quadratic":
        a, b, c = P("a"), P("b"), P("c")
        if a.is_zero:
            raise CalcError("a ≠ 0")
        disc = sp.simplify(b**2 - 4 * a * c)
        out.append(item("المميز Δ = b² − 4ac", disc))
        if disc.is_real:
            kind = ("جذران حقيقيان مختلفان" if disc > 0
                    else "جذران حقيقيان متساويان" if disc == 0
                    else "جذران مركبان (تخيليان)")
            out.append({"label": "نوع الجذرين", "text": kind})
        for idx, root in enumerate(sp.solve(a * v**2 + b * v + c, v), 1):
            out.append(item(f"الجذر {idx}", sp.simplify(root)))
        out.append(item("مجموع الجذرين = −b/a", sp.simplify(-b / a)))
        out.append(item("حاصل ضرب الجذرين = c/a", sp.simplify(c / a)))
        h = sp.simplify(-b / (2 * a))
        k = sp.simplify(a * h**2 + b * h + c)
        out.append({"label": "رأس المنحنى",
                    "latex": rf"\left({sp.latex(h)},\ {sp.latex(k)}\right)",
                    "text": f"({pretty(h)} , {pretty(k)})"})

    elif tool == "algebra":
        op = d.get("op")
        if op not in ("factor", "expand", "simplify"):
            raise CalcError("Syntax ERROR")
        labels = {"factor": "التحليل", "expand": "النشر", "simplify": "التبسيط"}
        out.append(item(labels[op], getattr(sp, op)(P("expr")), approx=False))

    elif tool == "stats":
        raw = [x for x in re.split(r"[,\s;،]+", str(d.get("data", "")).strip()) if x]
        nums = [float(x) for x in raw]
        if not nums or not all(math.isfinite(x) for x in nums) or len(nums) > 2000:
            raise CalcError("Syntax ERROR")
        n = len(nums)
        modes = statistics.multimode(nums)
        rows = [
            ("العدد n", n), ("المجموع Σx", sum(nums)),
            ("الوسط الحسابي x̄", statistics.mean(nums)),
            ("الوسيط", statistics.median(nums)),
            ("أكبر قيمة", max(nums)), ("أصغر قيمة", min(nums)),
            ("المدى", max(nums) - min(nums)),
            ("التباين (مجتمع) σ²", statistics.pvariance(nums)),
            ("الانحراف المعياري (مجتمع) σ", statistics.pstdev(nums)),
        ]
        if n > 1:
            rows += [("التباين (عينة) s²", statistics.variance(nums)),
                        ("الانحراف المعياري (عينة) s", statistics.stdev(nums))]
        for label, val in rows:
            out.append({"label": label, "text": fmt_real(val)})
        out.append({"label": "المنوال",
                    "text": "لا يوجد" if (len(modes) == n and n > 1)
                    else " , ".join(fmt_real(m) for m in modes)})
    else:
        raise CalcError("Syntax ERROR")
    return out


@app.post("/api/tool")
def api_tool():
    data = request.get_json(silent=True) or {}
    try:
        return jsonify({"ok": True, "items": run_tool(data)})
    except CalcError as err:
        return jsonify({"ok": False, "error": str(err)})
    except Exception:
        return jsonify({"ok": False, "error": "Math ERROR"})


if __name__ == "__main__":
    app.run(debug=True)