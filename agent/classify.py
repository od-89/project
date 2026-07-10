"""Zero-cost task classifier: regex/keyword routing across the 8 capability
categories. Misclassification is non-fatal — every handler is LLM-backed and
answers the raw prompt — but good routing picks the right verification tools.
"""
import re

CATEGORIES = (
    "ner", "sentiment", "summarize", "code_debug", "code_gen",
    "logic", "math", "factual",
)

_CODE_SIGNAL = re.compile(
    r"```|\bdef\s+\w+|\breturn\b|console\.log|\bfunction\s*\(|=>|\bclass\s+\w+"
)
_BUG_SIGNAL = re.compile(
    r"\bbug(gy|s)?\b|\bdebug\b|\bfix\b|error in|fails?\b|doesn'?t work|"
    r"incorrect(ly)?|wrong (output|result|answer)|broken",
    re.I,
)
_GEN_SIGNAL = re.compile(
    r"write (a|an|the)?\s*[\w,\- ]*(function|method|program|script|class|code)|"
    r"implement (a|an|the)|create (a|an)?\s*[\w\- ]*function|"
    r"code (a|an|the)? |write [\w ]*python",
    re.I,
)
_LOGIC_SIGNAL = re.compile(
    r"(who|which (one|person|friend|of them))\s+(owns?|has|have|likes?|drinks?|"
    r"drives?|lives?|plays?|works?|wears?|won|finish|sits?|is\b)|"
    r"each\s+(?:\w+\s+){0,4}different|exactly one (of|is|contains)|"
    r"no two\b|neither\b.*\bnor\b|logic puzzle|"
    r"(finishes|arrived|sits) (before|after|between|next to)",
    re.I,
)
_MATH_SIGNAL = re.compile(
    r"how (many|much)|percent|%|\btotal\b|remain(s|ing)?\b|left over|"
    r"sum of|average|mean\b|calculate|compute|difference between|"
    r"cost|price|profit|discount|interest|projection|per (day|week|month|year|hour)|"
    r"km/h|mph|speed|area|perimeter|probability|ratio",
    re.I,
)


def classify(prompt: str) -> str:
    p = prompt.lower()

    if re.search(r"named entit|entit(y|ies)\b.*(type|label)|extract.*entit", p):
        return "ner"
    if "sentiment" in p:
        return "sentiment"
    if re.search(r"summari[sz]e|\bsummary\b|condense|tl;?dr|shorten (this|the)", p):
        return "summarize"

    has_code = bool(_CODE_SIGNAL.search(prompt))
    if has_code and _BUG_SIGNAL.search(p):
        return "code_debug"
    if _GEN_SIGNAL.search(p) and not (has_code and _BUG_SIGNAL.search(p)):
        return "code_gen"

    has_digit = bool(re.search(r"\d", p))
    logic_hit = bool(_LOGIC_SIGNAL.search(p))
    math_hit = bool(_MATH_SIGNAL.search(p)) and has_digit

    if logic_hit and math_hit:
        # arithmetic verbs + money/percent lean math; pure deduction leans logic
        if re.search(r"%|percent|\$|\bcost|price|total of|sum of|average", p):
            return "math"
        return "logic"
    if logic_hit:
        return "logic"
    if math_hit:
        return "math"
    return "factual"
