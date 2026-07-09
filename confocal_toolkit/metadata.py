"""Parse animal/slide metadata from messy slide file names.

Fields: degeneration model, Samd7 genotype (KO/WT), eye (left/right), stain,
age (p##), and an optional animal/replicate id.

Strategy: a rule-based parser always works offline. If an ANTHROPIC_API_KEY is
set *and* the `anthropic` package is installed, an AI pass is tried first (better
on unusual names) and falls back to the rules on any failure. The result is then
shown in an interactive confirm/edit prompt (unless suppressed).

Eye convention (per the lab): S-opsin is imaged in the LEFT eye, M-opsin in the
RIGHT eye.

Default age when the name has no p## token, keyed on degeneration model:
    P23H -> p30,  RD10 -> p60,  RhoKO/Rho-/- -> p90.
"""

from __future__ import annotations

import os
import re

FIELDS = ["model", "samd7", "eye", "stain", "age", "animal_id"]

# keys are upper-cased model names (see default_age_for_model)
MODEL_DEFAULT_AGE = {"P23H": "p30", "RD10": "p60", "RHOKO": "p90"}


def default_age_for_model(model: str) -> str:
    return MODEL_DEFAULT_AGE.get((model or "").upper().replace("-/-", "KO"), "")


def _detect_model(low: str) -> str:
    if "rd10" in low:
        return "RD10"
    if "p23h" in low or "p23" in low:
        return "P23H"
    if "rho" in low or "rho-/-" in low:
        return "RhoKO"
    return ""


def _detect_samd7(low: str) -> str:
    m = re.search(r"(?:samd7|sd7)\s*[-_ ]?\s*(ko|wt)", low)
    return m.group(1).upper() if m else ""


def _detect_stain(low: str) -> str:
    if re.search(r"\bs[-_ ]?647\b|s[-_ ]?opsin|\bsopsin\b", low):
        return "S-opsin"
    if re.search(r"\bm[-_ ]?647\b|m[-_ ]?opsin|\bmopsin\b", low):
        return "M-opsin"
    return ""


def eye_for_stain(stain: str) -> str:
    s = (stain or "").lower()
    if s.startswith("s"):
        return "left"
    if s.startswith("m"):
        return "right"
    return ""


def _detect_age(low: str) -> str:
    # p## but not the "p23" inside model name "p23h", and not letters like "HP_05"
    m = re.search(r"(?<![a-z])p(\d{1,3})(?![\dh])", low)
    return f"p{m.group(1)}" if m else ""


def _detect_animal(name: str) -> str:
    m = re.match(r"\s*(R\d+)\b", name)
    return m.group(1) if m else ""


def parse_rules(name: str) -> dict:
    """Rule-based metadata from a file stem."""
    low = name.lower()
    model = _detect_model(low)
    stain = _detect_stain(low)
    meta = {
        "model": model,
        "samd7": _detect_samd7(low),
        "stain": stain,
        "eye": eye_for_stain(stain),
        "age": _detect_age(low) or default_age_for_model(model),
        "animal_id": _detect_animal(name),
        "slide": name,
    }
    return meta


def parse_ai(name: str) -> dict | None:
    """Try to parse with the Anthropic API. Returns None if unavailable/failed."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        import json
        client = anthropic.Anthropic()
        prompt = (
            "Extract retina-imaging metadata from this microscopy slide filename. "
            "Return ONLY compact JSON with keys: model (one of RD10, P23H, RhoKO, "
            "or \"\"), samd7 (KO, WT, or \"\"), stain (S-opsin, M-opsin, or \"\"), "
            "eye (left, right, or \"\"), age (like p30 or \"\"), animal_id (or \"\"). "
            "Eye convention: S-opsin=left, M-opsin=right. "
            f"Filename: {name!r}"
        )
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        text = text[text.find("{"): text.rfind("}") + 1]
        data = json.loads(text)
        data.setdefault("slide", name)
        if not data.get("eye"):
            data["eye"] = eye_for_stain(data.get("stain", ""))
        if not data.get("age"):
            data["age"] = default_age_for_model(data.get("model", ""))
        return {k: str(data.get(k, "")) for k in FIELDS} | {"slide": name}
    except Exception as e:
        print(f"  (AI metadata parse failed, using rules: {e})")
        return None


def parse_metadata(name: str, use_ai: bool = True) -> dict:
    """AI-assisted if available, else rule-based."""
    if use_ai:
        ai = parse_ai(name)
        if ai:
            return ai
    return parse_rules(name)


def standardized_name(meta: dict) -> str:
    """e.g. RD10_Samd7-KO_L_S-opsin_p60_R1 (empty fields omitted)."""
    eye = (meta.get("eye") or "").lower()
    eye_letter = "L" if eye.startswith("l") else "R" if eye.startswith("r") else ""
    parts = [
        meta.get("model", ""),
        f"Samd7-{meta['samd7']}" if meta.get("samd7") else "",
        eye_letter,
        meta.get("stain", ""),
        meta.get("age", ""),
        meta.get("animal_id", ""),
    ]
    slug = "_".join(p for p in parts if p)
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", slug) or "sample"


def confirm_metadata(meta: dict, interactive: bool = True) -> dict:
    """Show parsed metadata; let the user press Enter to accept or edit each field."""
    meta = dict(meta)
    print("\n  Metadata for:", meta.get("slide", ""))
    if not interactive:
        for k in FIELDS:
            print(f"    {k:10s}: {meta.get(k, '')}")
        print(f"    -> {standardized_name(meta)}")
        return meta
    print("  Press Enter to keep the value in [brackets], or type a new one.")
    for k in FIELDS:
        cur = meta.get(k, "")
        try:
            val = input(f"    {k:10s} [{cur}]: ").strip()
        except EOFError:
            val = ""
        if val:
            meta[k] = val
    # keep eye consistent with stain if the user changed stain but not eye
    if not meta.get("eye"):
        meta["eye"] = eye_for_stain(meta.get("stain", ""))
    print(f"  -> standardized name: {standardized_name(meta)}\n")
    return meta
