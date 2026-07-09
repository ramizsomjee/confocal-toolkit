"""Headless tests for metadata parsing and editable PDF figure generation."""
import os
import tempfile

import numpy as np

from confocal_toolkit import metadata as MD, figure as FIG


def test_metadata_rules():
    m = MD.parse_metadata("R1 20x RD10 SD7 KO S-647 2026_06_30_HP_05-EDF VAR")
    assert m["model"] == "RD10" and m["samd7"] == "KO"
    assert m["stain"] == "S-opsin" and m["eye"] == "left"
    assert m["age"] == "p60"          # RD10 default
    assert m["animal_id"] == "R1"

    m2 = MD.parse_metadata("20x Rho sd7 wt M-647 2026_07_02_05_HP-EDF VAR")
    assert m2["model"] == "RhoKO" and m2["samd7"] == "WT"
    assert m2["stain"] == "M-opsin" and m2["eye"] == "right"
    assert m2["age"] == "p90"         # RhoKO default

    m3 = MD.parse_metadata("SD7 ko m-opsin Corbo 02")
    assert m3["samd7"] == "KO" and m3["stain"] == "M-opsin" and m3["eye"] == "right"
    print("[ok] metadata rules (model/samd7/stain/eye + model-default age)")


def test_age_edge_cases():
    assert MD.default_age_for_model("P23H") == "p30"
    m = MD.parse_metadata("P23H sd7 wt s-opsin")
    # 'p23h' model must NOT be misread as age p23
    assert m["model"] == "P23H" and m["age"] == "p30" and m["age"] != "p23"
    # explicit p## in the name wins over the default
    m2 = MD.parse_metadata("RD10 sd7 ko m-opsin p45")
    assert m2["age"] == "p45"
    print("[ok] age: model default, p23h not misparsed, explicit p## wins")


def test_standardized_name():
    m = {"model": "RD10", "samd7": "KO", "eye": "left", "stain": "S-opsin",
         "age": "p60", "animal_id": "R1"}
    assert MD.standardized_name(m) == "RD10_Samd7-KO_L_S-opsin_p60_R1"
    print("[ok] standardized name")


def test_figure_pdf_editable_and_mirrored():
    panels = {f: (np.random.default_rng(i).random((120, 120, 3)) * 255).astype(np.uint8)
              for i, f in enumerate(FIG.PANEL_ORDER)}
    overlay = (np.random.default_rng(9).random((300, 300, 3)) * 255).astype(np.uint8)
    with tempfile.TemporaryDirectory() as td:
        for eye in ("left", "right"):
            meta = {"model": "RD10", "samd7": "KO", "eye": eye,
                    "stain": "S-opsin" if eye == "left" else "M-opsin",
                    "age": "p60", "slide": "demo"}
            out = os.path.join(td, f"{eye}.pdf")
            FIG.build_eye_figure(panels, overlay, meta, out, box_um=100.0,
                                 overlay_um_per_px=2.0, std_name=f"{eye}_fig")
            assert os.path.exists(out) and os.path.getsize(out) > 2000
            with open(out, "rb") as fh:
                assert fh.read(4) == b"%PDF", "not a PDF"
        left_sz = os.path.getsize(os.path.join(td, "left.pdf"))
        right_sz = os.path.getsize(os.path.join(td, "right.pdf"))
        assert left_sz > 0 and right_sz > 0
    import matplotlib
    assert matplotlib.rcParams["pdf.fonttype"] == 42, "text not editable"
    print("[ok] editable PDFs (fonttype 42) for left & right eyes, %PDF header")


if __name__ == "__main__":
    test_metadata_rules()
    test_age_edge_cases()
    test_standardized_name()
    test_figure_pdf_editable_and_mirrored()
    print("\nFIGURE/METADATA OK")
