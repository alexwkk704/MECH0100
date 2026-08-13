"""
refresh_charts.py
=================
Re-render all master charts from the current data + chart_config sheet.
Useful for iterating on chart titles / labels without running a simulation.
Called by REFRESH_CHARTS.bat.

Material properties are read from the 'settings' sheet of runs_input.xlsx
(the same source the simulations use) so the charts always match the dataset.
Nothing about the material is hardcoded here.
"""
from pathlib import Path

import chart_builder
from chart_builder import rebuild_charts

HERE = Path(__file__).resolve().parent
MASTER = HERE / "Results" / "Results_summary.xlsx"


def solid_modulus_gpa():
    """es_gpa from runs_input.xlsx; fall back to chart_builder's default."""
    try:
        import ntop_batch as nb
        _rows, settings = nb.load_input_xlsx()
        return float(settings["es_gpa"])
    except Exception as e:
        print(f"[i] could not read es_gpa from runs_input.xlsx ({e}) — "
              f"using default {chart_builder.ES_GPA} GPa")
        return chart_builder.ES_GPA


if __name__ == "__main__":
    chart_builder.ES_GPA = solid_modulus_gpa()
    print(f"Refreshing charts from {MASTER}")
    print(f"Solid modulus Es = {chart_builder.ES_GPA} GPa")
    rebuild_charts(MASTER)
    print("Done. Open the file to see updated charts.")
    input("Press Enter to exit ...")
