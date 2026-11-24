"""Structure diagrams of the five models, used by report/report.ipynb. They draw boxes only and use no data."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HMM_BOX = ("3-state HMM on the six day-ahead inputs\nforward filter gives\n"
           "2 regime probabilities per half-hour", "#F1E8FB", "#7A4DB5")
BASE_INPUTS = ["Day-ahead price  $P_{DA}$", "Day-ahead forecasts\nload · wind onshore · solar",
               "Scheduled trade\nimports · exports", "Time of day\nhour_sin · hour_cos",
               "Calendar\nday of week (6) · season (3)"]


def _box(ax, x, y, w, h, text, face, edge, size=10, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=face, edgecolor=edge, linewidth=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                 color="#555555", linewidth=1.2))


def _target_boxes(ax, q90):
    q_short, q_long = q90
    _box(ax, 9.4, 4.35, 3.5, 1.15, r"$P(S_{short} > q_{90,short})$" + "\nchance short stress is severe\n"
         + r"$S_{short}=P_{short}-P_{DA}$,  " + f"$q_{{90}}={q_short:.2f}$ €/MWh", "#E9F6EC", "#2E8B57", size=9)
    _box(ax, 9.4, 2.5, 3.5, 1.15, r"$P(S_{long} > q_{90,long})$" + "\nchance long stress is severe\n"
         + r"$S_{long}=P_{DA}-P_{long}$,  " + f"$q_{{90}}={q_long:.2f}$ €/MWh", "#E9F6EC", "#2E8B57", size=9)


def _time_split(ax, split, tune):
    split = split or [("2021-22  train: fit the model", 1.5, 4.2, "#2F6DB5"),
                      (f"2023  validation: {tune}", 5.75, 2.4, "#C27C0E"),
                      ("2024  test: final check", 8.2, 2.4, "#2E8B57"),
                      ("2025  holdout (later)", 10.65, 2.2, "#9A9A9A")]
    ax.text(0.2, 0.62, "Time split:", fontsize=10, weight="bold", va="center")
    for label, x, w, color in split:
        ax.add_patch(FancyBboxPatch((x, 0.35), w - 0.05, 0.55, boxstyle="round,pad=0.01,rounding_size=0.08",
                                    facecolor=color, edgecolor="none", alpha=0.85))
        ax.text(x + (w - 0.05) / 2, 0.62, label, ha="center", va="center", fontsize=9, color="white",
                weight="bold")


def model_diagram(title, model_title, formula, note, q90, side_box=None, tune="tune the settings",
                  inputs=None, split=None):
    """Inputs on the left, the model in the middle, the two targets on the right, the time split below.

    `q90` is (q90_short, q90_long) from the pipeline. `side_box` adds a step above the model (the HMM).
    """
    inputs = inputs or BASE_INPUTS
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    ax.text(1.9, 7.15, "Inputs: known after the day-ahead auction,\nbefore delivery (z-scored on the fit years)",
            ha="center", va="center", fontsize=10, weight="bold")
    height = 0.8 if len(inputs) <= 5 else 0.7
    ys = list(np.linspace(5.85, 1.85, len(inputs)))
    for y, text in zip(ys, inputs):
        _box(ax, 0.2, y, 3.4, height, text, "#E8F1FB", "#2F6DB5", size=9)

    mx, my, mw, mh = 4.9, 2.6, 3.4, 2.8
    _box(ax, mx, my, mw, mh, "", "#FFF6E5", "#C27C0E")
    ax.text(mx + mw / 2, my + mh - 0.4, model_title, ha="center", fontsize=12, weight="bold")
    ax.text(mx + mw / 2, my + mh / 2 + 0.1, formula, ha="center", va="center", fontsize=12)
    ax.text(mx + mw / 2, my + 0.55, note, ha="center", va="center", fontsize=9, style="italic")
    for y in ys:
        _arrow(ax, 3.6, y + height / 2, mx, my + mh / 2)
    if side_box:
        text, face, edge = side_box
        _box(ax, mx, 5.75, mw, 1.2, text, face, edge, size=9)
        for y in ys[:3]:
            _arrow(ax, 3.6, y + height / 2, mx, 6.35)
        _arrow(ax, mx + mw / 2, 5.75, mx + mw / 2, my + mh)

    _target_boxes(ax, q90)
    _arrow(ax, mx + mw, my + mh / 2, 9.4, 4.9)
    _arrow(ax, mx + mw, my + mh / 2, 9.4, 3.1)
    ax.text(11.15, 6.05, "Output: a probability from 0 to 1\nthat S lands in its top 10%\n(q90 fixed from 2021-22)",
            ha="center", va="center", fontsize=9, color="#333333")
    _box(ax, 4.6, 0.95, 4.0, 1.05, "Not used (only known after delivery):\nactual load · actual generation ·\n"
         "physical flows · imbalance prices", "#FBEAEA", "#B53A3A", size=8.5)
    _time_split(ax, split, tune)
    ax.set_title(title, fontsize=14, weight="bold", pad=10)
    plt.tight_layout()
    plt.show()
