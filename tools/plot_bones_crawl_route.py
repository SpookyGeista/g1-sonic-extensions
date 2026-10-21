#!/usr/bin/env python3
"""Plot an unchanged SONIC reference as a numbered XY route and height timeline.

No simulator, GPU, policy or generative image model is used. Plots show the
reference pelvis trajectory, not measured tracking or collision clearance.
Only NEW image/provenance files are written; motion data is never modified.

Example (this interpreter already has matplotlib):
  /home/loya2/miniconda3/envs/unitree_sim_env/bin/python tools/plot_bones_crawl_route.py \
    --motion data/bones_seed_crawl_extension_2026-09-16/01_complete_chain \
    --output data/bones_seed_crawl_extension_2026-09-16/laufweg_draufsicht.png

Metadata: build_metadata.json.timeline, each item with start_frame,
end_frame_exclusive and label/id/stage_id. Optional stages/sequence/source_stages
supply labels by id. FPS comes from metadata.fps or metadata.validation.fps;
otherwise pass --fps. Legacy source_stages + explicit overlap seams are supported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import textwrap

import numpy as np

PALETTE = ["#2378a8", "#b86029", "#6a62a5", "#327e86", "#a94851", "#857523",
           "#486ba0", "#a868a2", "#597c39", "#c88723", "#258472", "#854d36",
           "#507393", "#b34f84", "#486744", "#78728c", "#bf734f"]
PAPER, PLOT, INK = "#f3f2ed", "#e4e5dd", "#24343b"
KNOWN_LABELS = {"01_walk": "Laufen", "02_cover_left": "Links in Deckung / aufrichten",
                "03_curve_left": "Linkskurve", "04_walk": "Weiterlaufen",
                "05_cover_right": "Rechts in Deckung / aufrichten",
                "06_curve_right": "Rechtskurve", "07_walk_finish": "Weiterlaufen / abbremsen",
                "08_lie_down": "Hinlegen / Bauchlage", "09_crawl_start": "Krabbeln starten",
                "10_crawl": "Auf dem Boden krabbeln", "11_crawl_stop": "Krabbeln beenden",
                "12_stand_up": "Wieder aufstehen"}
CRAWL_WORDS = re.compile(r"crawl|krabb|prone|hinleg|aufsteh|boden|floor|lie_down|lying|stand_up|get_up|getup", re.I)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_fps(metadata, explicit=None):
    evidence = []
    if explicit is not None:
        evidence.append(("--fps", float(explicit)))
    if "fps" in metadata:
        evidence.append(("metadata.fps", float(metadata["fps"])))
    if "fps" in metadata.get("validation", {}):
        evidence.append(("metadata.validation.fps", float(metadata["validation"]["fps"])))
    if not evidence or any(not math.isfinite(fps) or fps <= 0 for _, fps in evidence):
        raise ValueError("FPS unknown/invalid; provide metadata.fps or explicit --fps")
    if any(not math.isclose(fps, evidence[0][1], rel_tol=0, abs_tol=1e-8) for _, fps in evidence):
        raise ValueError(f"Conflicting FPS evidence: {evidence}")
    return evidence[0][1], [name for name, _ in evidence]


def load_timeline(metadata, frames):
    labels = {}
    for key in ("stages", "sequence", "source_stages"):
        for stage in metadata.get(key, []):
            if isinstance(stage, dict):
                labels[str(stage.get("id", stage.get("stage_id", "")))] = stage
    timeline = metadata.get("timeline")
    basis = "build_metadata.json.timeline"
    if not timeline:
        stages, seams = metadata.get("source_stages", []), metadata.get("seams", [])
        if stages and len(seams) == len(stages)-1 and all(
            seam.get("next_stage_id") == stages[i+1].get("id")
            and "start_frame" in seam and "end_frame_exclusive" in seam
            for i, seam in enumerate(seams)
        ):
            timeline = [{"id": stage["id"],
                         "start_frame": 0 if i == 0 else seams[i-1]["start_frame"],
                         "end_frame_exclusive": frames if i == len(stages)-1 else seams[i]["end_frame_exclusive"]}
                        for i, stage in enumerate(stages)]
            basis = "Legacy source_stages + explicit seams (overlap belongs to both adjacent stages)"
        else:
            timeline = [{"id": "reference", "label": "Gesamtreferenz (keine Phasengrenzen verfügbar)",
                         "start_frame": 0, "end_frame_exclusive": frames}]
            basis = "Single-trajectory fallback; no phase boundaries inferred"
    stages, previous_start = [], -1
    for index, item in enumerate(timeline):
        start, stop = item["start_frame"], item["end_frame_exclusive"]
        if (type(start) is not int or type(stop) is not int or not 0 <= start < stop <= frames
                or start < previous_start):
            raise ValueError(f"Invalid/unsorted timeline frame range: {item}")
        stage_id = str(item.get("id", item.get("stage_id", f"stage_{index+1:02d}")))
        extra = labels.get(stage_id, {})
        label = str(item.get("label", extra.get("label", KNOWN_LABELS.get(stage_id, stage_id))))
        stages.append({**extra, **item, "id": stage_id, "label": label,
                       "number": index+1, "color": PALETTE[index % len(PALETTE)]})
        previous_start = start
    if len({s["id"] for s in stages}) != len(stages):
        raise ValueError("Timeline stage IDs must be unique")
    return stages, basis


def load_motion(folder, metadata_path=None, fps=None):
    folder = Path(folder).resolve()
    path = Path(metadata_path).resolve() if metadata_path else folder/"build_metadata.json"
    metadata = json.loads(path.read_text()) if path.exists() else {}
    positions = np.loadtxt(folder/"body_pos.csv", delimiter=",", skiprows=1, ndmin=2)
    quaternions = np.loadtxt(folder/"body_quat.csv", delimiter=",", skiprows=1, ndmin=2)
    if positions.shape[1] != 42 or quaternions.shape != (len(positions), 56) or len(positions) < 2:
        raise ValueError("Expected 14-body SONIC reference: matching [T,42]/[T,56] channels")
    if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
        raise ValueError("Reference contains NaN/Inf")
    root, quat = positions[:, :3], quaternions[:, :4]
    norms = np.linalg.norm(quat, axis=1)
    if np.max(np.abs(norms-1)) > 1e-3:
        raise ValueError("Invalid pelvis quaternion norms")
    quat = quat/norms[:, None]
    w, x, y, z = quat.T
    yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    actual_fps, fps_basis = resolve_fps(metadata, fps)
    stages, timeline_basis = load_timeline(metadata, len(root))
    paths = [folder/"body_pos.csv", folder/"body_quat.csv"] + ([path] if path.exists() else [])
    return {"folder": folder, "metadata": metadata, "root": root, "yaw": yaw,
            "fps": actual_fps, "fps_basis": fps_basis, "stages": stages,
            "timeline_basis": timeline_basis,
            "source_sha256": {str(source): digest(source) for source in paths}}


def arc_midpoint_index(xy):
    distance = np.r_[0.0, np.linalg.norm(np.diff(xy, axis=0), axis=1).cumsum()]
    return int(np.argmin(abs(distance-distance[-1]/2))) if distance[-1] > .01 else len(xy)//2


def bounds_for(xy, padding=.6):
    low, high = xy.min(axis=0), xy.max(axis=0)
    span = high-low
    margin = np.maximum(padding, .08*span)
    return low-margin, high+margin


def style_axis(ax, title=None):
    ax.set_facecolor(PLOT)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Referenz X [m]", color=INK)
    ax.set_ylabel("Referenz Y [m]", color=INK)
    ax.grid(alpha=.24, color="#a2aba5", linewidth=.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK, pad=10)
    for spine in ax.spines.values():
        spine.set_color("#b2bbb3")


def place_numbers(ax, xy, stages):
    """Greedy label placement in display space; never displace trajectory data."""
    ax.figure.canvas.draw()
    # Reserve the START/ENDE caption areas so the final phase number is not
    # subsequently covered by an endpoint caption.
    pixels_per_point = ax.figure.dpi/72
    occupied = [ax.transData.transform(xy[0])+np.array([23, -23])*pixels_per_point,
                ax.transData.transform(xy[-1])+np.array([23, 12])*pixels_per_point]
    axes_box = ax.get_window_extent()
    for stage in stages:
        segment = xy[stage["start_frame"]:stage["end_frame_exclusive"]]
        anchor = segment[arc_midpoint_index(segment)]
        pixel = ax.transData.transform(anchor)
        candidates = []
        for radius in (23, 39, 57, 76, 98):
            for angle in np.arange(8)*math.pi/4 + math.pi/4:
                offset = radius*np.array([math.cos(angle), math.sin(angle)])
                candidate = pixel + offset*ax.figure.dpi/72
                border = min(candidate[0]-axes_box.x0, axes_box.x1-candidate[0],
                             candidate[1]-axes_box.y0, axes_box.y1-candidate[1])
                overlaps = sum(np.linalg.norm(candidate-other) < 30*pixels_per_point for other in occupied)
                candidates.append((overlaps*1000 + (10000 if border < 16*pixels_per_point else 0) + radius,
                                   candidate, offset))
        _, candidate, offset = min(candidates, key=lambda value: value[0])
        occupied.append(candidate)
        ax.annotate(f'{stage["number"]:02d}', anchor, xytext=offset, textcoords="offset points",
                    ha="center", va="center", fontsize=9, fontweight="bold", color=stage["color"],
                    bbox={"boxstyle": "circle,pad=.35", "facecolor": PAPER,
                          "edgecolor": stage["color"], "linewidth": 1.1},
                    arrowprops={"arrowstyle": "-", "color": stage["color"], "linewidth": .8},
                    zorder=8)


def draw_segments(ax, xy, stages, *, faded_background=False):
    ax.plot(*xy.T, color="#8c9691", linewidth=1.3, alpha=.4 if faded_background else .7,
            linestyle="--", zorder=1)
    for stage in stages:
        segment = xy[stage["start_frame"]:stage["end_frame_exclusive"]]
        ax.plot(*segment.T, color=stage["color"], linewidth=2.7, alpha=.86, zorder=3)
        if len(segment) > 3:
            mid = arc_midpoint_index(segment)
            before, after = max(0, mid-8), min(len(segment)-1, mid+8)
            if np.linalg.norm(segment[after]-segment[before]) > .025:
                ax.annotate("", segment[after], xytext=segment[before],
                            arrowprops={"arrowstyle": "-|>", "color": stage["color"],
                                        "mutation_scale": 15, "linewidth": 1.6}, zorder=5)


def endpoint(ax, point, yaw, label, color, marker, scale, text_offset):
    ax.scatter(*point, s=75, marker=marker, color=color, edgecolor=PAPER, linewidth=1, zorder=9)
    direction = scale*np.array([np.cos(yaw), np.sin(yaw)])
    ax.annotate("", point+direction, xytext=point,
                arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.6}, zorder=8)
    ax.annotate(label, point, xytext=text_offset, textcoords="offset points", fontsize=9,
                fontweight="bold", color=color, zorder=10,
                bbox={"facecolor": PAPER, "edgecolor": "none", "alpha": .92, "pad": 2})


def plot_motion(data, output, title="BONES / G1 — Laufweg mit Bodenkrabbeln", detail=True, dpi=180):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    output = Path(output).resolve()
    report_path = output.with_suffix(".json")
    if output.suffix.lower() not in (".png", ".pdf", ".svg"):
        raise ValueError("Output must be .png, .pdf or .svg")
    if output.exists() or report_path.exists():
        raise FileExistsError("Image/provenance exists; choose a new --output to preserve earlier results")
    root, fps, stages = data["root"], data["fps"], data["stages"]
    xy, time = root[:, :2], np.arange(len(root))/fps
    crawling = [s for s in stages if CRAWL_WORDS.search(" ".join(str(s.get(k, "")) for k in ("id", "label", "kind")))]
    use_detail = detail and bool(crawling)
    fig = plt.figure(figsize=(15.8, 10.7), facecolor=PAPER)
    grid = fig.add_gridspec(3, 2, width_ratios=(1.6, 1), height_ratios=(4.5, 3.0, 1.6),
                           left=.07, right=.97, top=.86, bottom=.105, hspace=.45, wspace=.28)
    route_ax = fig.add_subplot(grid[:2, 0])
    legend_ax = fig.add_subplot(grid[0, 1] if use_detail else grid[:2, 1])
    height_ax = fig.add_subplot(grid[2, :])
    fig.suptitle(title, fontsize=20, fontweight="bold", color=INK, y=.955)
    distance = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    extent = xy.max(axis=0)-xy.min(axis=0)
    fig.text(.07, .910, f"{len(stages)} Phasen  ·  {time[-1]:.2f} s  ·  {distance:.2f} m Beckenweg  ·  "
             f"XY-Ausdehnung {extent[0]:.2f} × {extent[1]:.2f} m", fontsize=11, color=INK)
    fig.text(.07, .885, "Referenzbahn aus den Motion-Dateien — kein gemessener Simulationslauf, keine Kollisionsprüfung.",
             fontsize=10, color="#5e6968")
    style_axis(route_ax, "Gesamtablauf / Draufsicht")
    low, high = bounds_for(xy)
    route_ax.set_xlim(low[0], high[0]); route_ax.set_ylim(low[1], high[1])
    draw_segments(route_ax, xy, stages)
    place_numbers(route_ax, xy, stages)
    arrow_scale = max(.22, min(.55, float(np.max(extent))*.055))
    endpoint(route_ax, xy[0], data["yaw"][0], "START", "#2a7851", "o", arrow_scale, (8, -23))
    endpoint(route_ax, xy[-1], data["yaw"][-1], "ENDE", "#933f40", "s", arrow_scale, (8, 12))

    legend_ax.set_axis_off()
    legend_ax.set_title("Reihenfolge und Zeitfenster", fontsize=12, color=INK, fontweight="bold", loc="left", pad=12)
    # Wrapped labels get their actual line space; never shrink the plot to hide phases.
    entries = [(s, textwrap.wrap(s["label"], width=49) or [s["label"]]) for s in stages]
    total_lines = sum(len(lines)+.65 for _, lines in entries)
    row = .975
    line_h = min(.070 if use_detail else .052, .95/max(total_lines, 1))
    for stage, lines in entries:
        label = "\n".join(lines)
        legend_ax.text(0, row, f'{stage["number"]:02d}', transform=legend_ax.transAxes,
                       fontsize=10, color=stage["color"], weight="bold", va="top")
        legend_ax.text(.085, row, label, transform=legend_ax.transAxes,
                       fontsize=9, color=INK, va="top", linespacing=1.1)
        row -= line_h*len(lines)
        legend_ax.text(.085, row, f'{stage["start_frame"]/fps:.2f}–{(stage["end_frame_exclusive"]-1)/fps:.2f} s',
                       transform=legend_ax.transAxes, fontsize=8, color="#62716f", va="top")
        row -= line_h*.65

    if use_detail:
        crawl_ax = fig.add_subplot(grid[1, 1])
        style_axis(crawl_ax, "Detail: Hinlegen → Krabbeln → Aufstehen")
        chosen = np.concatenate([np.arange(s["start_frame"], s["end_frame_exclusive"]) for s in crawling])
        crawl_low, crawl_high = bounds_for(xy[chosen], padding=.22)
        crawl_ax.set_xlim(crawl_low[0], crawl_high[0]); crawl_ax.set_ylim(crawl_low[1], crawl_high[1])
        route_ax.add_patch(Rectangle(crawl_low, *(crawl_high-crawl_low), fill=False,
                                    edgecolor="#7f8882", linestyle=":", linewidth=1.2, zorder=0))
        draw_segments(crawl_ax, xy, crawling, faded_background=True)
        place_numbers(crawl_ax, xy, crawling)
        crawl_ax.tick_params(labelsize=8)

    height_ax.set_facecolor(PLOT)
    height_ax.plot(time, root[:, 2], color="#8c9691", linewidth=1.2, linestyle="--")
    for stage in stages:
        start, end = stage["start_frame"], stage["end_frame_exclusive"]
        height_ax.plot(time[start:end], root[start:end, 2], color=stage["color"], linewidth=2)
    for seam in data["metadata"].get("seams", []):
        start = seam.get("start_frame", seam.get("bridge_start_frame"))
        end = seam.get("end_frame_exclusive", seam.get("bridge_end_frame_exclusive"))
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(root):
            height_ax.axvspan(start/fps, min(time[-1], end/fps), color="#7f8882", alpha=.12, hatch="///", linewidth=0)
    height_ax.set_xlim(0, time[-1])
    height_ax.set_ylim(max(0, float(root[:, 2].min())-.10), float(root[:, 2].max())+.10)
    height_ax.set_title("Beckenhöhe über Zeit" + (" — niedrige Bodenphase wird sichtbar" if crawling else ""), loc="left", fontsize=10,
                        color=INK, fontweight="bold", pad=8)
    height_ax.set_ylabel("Höhe Z [m]"); height_ax.set_xlabel("Zeit [s]")
    height_ax.grid(alpha=.22); height_ax.set_axisbelow(True)
    height_ax.tick_params(labelsize=9)
    fig.text(.07, .045, "Pfeile auf der Linie: Bewegungsrichtung. Start-/Endpfeile: Körperausrichtung. "
             "Graue Schraffur: Überblendung, sofern dokumentiert.", fontsize=8.8, color="#5e6968")
    fig.text(.07, .024, "Kein räumliches Glätten, Versetzen oder Entkreuzen der Bahn. Überlagerungen und Rückwege bleiben sichtbar; "
             "die Beckenlinie ist kein Körperumriss.", fontsize=8.8, color="#5e6968")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive file creation avoids overwriting even if another renderer races us.
    with output.open("xb") as handle:
        fig.savefig(handle, format=output.suffix[1:], dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    if any(digest(path) != expected for path, expected in data["source_sha256"].items()):
        raise RuntimeError("Source files changed during plotting; image must not be treated as current")
    report = {"image": str(output), "image_sha256": digest(output),
              "motion": str(data["folder"]), "kind": "reference_pelvis_trajectory_not_measured_tracking",
              "walls_drawn": False, "collision_validation": False, "spatial_adjustments": "none",
              "fps": fps, "fps_basis": data["fps_basis"], "frames": len(root),
              "duration_first_to_last_s": float(time[-1]), "pelvis_xy_path_m": distance,
              "xy_extent_m": extent.tolist(), "height_range_m": [float(root[:, 2].min()), float(root[:, 2].max())],
              "timeline_basis": data["timeline_basis"], "phases": stages,
              "crawl_detail_shown": use_detail, "source_sha256": data["source_sha256"]}
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--motion", type=Path, required=True, help="SONIC reference folder (read only)")
    parser.add_argument("--output", type=Path, required=True, help="NEW .png/.pdf/.svg plus .json provenance")
    parser.add_argument("--metadata", type=Path, help="Optional alternate build_metadata.json")
    parser.add_argument("--fps", type=float, help="Explicit FPS only if metadata is absent; conflicts rejected")
    parser.add_argument("--title", default="BONES / G1 — Laufweg mit Bodenkrabbeln")
    parser.add_argument("--no-detail", action="store_true", help="Omit automatically selected crawl close-up")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if not 72 <= args.dpi <= 600:
        parser.error("--dpi must be between 72 and 600")
    try:
        data = load_motion(args.motion, args.metadata, args.fps)
        result = plot_motion(data, args.output, args.title, not args.no_detail, args.dpi)
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.error(str(error))
    print(json.dumps({key: result[key] for key in ("image", "frames", "duration_first_to_last_s", "pelvis_xy_path_m", "crawl_detail_shown")}, indent=2))


if __name__ == "__main__":
    main()
