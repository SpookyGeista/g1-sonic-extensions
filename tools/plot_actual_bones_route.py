#!/usr/bin/env python3
"""Plot raw measured Isaac/PhysX pelvis XY alongside an unchanged BONES reference.

No controller or simulator is started. A completed, single-motion trial must
contain one contiguous playback block with exactly one logger row per reference
frame. PhysX samples are associated causally by host-monotonic timestamps, just
as ground_report.py does; controller/physics latency is not fitted away.
No spatial alignment, smoothing, resampling or yaw fit is performed.

Example:
  python tools/plot_actual_bones_route.py \
    --motion data/bones_seed_crawl_extension_2026-09-16/01_complete_chain \
    --trial data/isaac_sonic_bones_crawl_2026-09-16/trials/complete_chain_gui_01 \
    --output data/bones_seed_crawl_extension_2026-09-16/laufweg_gemessen.png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import textwrap

import numpy as np

from plot_bones_crawl_route import (
    CRAWL_WORDS, INK, PAPER, PLOT, arc_midpoint_index, bounds_for, digest,
    draw_segments, endpoint, load_motion, place_numbers, style_axis,
)

REPO = Path(__file__).resolve().parents[1]


def playback_rows(names_path, playing_path, motion_id, frames, fps):
    """Reject missing, repeated, restarted, reordered or time-inconsistent play."""
    with Path(names_path).open(newline="") as handle:
        names_rows = list(csv.DictReader(handle))
    names = {int(row["index"]): row["motion_name"] for row in names_rows}
    if len(names) != len(names_rows):
        raise ValueError("Duplicate controller motion-name indices")
    with Path(playing_path).open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if float(row["playing_0"]) > .5
                and names.get(int(row["index"])) == motion_id]
    indices = np.array([int(row["index"]) for row in rows], dtype=np.int64)
    stamps = np.array([float(row["time_monotonic_ms"])/1000 for row in rows])
    if (len(rows) != frames or not np.isfinite(stamps).all()
            or np.any(np.diff(indices) != 1) or np.any(np.diff(stamps) <= 0)):
        raise ValueError("Playback must have exactly one contiguous, monotonic row per reference frame")
    duration = (frames-1)/fps
    if abs(float(stamps[-1]-stamps[0])-duration) > .1 or np.max(np.diff(stamps)) > .06:
        raise ValueError("Playback duration/gaps do not agree with reference FPS")
    return indices, stamps


def map_trace(trace, columns, stamps):
    """Retain every measured sample between first and last playback timestamp."""
    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate trace columns")
    ix = {name: index for index, name in enumerate(columns)}
    needed = ["wall_monotonic", "support"] + [f"root_{i}" for i in range(7)]
    if trace.ndim != 2 or trace.shape[1] != len(columns) or any(key not in ix for key in needed):
        raise ValueError("Incomplete measured PhysX trace schema")
    wall = trace[:, ix["wall_monotonic"]]
    if (not np.isfinite(trace[:, [ix[name] for name in needed]]).all()
            or np.any(np.diff(wall) <= 0) or len(wall) < 2
            or wall[0] > stamps[0] or wall[-1] < stamps[-1]):
        raise ValueError("Trace must be finite, monotonic and span the whole playback")
    pick = np.flatnonzero((wall >= stamps[0]) & (wall <= stamps[-1]))
    if len(pick) < 2:
        raise ValueError("Insufficient measured playback samples")
    frames = np.searchsorted(stamps, wall[pick], side="right")-1
    root = trace[pick][:, [ix[f"root_{i}"] for i in range(3)]]
    quat = trace[pick][:, [ix[f"root_{i}"] for i in range(3, 7)]]
    if np.max(np.abs(np.linalg.norm(quat, axis=1)-1)) > 1e-3:
        raise ValueError("Invalid measured root quaternions")
    # Only start/end upright quaternion yaw is used for endpoint arrows. Never
    # unwrap/integrate prone Euler yaw: it is singular during ground transitions.
    end_quat = quat[[0, -1]] / np.linalg.norm(quat[[0, -1]], axis=1)[:, None]
    w, x, y, z = end_quat.T
    endpoint_yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    upright = (1-2*(x*x+y*y)) > np.cos(np.radians(45))
    return {"root": root, "time": wall[pick]-stamps[0], "trace_rows": pick,
            "reference_frames": frames, "endpoint_yaw": endpoint_yaw,
            "endpoints_upright": bool(np.all(upright)),
            "support_free": bool(np.all(trace[pick, ix["support"]] == 0)),
            "max_trace_gap_s": float(np.max(np.diff(wall[pick])))}


def map_stages(stages, mapped_frames, reference_frames):
    """Require the requested new timeline to cover the reference without gaps."""
    cursor, mapped = 0, []
    for stage in stages:
        start, stop = stage["start_frame"], stage["end_frame_exclusive"]
        if start != cursor:
            raise ValueError("Measured phase colors require a contiguous, nonoverlapping reference timeline")
        a, b = np.searchsorted(mapped_frames, [start, stop], side="left")
        if b <= a:
            raise ValueError(f"No measured samples for phase {stage['id']}")
        mapped.append({**stage, "reference_start_frame": start,
                       "reference_end_frame_exclusive": stop,
                       "start_frame": int(a), "end_frame_exclusive": int(b)})
        cursor = stop
    if cursor != reference_frames:
        raise ValueError("Timeline does not cover the full reference")
    return mapped


def load_trial(reference, trial_path):
    trial = Path(trial_path).resolve()
    paths = [trial/"trial.json", trial/"summary.json", trial/"sim/metadata.json",
             trial/"sim/trace.npz", trial/"controller/motion_name.csv",
             trial/"controller/motion_playing.csv"]
    hashes = {str(path): digest(path) for path in paths}
    manifest, summary, sim = [json.loads(path.read_text()) for path in paths[:3]]
    stages = manifest["stages"]
    if len(stages) != 1 or manifest.get("sim_only") is not True or sim.get("preview_only") is not False:
        raise ValueError("Expected one actual, simulation-only reference trial (not a kinematic preview)")
    stage = stages[0]
    source = Path(stage["source"])
    if not source.is_absolute():
        source = REPO/source
    if source.resolve() != reference["folder"]:
        raise ValueError("--motion is not the exact reference recorded by this trial")
    for path, expected in reference["source_sha256"].items():
        if manifest.get("source_hashes", {}).get(path) != expected:
            raise ValueError(f"Reference hash not bound to this trial: {path}")
    if (summary.get("all_requested_stages_completed") is not True
            or summary.get("source_or_runtime_hash_changes") != []
            or summary.get("trial_error") is not None
            or not any(event.get("name") == "trial_closed" for event in manifest.get("events", []))):
        raise ValueError("Require a completed immutable trial; a failed tracking threshold may still be plotted")
    # These runs spawn at (0,0), yaw 0. Do not invent a correspondence for a
    # translated/rotated scene, and do not fit reference coordinates to actual.
    scene = sim["scene"]
    if (not np.allclose(scene["robot_start_xy_m"], [0, 0], rtol=0, atol=1e-9)
            or abs(float(scene["robot_start_yaw_deg"])) > 1e-9):
        raise ValueError("Nonzero scene spawn: an explicit coordinate-frame definition is required; no automatic alignment")
    validation = stage["reference_validation"]
    if validation["frames"] != len(reference["root"]) or validation["reference_fps"] != reference["fps"]:
        raise ValueError("Trial/reference frame count or FPS mismatch")
    indices, stamps = playback_rows(paths[4], paths[5], stage["id"], len(reference["root"]), reference["fps"])
    with np.load(paths[3], allow_pickle=False) as data:
        measured = map_trace(data["data"], data["columns"].tolist(), stamps)
    measured["stages"] = map_stages(reference["stages"], measured["reference_frames"], len(reference["root"]))
    recorded = summary["actual_controller_playback"]["stages"][stage["id"]]
    if (recorded["playing_rows"] != len(stamps)
            or abs(recorded["first_play_monotonic"]-stamps[0]) > 1e-6
            or abs(recorded["last_play_monotonic"]-stamps[-1]) > 1e-6):
        raise ValueError("Summary does not match independently read controller playback")
    return {**measured, "trial": trial, "manifest": manifest, "summary": summary,
            "scene": scene, "source_sha256": {**reference["source_sha256"], **hashes},
            "play_indices": indices, "play_stamps": stamps}


def status_text(summary):
    ground = summary.get("ground_tracking", {})
    if summary.get("passed") is True:
        return "Vollständig abgespielt; protokollierte Simulations-Prüfungen bestanden", "#2a7851"
    if ground.get("ground_reference_tracking", {}).get("passed") is False:
        return "Vollständig abgespielt; Krabbel-Tracking außerhalb Prüfgrenze", "#9e5228"
    return "Vollständig abgespielt; mindestens eine protokollierte Prüfung nicht bestanden", "#9e5228"


def plot_actual(reference, measured, output, dpi=180):
    import matplotlib
    matplotlib.use("Agg")
    # Disable matplotlib path simplification: every recorded XY sample matters.
    matplotlib.rcParams["path.simplify"] = False
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    output = Path(output).resolve()
    report_path = output.with_suffix(".json")
    if output.suffix.lower() not in (".png", ".pdf", ".svg"):
        raise ValueError("Output must be .png, .pdf or .svg")
    if output.exists() or report_path.exists():
        raise FileExistsError("Image/provenance exists; choose a NEW --output")
    root, xy = measured["root"], measured["root"][:, :2]
    ref_xy, fps = reference["root"][:, :2], reference["fps"]
    stages = measured["stages"]
    crawling = [s for s in stages if CRAWL_WORDS.search(s["id"]+" "+s["label"])]
    distance = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    extent = np.ptp(xy, axis=0)
    status, status_color = status_text(measured["summary"])
    fig = plt.figure(figsize=(15.8, 11.0), facecolor=PAPER)
    grid = fig.add_gridspec(3, 2, width_ratios=(1.6, 1), height_ratios=(4.5, 3, 1.6),
                           left=.07, right=.97, top=.825, bottom=.12, hspace=.48, wspace=.28)
    route_ax = fig.add_subplot(grid[:2, 0])
    legend_ax = fig.add_subplot(grid[0, 1] if crawling else grid[:2, 1])
    height_ax = fig.add_subplot(grid[2, :])
    fig.suptitle("BONES / G1 — Gemessener Laufweg in Isaac Sim", fontsize=20,
                 fontweight="bold", color=INK, y=.963)
    fig.text(.07, .925, status, fontsize=12, color=status_color, fontweight="bold")
    fig.text(.07, .899, f"{len(stages)} Phasen · {(len(reference['root'])-1)/fps:.2f} s Referenz · "
             f"{distance:.2f} m gemessener Beckenweg · Ausdehnung {extent[0]:.2f} × {extent[1]:.2f} m",
             fontsize=11, color=INK)
    fig.text(.07, .876, "Farbig: unveränderte PhysX-Weltpositionen. Grau gestrichelt: ursprüngliche Referenz — keine räumliche Ausrichtung.",
             fontsize=9.6, color="#5e6968")
    ground = measured["summary"].get("ground_tracking", {})
    error = ground.get("ground_reference_tracking", {}).get("joint_rms_error_rad", {}).get("p95")
    threshold = ground.get("thresholds", {}).get("offline_p95_joint_rms_error_rad")
    if error is not None and threshold is not None:
        fig.text(.07, .854, f"Krabbelbereich: Gelenk-RMS-Fehler p95 {error:.3f} rad; Prüfgrenze {threshold:.3f} rad. "
                 "Keine Hindernisse / keine Kollisionsfreigabe.", fontsize=9.6, color=status_color)

    def draw_panel(ax, chosen_stages, title):
        style_axis(ax, title)
        ax.set_xlabel("Welt X [m]"); ax.set_ylabel("Welt Y [m]")
        ax.plot(*ref_xy.T, color="#727c79", linestyle="--", linewidth=1.6, alpha=.8, zorder=1)
        draw_segments(ax, xy, chosen_stages, faded_background=True)

    draw_panel(route_ax, stages, "Gesamtablauf / Draufsicht")
    low, high = bounds_for(np.vstack((xy, ref_xy)))
    route_ax.set_xlim(low[0], high[0]); route_ax.set_ylim(low[1], high[1])
    place_numbers(route_ax, xy, stages)
    scale = max(.22, min(.55, float(np.max(extent))*.055))
    # Endpoint arrows are omitted if either endpoint is not upright. Motion
    # direction arrows on the path use only measured XY tangents.
    for i, label, color, marker, offset in ((0, "START", "#2a7851", "o", (8, -23)),
                                           (-1, "ENDE", "#933f40", "s", (8, 12))):
        if measured["endpoints_upright"]:
            endpoint(route_ax, xy[i], measured["endpoint_yaw"][i], label, color, marker, scale, offset)
        else:
            route_ax.scatter(*xy[i], s=75, marker=marker, color=color, zorder=9)
            route_ax.annotate(label, xy[i], xytext=offset, textcoords="offset points", color=color)
    legend_ax.set_axis_off()
    legend_ax.set_title("Reihenfolge · Zeitfenster der Referenz", loc="left", fontsize=11,
                        fontweight="bold", color=INK, pad=12)
    entries = [(s, textwrap.wrap(s["label"], width=47) or [s["label"]]) for s in stages]
    row, line_h = .97, min(.072, .95/max(1, sum(len(lines)+.7 for _, lines in entries)))
    for stage, lines in entries:
        legend_ax.text(0, row, f'{stage["number"]:02d}', transform=legend_ax.transAxes,
                       color=stage["color"], weight="bold", fontsize=10, va="top")
        legend_ax.text(.085, row, "\n".join(lines), transform=legend_ax.transAxes,
                       color=INK, fontsize=9, va="top", linespacing=1.1)
        row -= line_h*len(lines)
        legend_ax.text(.085, row, f'{stage["reference_start_frame"]/fps:.2f}–'
                       f'{(stage["reference_end_frame_exclusive"]-1)/fps:.2f} s',
                       transform=legend_ax.transAxes, color="#62716f", fontsize=8, va="top")
        row -= line_h*.7
    if crawling:
        crawl_ax = fig.add_subplot(grid[1, 1])
        draw_panel(crawl_ax, crawling, "Detail: Hinlegen → Krabbeln → Aufstehen")
        actual_idx = np.concatenate([np.arange(s["start_frame"], s["end_frame_exclusive"]) for s in crawling])
        ref_idx = np.concatenate([np.arange(s["reference_start_frame"], s["reference_end_frame_exclusive"]) for s in crawling])
        clow, chigh = bounds_for(np.vstack((xy[actual_idx], ref_xy[ref_idx])), padding=.25)
        crawl_ax.set_xlim(clow[0], chigh[0]); crawl_ax.set_ylim(clow[1], chigh[1])
        route_ax.add_patch(Rectangle(clow, *(chigh-clow), fill=False, edgecolor="#7f8882", linestyle=":", linewidth=1))
        place_numbers(crawl_ax, xy, crawling)
        crawl_ax.tick_params(labelsize=8)
    height_ax.set_facecolor(PLOT)
    height_ax.plot(np.arange(len(reference["root"]))/fps, reference["root"][:, 2],
                   color="#727c79", linewidth=1.4, linestyle="--")
    for stage in stages:
        a, b = stage["start_frame"], stage["end_frame_exclusive"]
        height_ax.plot(measured["time"][a:b], root[a:b, 2], color=stage["color"], linewidth=1.7)
    height_ax.set_xlim(0, (len(reference["root"])-1)/fps)
    height_ax.set_title("Beckenhöhe — gemessen farbig / Referenz grau", loc="left", fontsize=10, color=INK, fontweight="bold")
    height_ax.set_xlabel("Sekunden seit erstem Playback-Zeitstempel"); height_ax.set_ylabel("Welt Z [m]")
    height_ax.grid(alpha=.22); height_ax.set_axisbelow(True)
    fig.text(.07, .061, "Zeitzuordnung: fortlaufende Controller-Frames → monotone PhysX-Zeitstempel; keine Latenzkorrektur. "
             "Start-/Endpfeile nur bei aufrechtem Körper.", fontsize=8.7, color="#5e6968")
    fig.text(.07, .039, "Alle Messpunkte bleiben erhalten, auch Schwanken und Überlagerungen. Die Beckenlinie ist kein Körperumriss "
             "und erlaubt keine Aussage zur Kollisionsfreiheit späterer Wände.", fontsize=8.7, color="#5e6968")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        fig.savefig(handle, format=output.suffix[1:], dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    if any(digest(path) != expected for path, expected in measured["source_sha256"].items()):
        raise RuntimeError("Input changed while plotting; image is not a stable source snapshot")
    report = {"kind": "measured_PhysX_pelvis_world_trajectory_with_native_reference_overlay",
              "image": str(output), "image_sha256": digest(output), "trial": str(measured["trial"]),
              "motion": str(reference["folder"]), "reference_frames": len(reference["root"]), "reference_fps": fps,
              "reference_duration_s": (len(reference["root"])-1)/fps,
              "playback_logger_index_range": [int(measured["play_indices"][0]), int(measured["play_indices"][-1])],
              "playback_monotonic_range_s": [float(measured["play_stamps"][0]), float(measured["play_stamps"][-1])],
              "measured_samples": len(xy), "measured_trace_row_range": [int(measured["trace_rows"][0]), int(measured["trace_rows"][-1])],
              "measured_elapsed_range_s": [float(measured["time"][0]), float(measured["time"][-1])],
              "mapped_reference_frame_range": [int(measured["reference_frames"][0]), int(measured["reference_frames"][-1])],
              "mapping": "searchsorted(controller_monotonic, trace_monotonic, side=right)-1; first<=trace<=last playback timestamp",
              "latency_adjustment": "none; causal timestamp association is not zero-latency tracking proof",
              "spatial_adjustments": "none: raw world XY and original reference XY, zero nominal spawn",
              "smoothing_resampling_decimation": "none; matplotlib path simplification disabled",
              "measured_pelvis_xy_path_m": distance, "measured_xy_extent_m": extent.tolist(),
              "measured_xy_min_m": xy.min(axis=0).tolist(), "measured_xy_max_m": xy.max(axis=0).tolist(),
              "measured_start_xyz_m": root[0].tolist(), "measured_end_xyz_m": root[-1].tolist(),
              "reference_start_xyz_m": reference["root"][0].tolist(), "reference_end_xyz_m": reference["root"][-1].tolist(),
              "support_free_during_playback": measured["support_free"], "max_measured_gap_s": measured["max_trace_gap_s"],
              "status_banner": status, "recorded_trial_passed": measured["summary"].get("passed"),
              "recorded_observed_fall": measured["summary"].get("observed_fall"),
              "recorded_ground_tracking": ground, "recorded_scene": measured["scene"],
              "collision_validation": False, "walls_drawn": False, "phases": stages,
              "source_sha256": measured["source_sha256"]}
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    if not 72 <= args.dpi <= 600:
        parser.error("--dpi must be between 72 and 600")
    try:
        reference = load_motion(args.motion)
        measured = load_trial(reference, args.trial)
        report = plot_actual(reference, measured, args.output, args.dpi)
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.error(str(error))
    print(json.dumps({key: report[key] for key in ("image", "reference_frames", "measured_samples",
                      "measured_xy_extent_m", "measured_pelvis_xy_path_m", "status_banner")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
