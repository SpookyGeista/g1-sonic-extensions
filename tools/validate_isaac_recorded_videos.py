#!/usr/bin/env python3
"""CPU-only integrity and edit validation for recorded Isaac video exports.

This checks encoded media and provenance, not motion tracking or collision
validity. No simulator, DDS, renderer, or GPU decoder is started.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

FFMPEG = Path('/home/loya2/miniconda3/envs/unitree_sim_env/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2')
FFPROBE = Path('/home/loya2/miniconda3/envs/fruit-detection/bin/ffprobe')
VIEWS = ('top', 'first', 'chase')
NAMES = (*VIEWS, 'mix')
HASH = re.compile(r'^[a-f0-9]{64}$')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def choose_view(edit, seconds):
    found = [item['camera'] for item in edit
             if item['start_s'] <= seconds < item['end_s']]
    if len(found) != 1:
        raise ValueError(f'Expected one edit segment at {seconds}s, found {len(found)}')
    if found[0] not in VIEWS:
        raise ValueError(f'Unknown camera {found[0]}')
    return found[0]


def audit_edit(edit, frame_count, fps, full_export):
    errors = []
    if not edit or abs(edit[0]['start_s']) > 1e-9:
        raise ValueError('Edit must start at zero')
    for index, item in enumerate(edit):
        if item['camera'] not in VIEWS:
            errors.append(f'Unknown camera at segment {index}')
        if not (math.isfinite(item['start_s']) and math.isfinite(item['end_s'])):
            errors.append(f'Nonfinite time at segment {index}')
        if item['end_s'] <= item['start_s']:
            errors.append(f'Empty or reversed segment {index}')
        if index and abs(edit[index-1]['end_s'] - item['start_s']) > 1e-9:
            errors.append(f'Gap/overlap before segment {index}')
    if edit[-1]['end_s'] < frame_count / fps - 1e-6:
        errors.append('Edit does not cover complete encoded duration')
    if errors:
        return {'passed': False, 'errors': errors}
    selected = [choose_view(edit, index / fps) for index in range(frame_count)]
    counts = {name: selected.count(name) for name in VIEWS}
    all_three = all(counts.values())
    if full_export and not all_three:
        errors.append('Full export does not actually select all three camera views')
    return {'passed': not errors, 'errors': errors, 'selected_frame_counts': counts,
            'all_three_views_selected': all_three,
            'all_three_required': full_export,
            'short_export_note': None if full_export else 'Short smoke export may end before later camera cuts'}


def comparison_indices(edit, frame_count, fps):
    """Cover first/last, every cut on both sides, and each visible segment middle."""
    indices = {0, frame_count - 1}
    for item in edit:
        first = int(math.ceil(item['start_s'] * fps - 1e-7))
        last = min(frame_count, int(math.ceil(item['end_s'] * fps - 1e-7)))
        if 0 <= first < frame_count:
            indices.update(i for i in (first-1, first, first+1, (first+last-1)//2)
                           if 0 <= i < frame_count)
    return sorted(indices)


def rgb_hash_audit(records, frame_count):
    errors, views = [], {}
    if len(records) != frame_count:
        errors.append(f'RGB hash records {len(records)} != frame count {frame_count}')
    for name in VIEWS:
        values = [record.get(name, '') for record in records]
        invalid = [i for i, value in enumerate(values)
                   if not isinstance(value, str) or not HASH.fullmatch(value)]
        repeats = [i for i in range(1, len(values)) if values[i] == values[i-1]]
        views[name] = {'unique_hashes': len(set(values)), 'consecutive_repeated_frames': repeats,
                       'invalid_hash_indices': invalid}
        if invalid:
            errors.append(f'{name}: malformed raw RGB hashes')
        if repeats:
            errors.append(f'{name}: identical consecutive raw RGB frames')
        if frame_count > 1 and len(set(values)) <= 1:
            errors.append(f'{name}: no changing raw rendered frames')
    return {'passed': not errors, 'errors': errors, 'views': views,
            'scope': 'Exporter-recorded pre-encoding RGB hashes; independent decoded edit checks below'}


def compare_rgb(actual, expected, mean_limit=6.0, p95_limit=20.0):
    if actual.shape != expected.shape:
        raise ValueError('Comparison frames differ in shape')
    error = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    mean = float(error.mean())
    p95 = float(np.percentile(error, 95))
    return {'mean_absolute_8bit_error': mean, 'p95_absolute_8bit_error': p95,
            'passed': mean <= mean_limit and p95 <= p95_limit}


def probe_video(path, ffprobe):
    result = subprocess.run([str(ffprobe), '-v', 'error', '-threads', '2',
        '-count_frames', '-show_streams', '-show_format', '-of', 'json', str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180, check=False)
    if result.returncode or result.stderr.strip():
        raise RuntimeError(f'ffprobe failed: {result.stderr.decode(errors="replace")[:2000]}')
    return json.loads(result.stdout)


def check_probe(probe, expected_frames, fps, resolution):
    errors = []
    streams = probe['streams']
    videos = [stream for stream in streams if stream.get('codec_type') == 'video']
    audio = [stream for stream in streams if stream.get('codec_type') == 'audio']
    if len(videos) != 1:
        raise ValueError(f'Expected one video stream, found {len(videos)}')
    video = videos[0]
    duration = float(video.get('duration', probe['format']['duration']))
    measured = {'width': int(video['width']), 'height': int(video['height']),
        'fps': float(Fraction(video['avg_frame_rate'])),
        'nominal_fps': float(Fraction(video['r_frame_rate'])),
        'decoded_frames': int(video['nb_read_frames']),
        'duration_s': duration, 'container_duration_s': float(probe['format']['duration']),
        'audio_streams': len(audio), 'codec': video['codec_name'],
        'pixel_format': video.get('pix_fmt')}
    if [measured['width'], measured['height']] != list(resolution):
        errors.append('Resolution differs from provenance')
    if measured['fps'] != fps or measured['nominal_fps'] != fps:
        errors.append('Frame rate differs from provenance')
    if measured['decoded_frames'] != expected_frames:
        errors.append('Decoded frame count differs from provenance')
    if video.get('nb_frames', 'N/A') != 'N/A' and int(video['nb_frames']) != expected_frames:
        errors.append('Container frame count differs from provenance')
    if abs(duration - expected_frames / fps) > 1e-3:
        errors.append('Video duration differs from expected CFR duration')
    if abs(measured['container_duration_s'] - expected_frames / fps) > 1e-3:
        errors.append('Container duration differs from expected CFR duration')
    if audio:
        errors.append('Unexpected audio stream')
    return {'passed': not errors, 'errors': errors, **measured}


def decode_audit(path, ffmpeg):
    result = subprocess.run([str(ffmpeg), '-hide_banner', '-v', 'error', '-nostdin',
        '-xerror', '-err_detect', 'explode', '-threads', '2', '-i', str(path),
        '-map', '0:v:0', '-an', '-f', 'null', '-'], stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE, timeout=180, check=False)
    stderr = result.stderr.decode(errors='replace')
    return {'passed': result.returncode == 0 and not stderr.strip(),
            'returncode': result.returncode, 'stderr': stderr[:4000],
            'decoder': 'FFmpeg CPU, complete stream, -xerror -err_detect explode'}


def extract_frames(path, indices, ffmpeg, width=320, height=180):
    select = '+'.join(f'eq(n\\,{index})' for index in indices)
    result = subprocess.run([str(ffmpeg), '-hide_banner', '-v', 'error', '-nostdin',
        '-threads', '2', '-i', str(path), '-map', '0:v:0', '-an',
        '-vf', f'select={select},scale={width}:{height}:flags=area',
        '-fps_mode', 'passthrough', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180, check=False)
    if result.returncode or result.stderr.strip():
        raise RuntimeError(f'Frame extraction failed: {result.stderr.decode(errors="replace")[:2000]}')
    size = width * height * 3
    if len(result.stdout) != len(indices) * size:
        raise ValueError('Decoded comparison frame count mismatch')
    arrays = np.frombuffer(result.stdout, dtype=np.uint8).reshape(len(indices), height, width, 3)
    return dict(zip(indices, arrays))


def validate(export_dir, ffmpeg=FFMPEG, ffprobe=FFPROBE, compare_mix=True):
    export_dir = Path(export_dir).resolve()
    provenance_path = export_dir / 'provenance.json'
    provenance = json.loads(provenance_path.read_text())
    count = int(provenance['video_frames'])
    fps = float(provenance['video_fps'])
    resolution = provenance['resolution']
    duration = count / fps
    full = count == int(math.floor(provenance['source_playback_duration_s'] * fps + 1e-5)) + 1
    errors = []
    if count < 2 or fps != 25:
        raise ValueError('Expected at least two frames at 25 fps')
    if provenance.get('status') != 'rendered':
        errors.append('Export did not finish with rendered status')
    if not provenance.get('no_dds') or provenance.get('physics_steps_advanced') != 0:
        errors.append('Provenance does not confirm render-only reconstruction')
    if abs(provenance['encoded_duration_s'] - duration) > 1e-6:
        errors.append('Provenance duration inconsistent with frame count and FPS')
    if set(provenance['videos']) != set(NAMES):
        raise ValueError('Expected exactly top, first, chase, mix videos in provenance')
    report = {'kind': 'CPU encoded-video validation, not physical motion validation',
        'export_dir': str(export_dir), 'provenance_sha256': sha256(provenance_path),
        'expected_frames': count, 'expected_fps': fps, 'expected_resolution': resolution,
        'expected_duration_s': duration, 'full_export': full,
        'ffmpeg': str(ffmpeg), 'ffprobe': str(ffprobe),
        'source_ground_tracking_passed': provenance.get('ground_tracking_passed'),
        'source_overall_passed': provenance.get('source_overall_passed'),
        'does_not_override_source_tracking_result': True, 'videos': {}}
    paths = {}
    for name in NAMES:
        info = provenance['videos'][name]
        path = (export_dir / info['file']).resolve()
        if path.parent != export_dir or path.suffix != '.mp4':
            raise ValueError('Video must be an MP4 directly inside export directory')
        paths[name] = path
        actual_hash = sha256(path)
        item = {'file': path.name, 'sha256': actual_hash,
                'sha256_matches': actual_hash == info['sha256'],
                'bytes_match': path.stat().st_size == info['bytes'],
                'provenance_frame_count_matches': info['frames'] == count}
        item['probe'] = check_probe(probe_video(path, ffprobe), count, fps, resolution)
        item['decode'] = decode_audit(path, ffmpeg)
        item['passed'] = all(item[key] for key in
            ('sha256_matches', 'bytes_match', 'provenance_frame_count_matches')) \
            and item['probe']['passed'] and item['decode']['passed']
        report['videos'][name] = item
        if not item['passed']:
            errors.append(f'{name}: video integrity/stream/decode check failed')
    report['raw_rgb_hashes'] = rgb_hash_audit(provenance.get('frame_rgb_hashes', []), count)
    if not report['raw_rgb_hashes']['passed']:
        errors.extend(report['raw_rgb_hashes']['errors'])
    report['edit'] = audit_edit(provenance['mix_edit'], count, fps, full)
    if not report['edit']['passed']:
        errors.extend(report['edit']['errors'])
    if compare_mix and report['edit']['passed']:
        selected = comparison_indices(provenance['mix_edit'], count, fps)
        mixed = extract_frames(paths['mix'], selected, ffmpeg)
        comparisons = []
        for name in VIEWS:
            indices = [i for i in selected if choose_view(provenance['mix_edit'], i / fps) == name]
            if not indices:
                continue
            reference = extract_frames(paths[name], indices, ffmpeg)
            for index in indices:
                comparisons.append({'frame': index, 'time_s': index / fps, 'camera': name,
                    **compare_rgb(mixed[index], reference[index])})
        comparisons.sort(key=lambda item: item['frame'])
        passed = all(item['passed'] for item in comparisons)
        report['decoded_mix_comparison'] = {'performed': True, 'passed': passed,
            'comparison_size': [320, 180], 'mean_8bit_error_limit': 6.0,
            'p95_8bit_error_limit': 20.0, 'frames': comparisons,
            'note': 'Same time and selected view; lossy independent encodes need not have identical pixels'}
        if not passed:
            errors.append('Decoded mixed frames differ from selected same-time source views')
    else:
        report['decoded_mix_comparison'] = {'performed': False, 'passed': None}
    # Source code may legitimately change AFTER export. Report this separately
    # from encoded-file integrity; hashes still document the original source.
    changes = []
    for filename, expected in provenance.get('source_sha256', {}).items():
        source = Path(filename)
        actual = sha256(source) if source.is_file() else None
        if actual != expected:
            changes.append({'file': filename, 'expected_sha256': expected, 'current_sha256': actual})
    report['source_hash_audit'] = {'current_files_match': not changes, 'changes': changes,
        'source_files_unchanged_during_export_claim': provenance.get('source_files_unchanged'),
        'note': 'Post-export source changes are reported, not treated as corruption of the encoded videos'}
    report.update(passed=not errors, errors=errors)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='NEW JSON path; never overwritten')
    parser.add_argument('--ffmpeg', type=Path, default=FFMPEG)
    parser.add_argument('--ffprobe', type=Path, default=FFPROBE)
    parser.add_argument('--skip-mix-comparison', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a NEW validation JSON')
    if not args.output.parent.is_dir():
        parser.error('Output parent directory must already exist')
    if not args.ffmpeg.is_file() or not args.ffprobe.is_file():
        parser.error('FFmpeg and ffprobe executables must exist locally')
    try:
        report = validate(args.export_dir, args.ffmpeg, args.ffprobe, not args.skip_mix_comparison)
    except Exception as error:
        report = {'passed': False, 'export_dir': str(args.export_dir.resolve()),
            'kind': 'CPU encoded-video validation, not physical motion validation',
            'errors': [f'{type(error).__name__}: {error}']}
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'passed': report['passed'], 'report': str(args.output.resolve()),
                      'errors': report['errors']}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
