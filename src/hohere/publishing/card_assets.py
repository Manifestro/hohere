"""Visual assets for the dataset card: a stereo listening sample and a timeline SVG.

Both are rendered from one real conversation of the export, with no extra dependencies.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import soundfile as sf

from hohere.schemas.dataset import Event, EventType
from hohere.storage.local import DatasetStore

SAMPLE_SECONDS = 40

# Event colours (Tableau-like palette, readable on light and dark pages).
COLORS = {
    EventType.NORMAL_TURN: {"A": "#4C78A8", "B": "#F58518"},
    EventType.BACKCHANNEL: "#54A24B",
    EventType.OVERLAP: "#B279A2",
    EventType.INTERRUPTION: "#E45756",
    EventType.SELF_CORRECTION: "#72B7B2",
}
LEGEND = [
    ("User turn", "#4C78A8"),
    ("Assistant turn", "#F58518"),
    ("Backchannel", "#54A24B"),
    ("Overlap", "#B279A2"),
    ("Interruption", "#E45756"),
    ("Self-correction", "#72B7B2"),
    ("Cutoff", "#222222"),
    ("Pause", "#D9D9D9"),
]


def _color(e: Event) -> str:
    c = COLORS.get(e.event_type, "#999999")
    return c[e.speaker] if isinstance(c, dict) else c


def timeline_svg(events: list[Event], sample_rate: int, seconds: float, width: int = 1000) -> str:
    """Two-lane timeline of the first `seconds` of a conversation."""
    left, right, lane_h = 120, 20, 34
    top_a, top_b = 52, 52 + lane_h + 14
    axis_y = top_b + lane_h + 10
    legend_y = axis_y + 38
    height = legend_y + 30
    span = width - left - right
    end_sample = int(seconds * sample_rate)

    def x(sample: int) -> float:
        return left + span * min(sample, end_sample) / end_sample

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Inter, Helvetica, Arial, sans-serif">',
        f'<rect width="{width}" height="{height}" rx="14" fill="#FFFFFF"/>',
        f'<text x="{left}" y="30" font-size="15" font-weight="700" fill="#222">'
        f"Two synchronized channels, first {seconds:.0f} s of one conversation</text>",
        f'<text x="{left - 14}" y="{top_a + 22}" font-size="13" text-anchor="end" '
        f'fill="#333">User · A</text>',
        f'<text x="{left - 14}" y="{top_b + 22}" font-size="13" text-anchor="end" '
        f'fill="#333">Assistant · B</text>',
        f'<rect x="{left}" y="{top_a}" width="{span}" height="{lane_h}" rx="6" fill="#F4F5F7"/>',
        f'<rect x="{left}" y="{top_b}" width="{span}" height="{lane_h}" rx="6" fill="#F4F5F7"/>',
    ]
    for e in events:
        if e.start_sample >= end_sample:
            continue
        x0, x1 = x(e.start_sample), x(e.end_sample)
        w = max(x1 - x0, 1.5)
        if e.event_type in (EventType.SHORT_PAUSE, EventType.LONG_PAUSE):
            parts.append(
                f'<rect x="{x0:.1f}" y="{top_a}" width="{w:.1f}" height="{top_b + lane_h - top_a}" '
                f'fill="#D9D9D9" opacity="0.45"/>'
            )
            continue
        if e.speaker is None:
            continue
        y = top_a if e.speaker == "A" else top_b
        if e.event_type is EventType.CUTOFF:
            parts.append(
                f'<rect x="{x1 - 2:.1f}" y="{y - 3}" width="3" height="{lane_h + 6}" '
                f'rx="1.5" fill="#222222"/>'
            )
            continue
        title = escape(f"{e.event_type.value}: {e.text or ''}")[:200]
        parts.append(
            f'<rect x="{x0:.1f}" y="{y + 3}" width="{w:.1f}" height="{lane_h - 6}" rx="5" '
            f'fill="{_color(e)}"><title>{title}</title></rect>'
        )
    for s in range(0, int(seconds) + 1, 5):
        tx = x(s * sample_rate)
        parts += [
            f'<line x1="{tx:.1f}" y1="{axis_y}" x2="{tx:.1f}" y2="{axis_y + 5}" stroke="#888"/>',
            f'<text x="{tx:.1f}" y="{axis_y + 18}" font-size="11" text-anchor="middle" '
            f'fill="#666">{s}s</text>',
        ]
    lx = left
    for label, color in LEGEND:
        parts += [
            f'<rect x="{lx}" y="{legend_y - 10}" width="12" height="12" rx="3" fill="{color}"/>',
            f'<text x="{lx + 17}" y="{legend_y}" font-size="12" fill="#333">{label}</text>',
        ]
        lx += 18 + 7.2 * len(label) + 16
    parts.append("</svg>")
    return "\n".join(parts)


def pick_sample(store: DatasetStore, ids: list[str]) -> str | None:
    """Prefer a train conversation that shows interruptions and backchannels early on."""
    best, best_score = None, -1
    for cid in ids[:400]:
        meta = store.load_metadata(cid)
        if meta.split != "train" or meta.quality_status != "passed":
            continue
        sr = meta.sample_rate
        early = [
            e.event_type
            for e in store.load_events(cid).events
            if e.start_sample < SAMPLE_SECONDS * sr
        ]
        score = (
            3 * early.count(EventType.INTERRUPTION)
            + 2 * early.count(EventType.BACKCHANNEL)
            + early.count(EventType.OVERLAP)
            + early.count(EventType.CUTOFF)
        )
        if score > best_score:
            best, best_score = cid, score
    return best


def write_card_assets(store: DatasetStore, ids: list[str], out_dir: Path) -> dict | None:
    """Write assets/sample_stereo.wav and assets/timeline.svg; returns their paths."""
    cid = pick_sample(store, ids)
    if cid is None:
        return None
    a, b, sr = store.load_channels(cid)
    n = min(len(a), SAMPLE_SECONDS * sr)
    seconds = n / sr
    assets = Path(out_dir) / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    stereo = np.stack([a[:n], b[:n]], axis=1)
    fade = min(n, sr // 2)
    stereo[n - fade :] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)[:, None]
    sf.write(assets / "sample_stereo.wav", stereo, sr, subtype="PCM_16")
    svg = timeline_svg(store.load_events(cid).events, sr, seconds)
    (assets / "timeline.svg").write_text(svg)
    return {
        "conversation_id": cid,
        "seconds": round(seconds, 1),
        "audio": "assets/sample_stereo.wav",
        "timeline": "assets/timeline.svg",
    }
