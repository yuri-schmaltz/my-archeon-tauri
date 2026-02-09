#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import numpy as np
import requests
from PIL import Image, ImageChops
from playwright.sync_api import Browser, Page, sync_playwright

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _mock_heavy_modules() -> None:
    heavy = [
        "hy3dgen.manager",
        "hy3dgen.inference",
        "hy3dgen.shapegen.utils",
        "diffusers",
        "transformers",
        "torch",
        "numba",
    ]
    for mod in heavy:
        sys.modules[mod] = MagicMock()


def _start_mock_server(host: str, port: int):
    _mock_heavy_modules()

    import gradio as gr
    import uvicorn
    from fastapi import FastAPI
    import hy3dgen.apps.gradio_app as gradio_app
    from hy3dgen.apps.ui_templates import CSS_STYLES

    gradio_app.request_manager = MagicMock()
    demo = gradio_app.build_app()

    app = FastAPI()
    app = gr.mount_gradio_app(
        app,
        demo,
        path="/",
        head=f"<style>{CSS_STYLES}</style>",
        theme=gr.themes.Base(
            font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
            font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "Consolas", "monospace"],
        ),
    )

    config = uvicorn.Config(app, host=host, port=port, log_level="error")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 40
    url = f"http://{host}:{port}/"
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=1.5)
            if r.status_code == 200:
                return server, thread
        except Exception:
            pass
        time.sleep(0.5)

    raise RuntimeError(f"UI server did not start at {url}")


def _click_by_labels(page: Page, labels: List[str]) -> bool:
    return bool(
        page.evaluate(
            """
            (labels) => {
              const q = [...document.querySelectorAll('button,[role="tab"],[role="button"]')];
              const normalized = labels.map(s => s.toLowerCase());
              for (const el of q) {
                const t = ((el.innerText || el.textContent || '').trim()).toLowerCase();
                if (!t) continue;
                if (normalized.some(n => t.includes(n))) {
                  el.click();
                  return true;
                }
              }
              return false;
            }
            """,
            labels,
        )
    )


def _mark_button(page: Page, labels: List[str], marker: str) -> bool:
    return bool(
        page.evaluate(
            """
            ({labels, marker}) => {
              const q = [...document.querySelectorAll('button,[role="button"]')];
              const normalized = labels.map(s => s.toLowerCase());
              for (const el of q) {
                if (el.getAttribute('role') === 'tab') continue;
                if (el.hasAttribute('data-tab-id')) continue;
                const t = ((el.innerText || el.textContent || '').trim()).toLowerCase();
                if (!t) continue;
                if (normalized.some(n => t.includes(n))) {
                  el.setAttribute(marker, '1');
                  return true;
                }
              }
              return false;
            }
            """,
            {"labels": labels, "marker": marker},
        )
    )


def _capture(path: Path, page: Page) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(path), full_page=True)


def _overflow_metrics(page: Page) -> Dict[str, Any]:
    return page.evaluate(
        """
        () => {
          const doc = document.documentElement;
          const body = document.body;
          const rootOverflow = Math.max(doc.scrollWidth, body.scrollWidth) - doc.clientWidth;
          const selectors = ['.main-row','.left-col','.right-col','.scroll-area','#gen_output_container','#model_3d_viewer'];
          const critical = selectors.map((s) => {
            const el = document.querySelector(s);
            if (!el) return { selector: s, missing: true };
            return {
              selector: s,
              missing: false,
              overflowX: el.scrollWidth - el.clientWidth,
              clientWidth: el.clientWidth,
              scrollWidth: el.scrollWidth,
            };
          });

          const left = document.querySelector('.left-col')?.getBoundingClientRect();
          const right = document.querySelector('.right-col')?.getBoundingClientRect();
          const topDelta = (left && right) ? Math.abs(left.top - right.top) : null;
          const stackedLayout = (left && right) ? Math.abs(left.top - right.top) > 40 : null;

          const footerButtons = [...document.querySelectorAll('.footer-area button')]
            .map((b) => b.getBoundingClientRect().height)
            .filter((h) => h > 0);
          const buttonHeightSpread = footerButtons.length
            ? Math.max(...footerButtons) - Math.min(...footerButtons)
            : null;

          return { rootOverflow, critical, topDelta, buttonHeightSpread, stackedLayout };
        }
        """
    )


def _collect_color_samples(page: Page) -> Dict[str, Any]:
    return page.evaluate(
        """
        () => {
          const toRgb = (value) => {
            if (!value || typeof value !== 'string') return null;
            const m = value.match(/rgba?\\(([^)]+)\\)/i);
            if (!m) return null;
            const parts = m[1].split(',').map((x) => Number.parseFloat(x.trim()));
            if (parts.length < 3 || parts.slice(0, 3).some((n) => Number.isNaN(n))) return null;
            const r = parts[0];
            const g = parts[1];
            const b = parts[2];
            const a = Number.isFinite(parts[3]) ? Math.max(0, Math.min(1, parts[3])) : 1;
            // Composite semitransparent colors on white for deterministic comparison.
            return [
              Math.round(r * a + 255 * (1 - a)),
              Math.round(g * a + 255 * (1 - a)),
              Math.round(b * a + 255 * (1 - a)),
            ];
          };

          const byButtonLabels = (labels, scopeSelector = '') => {
            const normalized = labels.map((s) => s.toLowerCase());
            const scoped = scopeSelector
              ? `${scopeSelector} button, ${scopeSelector} [role="button"]`
              : 'button,[role="button"]';
            const nodes = [...document.querySelectorAll(scoped)];
            for (const node of nodes) {
              if (node.getAttribute('role') === 'tab') continue;
              if (node.hasAttribute('data-tab-id')) continue;
              const text = ((node.innerText || node.textContent || '').trim()).toLowerCase();
              if (!text) continue;
              if (normalized.some((n) => text.includes(n))) return node;
            }
            return null;
          };

          const styleSample = (node) => {
            if (!node) return null;
            const cs = getComputedStyle(node);
            return {
              bg: toRgb(cs.backgroundColor),
              fg: toRgb(cs.color),
              border: toRgb(cs.borderTopColor),
            };
          };

          const findActionButton = (labels) =>
            byButtonLabels(labels, '.footer-area') || byButtonLabels(labels);

          const generate = styleSample(
            findActionButton(['generate 3d model', 'generate model', 'generate', 'gerar'])
          );
          const download = styleSample(
            findActionButton(['download .glb', 'download', 'baixar'])
          );

          const tabs = [...document.querySelectorAll('[role="tab"]')];
          const activeTab = tabs.find((t) => t.getAttribute('aria-selected') === 'true') || tabs[0] || null;
          const inactiveTab = tabs.find((t) => t !== activeTab) || null;

          const activeTabSample = styleSample(activeTab);
          const inactiveTabSample = styleSample(inactiveTab);

          const sample = {};
          if (generate) {
            sample.primary_generate_bg = generate.bg;
            sample.primary_generate_text = generate.fg;
            sample.primary_generate_border = generate.border;
          }
          if (download) {
            sample.primary_download_bg = download.bg;
            sample.primary_download_text = download.fg;
            sample.primary_download_border = download.border;
          }
          if (activeTabSample) {
            sample.tab_active_bg = activeTabSample.bg;
            sample.tab_active_text = activeTabSample.fg;
            sample.tab_active_border = activeTabSample.border;
          }
          if (inactiveTabSample) {
            sample.tab_inactive_bg = inactiveTabSample.bg;
            sample.tab_inactive_text = inactiveTabSample.fg;
            sample.tab_inactive_border = inactiveTabSample.border;
          }
          return sample;
        }
        """
    )


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    srgb = np.clip(rgb.astype(np.float64) / 255.0, 0.0, 1.0)
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)

    # sRGB D65 matrix
    xyz = np.empty_like(linear)
    xyz[..., 0] = (
        linear[..., 0] * 0.4124564 + linear[..., 1] * 0.3575761 + linear[..., 2] * 0.1804375
    )
    xyz[..., 1] = (
        linear[..., 0] * 0.2126729 + linear[..., 1] * 0.7151522 + linear[..., 2] * 0.0721750
    )
    xyz[..., 2] = (
        linear[..., 0] * 0.0193339 + linear[..., 1] * 0.1191920 + linear[..., 2] * 0.9503041
    )

    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    xyz_scaled = xyz / white

    delta = 6.0 / 29.0
    delta3 = delta**3
    f = np.where(
        xyz_scaled > delta3,
        np.cbrt(xyz_scaled),
        xyz_scaled / (3.0 * delta * delta) + 4.0 / 29.0,
    )

    lab = np.empty_like(f)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def _delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    l1 = lab1[..., 0]
    a1 = lab1[..., 1]
    b1 = lab1[..., 2]
    l2 = lab2[..., 0]
    a2 = lab2[..., 1]
    b2 = lab2[..., 2]

    c1 = np.sqrt(a1 * a1 + b1 * b1)
    c2 = np.sqrt(a2 * a2 + b2 * b2)
    c_bar = (c1 + c2) * 0.5

    c_bar7 = c_bar**7
    g = 0.5 * (1.0 - np.sqrt(c_bar7 / (c_bar7 + 25.0**7 + 1e-12)))

    a1_p = (1.0 + g) * a1
    a2_p = (1.0 + g) * a2
    c1_p = np.sqrt(a1_p * a1_p + b1 * b1)
    c2_p = np.sqrt(a2_p * a2_p + b2 * b2)

    h1_p = np.degrees(np.arctan2(b1, a1_p)) % 360.0
    h2_p = np.degrees(np.arctan2(b2, a2_p)) % 360.0

    d_l_p = l2 - l1
    d_c_p = c2_p - c1_p

    c_prod = c1_p * c2_p
    d_h_p = h2_p - h1_p
    d_h_p = np.where(c_prod == 0, 0.0, d_h_p)
    d_h_p = np.where((c_prod != 0) & (d_h_p > 180.0), d_h_p - 360.0, d_h_p)
    d_h_p = np.where((c_prod != 0) & (d_h_p < -180.0), d_h_p + 360.0, d_h_p)

    d_big_h_p = 2.0 * np.sqrt(c_prod) * np.sin(np.radians(d_h_p * 0.5))

    l_bar_p = (l1 + l2) * 0.5
    c_bar_p = (c1_p + c2_p) * 0.5

    h_sum = h1_p + h2_p
    h_diff = np.abs(h1_p - h2_p)
    h_bar_p = np.where(c_prod == 0, h_sum, (h1_p + h2_p) * 0.5)
    h_bar_p = np.where(
        (c_prod != 0) & (h_diff > 180.0) & (h_sum < 360.0),
        (h_sum + 360.0) * 0.5,
        h_bar_p,
    )
    h_bar_p = np.where(
        (c_prod != 0) & (h_diff > 180.0) & (h_sum >= 360.0),
        (h_sum - 360.0) * 0.5,
        h_bar_p,
    )

    t = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar_p - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar_p))
        + 0.32 * np.cos(np.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar_p - 63.0))
    )

    d_theta = 30.0 * np.exp(-(((h_bar_p - 275.0) / 25.0) ** 2))
    r_c = 2.0 * np.sqrt((c_bar_p**7) / (c_bar_p**7 + 25.0**7 + 1e-12))

    l_term = (l_bar_p - 50.0) ** 2
    s_l = 1.0 + 0.015 * l_term / np.sqrt(20.0 + l_term)
    s_c = 1.0 + 0.045 * c_bar_p
    s_h = 1.0 + 0.015 * c_bar_p * t
    r_t = -np.sin(np.radians(2.0 * d_theta)) * r_c

    d_l = d_l_p / s_l
    d_c = d_c_p / s_c
    d_h = d_big_h_p / s_h
    return np.sqrt(d_l * d_l + d_c * d_c + d_h * d_h + r_t * d_c * d_h)


def _delta_e_ciede2000_from_rgb(rgb_a: List[float], rgb_b: List[float]) -> float:
    arr_a = np.asarray(rgb_a, dtype=np.float64).reshape((1, 1, 3))
    arr_b = np.asarray(rgb_b, dtype=np.float64).reshape((1, 1, 3))
    lab_a = _srgb_to_lab(arr_a)
    lab_b = _srgb_to_lab(arr_b)
    return float(_delta_e_ciede2000(lab_a, lab_b).reshape(-1)[0])


def _image_delta_e_stats(img_a: Path, img_b: Path, max_pixels: int = 300_000) -> Dict[str, float]:
    a = np.asarray(Image.open(img_a).convert("RGB"), dtype=np.float64)
    b = np.asarray(Image.open(img_b).convert("RGB"), dtype=np.float64)

    if a.shape != b.shape:
        b_img = Image.fromarray(np.clip(b, 0, 255).astype(np.uint8), mode="RGB").resize(
            (a.shape[1], a.shape[0])
        )
        b = np.asarray(b_img, dtype=np.float64)

    total_pixels = int(a.shape[0] * a.shape[1])
    if total_pixels > max_pixels:
        stride = max(1, int(math.ceil(math.sqrt(total_pixels / max_pixels))))
        a = a[::stride, ::stride, :]
        b = b[::stride, ::stride, :]

    lab_a = _srgb_to_lab(a)
    lab_b = _srgb_to_lab(b)
    delta = _delta_e_ciede2000(lab_a, lab_b).reshape(-1)
    return {
        "mean": float(np.mean(delta)),
        "p95": float(np.percentile(delta, 95)),
        "max": float(np.max(delta)),
    }


def _diff_ratio(img_a: Path, img_b: Path, out_diff: Path) -> float:
    a = Image.open(img_a).convert("RGBA")
    b = Image.open(img_b).convert("RGBA")

    if a.size != b.size:
        b = b.resize(a.size)

    diff = ImageChops.difference(a, b)
    out_diff.parent.mkdir(parents=True, exist_ok=True)
    diff.save(out_diff)

    arr = np.asarray(diff)
    changed = int(np.any(arr != 0, axis=-1).sum())
    total = int(arr.shape[0] * arr.shape[1]) if arr.size else 0
    return (changed / total) * 100.0 if total else 0.0


def _capture_viewport(
    browser: Browser,
    base_url: str,
    name: str,
    viewport: Dict[str, Any],
    out_dir: Path,
) -> Tuple[Dict[str, Path], Dict[str, Any]]:
    context = browser.new_context(**viewport)
    page = context.new_page()
    page.goto(base_url, wait_until="domcontentloaded", timeout=60_000)
    page.wait_for_function(
        """
        () => [...document.querySelectorAll('button')]
          .some((b) => /generate|gerar/i.test((b.innerText || b.textContent || '').trim()))
        """,
        timeout=30_000,
    )
    page.wait_for_timeout(1400)

    shots: Dict[str, Path] = {}

    def snap(key: str) -> None:
        path = out_dir / name / f"{key}.png"
        _capture(path, page)
        shots[key] = path

    snap("01_default")

    _click_by_labels(page, ["multiview", "multi view", "multivis", "multi"])
    page.wait_for_timeout(350)
    snap("02_tab_multiview")

    _click_by_labels(page, ["texture", "textura"])
    page.wait_for_timeout(350)
    snap("03_tab_texture")

    marker = "data-ui-audit-generate"
    found = _mark_button(page, ["generate 3d model", "generate model", "generate", "gerar"], marker)
    if found:
        target = page.locator(f"[{marker}='1']").first
        try:
            target.hover(force=True, timeout=3_000)
            page.wait_for_timeout(250)
            snap("04_hover_generate")
        except Exception:
            snap("04_hover_generate_missing")

        try:
            target.focus(timeout=3_000)
            page.wait_for_timeout(250)
            snap("05_focus_generate")
        except Exception:
            snap("05_focus_generate_missing")

        page.evaluate(
            """
            (marker) => {
              const el = document.querySelector(`[${marker}='1']`);
              if (!el) return;
              el.setAttribute('disabled', '');
              el.setAttribute('aria-disabled', 'true');
            }
            """,
            marker,
        )
        page.wait_for_timeout(250)
        snap("06_disabled_generate_synthetic")
    else:
        # Keep deterministic key set if the button text changes.
        snap("04_hover_generate_missing")
        snap("05_focus_generate_missing")

    metrics = _overflow_metrics(page)
    metrics["colorSamples"] = _collect_color_samples(page)
    context.close()
    return shots, metrics


def _evaluate_criteria(
    metrics_by_viewport: Dict[str, Dict[str, Any]],
    delta_e_threshold: float,
    image_delta_e_by_viewport: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    overflow_failures = []
    alignment_failures = []
    button_height_failures = []

    for vp, m in metrics_by_viewport.items():
        root_overflow = float(m.get("rootOverflow", 0))
        if root_overflow > 0.5:
            overflow_failures.append((vp, "root", root_overflow))

        for c in m.get("critical", []):
            if c.get("missing"):
                continue
            ox = float(c.get("overflowX", 0))
            if ox > 0.5:
                overflow_failures.append((vp, c["selector"], ox))

        top_delta = m.get("topDelta")
        stacked = bool(m.get("stackedLayout"))
        if not stacked and top_delta is not None and float(top_delta) > 2.0:
            alignment_failures.append((vp, float(top_delta)))

        spread = m.get("buttonHeightSpread")
        if spread is not None and float(spread) > 2.0:
            button_height_failures.append((vp, float(spread)))

    results["no_horizontal_scroll"] = {
        "status": "PASS" if not overflow_failures else "FAIL",
        "details": overflow_failures,
    }
    results["column_alignment_delta_le_2px"] = {
        "status": "PASS" if not alignment_failures else "FAIL",
        "details": alignment_failures,
    }
    results["equivalent_button_height_spread_le_2px"] = {
        "status": "PASS" if not button_height_failures else "FAIL",
        "details": button_height_failures,
    }

    semantic_delta_e_measurements: List[Dict[str, Any]] = []
    semantic_delta_e_failures: List[Dict[str, Any]] = []

    # 1) Equivalent components inside each viewport.
    same_viewport_pairs = [
        ("primary_generate_bg", "primary_download_bg"),
        ("primary_generate_text", "primary_download_text"),
        ("primary_generate_border", "primary_download_border"),
    ]
    for vp, m in metrics_by_viewport.items():
        samples = m.get("colorSamples") or {}
        for left_key, right_key in same_viewport_pairs:
            left = samples.get(left_key)
            right = samples.get(right_key)
            if left is None or right is None:
                continue
            value = _delta_e_ciede2000_from_rgb(left, right)
            item = {
                "scope": vp,
                "pair": f"{left_key} vs {right_key}",
                "delta_e": round(value, 4),
            }
            semantic_delta_e_measurements.append(item)
            if value > delta_e_threshold:
                semantic_delta_e_failures.append(item)

    # 2) Same semantic tokens between desktop/mobile.
    cross_viewport_keys = [
        "primary_generate_bg",
        "primary_generate_text",
        "primary_generate_border",
        "primary_download_bg",
        "primary_download_text",
        "primary_download_border",
        "tab_active_bg",
        "tab_active_text",
        "tab_active_border",
        "tab_inactive_bg",
        "tab_inactive_text",
        "tab_inactive_border",
    ]
    desktop_samples = (metrics_by_viewport.get("desktop") or {}).get("colorSamples") or {}
    mobile_samples = (metrics_by_viewport.get("mobile") or {}).get("colorSamples") or {}
    for key in cross_viewport_keys:
        left = desktop_samples.get(key)
        right = mobile_samples.get(key)
        if left is None or right is None:
            continue
        value = _delta_e_ciede2000_from_rgb(left, right)
        item = {
            "scope": "desktop_vs_mobile",
            "pair": key,
            "delta_e": round(value, 4),
        }
        semantic_delta_e_measurements.append(item)
        if value > delta_e_threshold:
            semantic_delta_e_failures.append(item)

    results["delta_e_le_2"] = {
        "status": "PASS" if not semantic_delta_e_failures else "FAIL",
        "details": {
            "threshold": delta_e_threshold,
            "failures": semantic_delta_e_failures,
            "measurements": semantic_delta_e_measurements,
        },
    }

    # 3) Baseline screenshot color drift (optional, only when baseline exists).
    baseline_delta_e_measurements: List[Dict[str, Any]] = []
    baseline_delta_e_failures: List[Dict[str, Any]] = []
    for vp, diffs in image_delta_e_by_viewport.items():
        for shot, stats in diffs.items():
            value = float(stats.get("mean", 0.0))
            item = {
                "scope": f"{vp}_baseline",
                "pair": shot,
                "delta_e": round(value, 4),
                "p95": round(float(stats.get("p95", 0.0)), 4),
                "max": round(float(stats.get("max", 0.0)), 4),
            }
            baseline_delta_e_measurements.append(item)
            if value > delta_e_threshold:
                baseline_delta_e_failures.append(item)

    if baseline_delta_e_measurements:
        results["baseline_delta_e_le_2"] = {
            "status": "PASS" if not baseline_delta_e_failures else "FAIL",
            "details": {
                "threshold": delta_e_threshold,
                "failures": baseline_delta_e_failures,
                "measurements": baseline_delta_e_measurements,
            },
        }
    else:
        results["baseline_delta_e_le_2"] = {
            "status": "NAO_VERIFICADO",
            "details": "Baseline screenshots not available for color drift comparison.",
        }
    return results


def _write_report(
    out_root: Path,
    run_id: str,
    metrics_by_viewport: Dict[str, Dict[str, Any]],
    criteria: Dict[str, Any],
    intra_diff: Dict[str, Dict[str, float]],
    regression_diff: Dict[str, Dict[str, float]],
    regression_delta_e: Dict[str, Dict[str, Dict[str, float]]],
) -> Path:
    report_path = out_root / "reports" / f"{run_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append(f"# UI Visual Audit Report ({run_id})")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Current screenshots: `artifacts/ui_audit/current/{run_id}`")
    lines.append("- Baseline screenshots: `artifacts/ui_audit/baseline`")
    lines.append("")

    lines.append("## Criteria")
    for key, payload in criteria.items():
        lines.append(f"- `{key}`: **{payload['status']}**")
        lines.append(f"  Details: `{payload['details']}`")
    lines.append("")

    lines.append("## Overflow Metrics")
    for vp, metrics in metrics_by_viewport.items():
        lines.append(f"- `{vp}` rootOverflow: `{metrics.get('rootOverflow')}`")
        lines.append(f"  topDelta: `{metrics.get('topDelta')}`")
        lines.append(f"  buttonHeightSpread: `{metrics.get('buttonHeightSpread')}`")
        for c in metrics.get("critical", []):
            lines.append(
                f"  selector `{c.get('selector')}` overflowX=`{c.get('overflowX')}` missing=`{c.get('missing')}`"
            )
    lines.append("")

    lines.append("## Intra-run Screenshot Diff (%)")
    for vp, diffs in intra_diff.items():
        lines.append(f"- `{vp}`")
        for shot, pct in diffs.items():
            lines.append(f"  {shot}: `{pct:.4f}%` changed vs `01_default`")
    lines.append("")

    lines.append("## Regression Diff vs Baseline (%)")
    if regression_diff:
        for vp, diffs in regression_diff.items():
            lines.append(f"- `{vp}`")
            for shot, pct in diffs.items():
                lines.append(f"  {shot}: `{pct:.4f}%` changed vs baseline")
    else:
        lines.append("- Baseline not present for regression comparison in this run.")

    lines.append("")
    lines.append("## Regression DeltaE (CIEDE2000) vs Baseline")
    if regression_delta_e:
        for vp, shots in regression_delta_e.items():
            lines.append(f"- `{vp}`")
            for shot, stats in shots.items():
                lines.append(
                    f"  {shot}: mean=`{stats['mean']:.4f}` p95=`{stats['p95']:.4f}` max=`{stats['max']:.4f}`"
                )
    else:
        lines.append("- Baseline not present for DeltaE regression comparison in this run.")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _collect_critical_failures(
    criteria: Dict[str, Any], critical_criteria: List[str]
) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    for key in critical_criteria:
        payload = criteria.get(key)
        if payload is None:
            continue
        if payload.get("status") == "FAIL":
            failures.append({"criterion": key, "details": payload.get("details")})
    return failures


def run_audit(
    out_root: Path,
    host: str,
    port: int,
    update_baseline: bool,
    delta_e_threshold: float = 2.0,
    critical_criteria: List[str] | None = None,
) -> Dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = out_root / "current" / run_id

    server, _thread = _start_mock_server(host, port)
    base_url = f"http://{host}:{port}/"

    viewports = {
        "desktop": {"viewport": {"width": 1440, "height": 900}},
        "mobile": {
            "viewport": {"width": 390, "height": 844},
            "is_mobile": True,
            "has_touch": True,
            "device_scale_factor": 2,
        },
    }

    shots_by_vp: Dict[str, Dict[str, Path]] = {}
    metrics_by_viewport: Dict[str, Dict[str, Any]] = {}

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            for vp_name, vp in viewports.items():
                shots, metrics = _capture_viewport(browser, base_url, vp_name, vp, run_dir)
                shots_by_vp[vp_name] = shots
                metrics_by_viewport[vp_name] = metrics
            browser.close()
    finally:
        server.should_exit = True

    baseline_dir = out_root / "baseline"
    intra_diff_root = out_root / "diff" / run_id / "intra"
    regression_diff_root = out_root / "diff" / run_id / "regression"

    intra_diff: Dict[str, Dict[str, float]] = {}
    regression_diff: Dict[str, Dict[str, float]] = {}
    regression_delta_e: Dict[str, Dict[str, Dict[str, float]]] = {}

    for vp, shots in shots_by_vp.items():
        intra_diff[vp] = {}
        base_shot = shots.get("01_default")
        if base_shot is not None:
            for key, shot_path in shots.items():
                if key == "01_default":
                    continue
                diff_path = intra_diff_root / vp / f"01_default__{key}.png"
                intra_diff[vp][key] = _diff_ratio(base_shot, shot_path, diff_path)

        regression_diff[vp] = {}
        regression_delta_e[vp] = {}
        for key, shot_path in shots.items():
            baseline_path = baseline_dir / vp / f"{key}.png"
            if baseline_path.exists():
                diff_path = regression_diff_root / vp / f"baseline__{key}.png"
                regression_diff[vp][key] = _diff_ratio(baseline_path, shot_path, diff_path)
                regression_delta_e[vp][key] = _image_delta_e_stats(baseline_path, shot_path)

    has_any_baseline = any((baseline_dir / vp).exists() for vp in viewports)
    if update_baseline or not has_any_baseline:
        for vp, shots in shots_by_vp.items():
            vp_dir = baseline_dir / vp
            vp_dir.mkdir(parents=True, exist_ok=True)
            for key, shot in shots.items():
                shutil.copy2(shot, vp_dir / f"{key}.png")

    criteria = _evaluate_criteria(metrics_by_viewport, delta_e_threshold, regression_delta_e)
    report_path = _write_report(
        out_root=out_root,
        run_id=run_id,
        metrics_by_viewport=metrics_by_viewport,
        criteria=criteria,
        intra_diff=intra_diff,
        regression_diff={k: v for k, v in regression_diff.items() if v},
        regression_delta_e={k: v for k, v in regression_delta_e.items() if v},
    )

    critical_keys = critical_criteria or [
        "no_horizontal_scroll",
        "column_alignment_delta_le_2px",
        "equivalent_button_height_spread_le_2px",
        "delta_e_le_2",
    ]
    critical_failures = _collect_critical_failures(criteria, critical_keys)

    summary = {
        "run_id": run_id,
        "report": str(report_path),
        "current_dir": str(run_dir),
        "baseline_dir": str(baseline_dir),
        "criteria": criteria,
        "critical_criteria": critical_keys,
        "critical_failures": critical_failures,
        "overall_status": "FAIL" if critical_failures else "PASS",
    }
    (out_root / "reports" / f"{run_id}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run visual UI audit with screenshots and diffs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7866)
    parser.add_argument("--out", default="artifacts/ui_audit")
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--delta-e-threshold", type=float, default=2.0)
    parser.add_argument(
        "--critical-criteria",
        default=(
            "no_horizontal_scroll,column_alignment_delta_le_2px,"
            "equivalent_button_height_spread_le_2px,delta_e_le_2"
        ),
    )
    parser.add_argument("--fail-on-critical", action="store_true")
    parser.add_argument("--json-summary", default="")
    args = parser.parse_args()

    critical_criteria = [x.strip() for x in args.critical_criteria.split(",") if x.strip()]
    summary = run_audit(
        Path(args.out),
        args.host,
        args.port,
        args.update_baseline,
        delta_e_threshold=float(args.delta_e_threshold),
        critical_criteria=critical_criteria,
    )
    if args.json_summary:
        Path(args.json_summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.fail_on_critical and summary.get("overall_status") == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
