"""
capture_screenshots.py
======================
Capture real screenshots of the live Streamlit dashboard for the README.

Usage:
    python tools/capture_screenshots.py [URL]

Streamlit Community Cloud serves the app inside an iframe, so we click the
sidebar nav *inside the app frame* but screenshot the outer page (which renders
the iframe content inline). One-off developer tool, not part of the runtime.
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else \
    "https://smart-traffic-command-center-5aa5hpssbwajldkgfmawc6.streamlit.app/"
OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

# (sidebar label substring, page-title substring, output filename, extra wait ms)
PAGES = [
    ("Executive Overview",      "Executive Command Overview", "01_executive_overview.png", 0),
    ("Live Incident Analytics", "Live Incident Analytics",    "02_live_analytics.png",     3000),
    ("Road Health Monitoring",  "Road Health Monitoring",     "03_road_health.png",        0),
    ("Traffic Hotspots",        "Traffic Hotspots",           "04_hotspots.png",           3000),
    ("Forecasting Center",      "Forecasting Center",         "05_forecasting.png",        30000),
    ("Resource Allocation",     "Resource Allocation",        "06_resources.png",          0),
    ("ML Predictions",          "ML Predictions",             "07_ml_predictions.png",     20000),
    ("Data Quality",            "Data Quality Monitoring",    "08_data_quality.png",       0),
]


def get_app_frame(page):
    """Return the frame that actually contains the Streamlit app (the iframe)."""
    for f in page.frames:
        try:
            if f.locator('section[data-testid="stSidebar"]').count() > 0:
                return f
        except Exception:
            continue
    return page.main_frame


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900},
                                device_scale_factor=2)
        print("Opening", URL)
        page.goto(URL, wait_until="domcontentloaded", timeout=120000)

        # Wait for the app frame + sidebar to exist (handles cold-start boot).
        frame = None
        for _ in range(40):
            frame = get_app_frame(page)
            if frame.locator('section[data-testid="stSidebar"]').count() > 0:
                break
            page.wait_for_timeout(3000)
        sidebar = frame.locator('section[data-testid="stSidebar"]')
        sidebar.wait_for(timeout=120000)
        print("App loaded; sidebar ready.")

        ok = 0
        for i, (label, title, fname, extra) in enumerate(PAGES):
            try:
                if i > 0:
                    sidebar.get_by_text(label, exact=False).first.click()
                # Wait for the target page title to render inside the frame.
                frame.locator(".cc-title", has_text=title).first.wait_for(timeout=150000)
                try:
                    page.wait_for_load_state("networkidle", timeout=20000)
                except Exception:
                    pass
                page.wait_for_timeout(4500 + extra)
                page.screenshot(path=str(OUT / fname), full_page=True)
                print("  [ok]", fname)
                ok += 1
            except Exception as exc:
                print("  [x]", fname, "->", str(exc)[:120])
        browser.close()
        print(f"Captured {ok}/{len(PAGES)} screenshots -> {OUT}")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
