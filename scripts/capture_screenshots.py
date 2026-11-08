"""Exercise real uploads, recovery and mobile layout; save browser screenshots.

Uses system Chromium when available, otherwise Playwright's installed Chromium.
"""

import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

from fourierforensics.synthetic import jpeg_bytes, scene


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "docs/images"
    output.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            str(root / ".venv311/bin/streamlit") if (root / ".venv311").exists() else "streamlit",
            "run",
            "app.py",
            "--server.headless=true",
            "--server.port=8502",
            "--server.fileWatcherType=none",
            "--browser.gatherUsageStats=false",
        ],
        cwd=root,
        stdout=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen("http://localhost:8502/_stcore/health", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError("Streamlit failed to start")
        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=shutil.which("chromium"), args=["--no-sandbox"]
            )
            page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
            page.goto("http://localhost:8502")
            page.get_by_text("Download analysis JSON").wait_for(timeout=60000)
            page.screenshot(path=str(output / "streamlit-overview.png"), full_page=True)
            upload = page.locator("input[type=file]")
            upload.set_input_files(
                {"name": "camera.jpg", "mimeType": "image/jpeg", "buffer": jpeg_bytes(scene(123))}
            )
            page.get_by_role("heading", name="camera.jpg", exact=True).wait_for(timeout=60000)
            page.get_by_role("tab", name="Diagnostics & mathematics").click()
            page.locator('[data-testid="stStatusWidget"]').wait_for(state="hidden", timeout=60000)
            page.screenshot(path=str(output / "streamlit-diagnostics.png"), full_page=True)
            upload.set_input_files(
                {"name": "broken.png", "mimeType": "image/png", "buffer": b"invalid"}
            )
            page.get_by_text("Cannot decode a supported image safely").wait_for(timeout=30000)
            upload.set_input_files(
                {
                    "name": "recovered.jpg",
                    "mimeType": "image/jpeg",
                    "buffer": jpeg_bytes(scene(124)),
                }
            )
            page.get_by_role("heading", name="recovered.jpg", exact=True).wait_for(timeout=60000)
            page.set_viewport_size({"width": 390, "height": 844})
            page.get_by_role("tab", name="Image & evidence").click()
            page.locator('[data-testid="stStatusWidget"]').wait_for(state="hidden", timeout=60000)
            page.screenshot(path=str(output / "streamlit-mobile.png"), full_page=True)
            assert page.locator('[data-testid="stException"]').count() == 0
            browser.close()
    finally:
        process.terminate()
        process.wait(timeout=10)


if __name__ == "__main__":
    main()
