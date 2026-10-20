"""Exercise real clips in Chromium using a NEW, isolated test database only.

Requires Playwright separately. Votes are scripted plumbing checks, not ratings.
python evaluation/pairwise/tests/browser_smoke.py --study ... --output ...
"""

import argparse
import hashlib
import itertools
import json
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-python", default=sys.executable)
    parser.add_argument("--chromium", help="Optional installed Chromium executable")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    db_path = args.output / "smoke.sqlite3"
    if db_path.exists():
        raise ValueError("Use a new output folder; smoke tests never reuse or modify a voting database")
    root = Path(__file__).resolve().parents[3]
    study = json.loads(args.study.read_text())
    quota = study.get("raters_per_pair", 3)
    expected = len(study["cases"]) * (len(study["models"]) - 1 if study.get("baseline") else len(list(itertools.combinations(study["models"], 2))))
    if len(study["cases"]) > 3 or quota > 3:
        raise ValueError("This smoke test is limited to 1–3 cases and at most 3 raters per pair")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log = (args.output / "server.log").open("w")
    process = subprocess.Popen([
        args.server_python, "-m", "evaluation.pairwise.server", "--study", str(args.study.resolve()),
        "--db", str(db_path), "--port", str(port),
    ], cwd=root, stdout=log, stderr=log)
    receipt = {"validation_only": True, "note": "SCRIPTED TEST VOTES; DO NOT USE AS HUMAN PREFERENCES",
               "examples": [], "browser_errors": [], "submitted": 0}
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(url + "/api/config", timeout=1).close()
                break
            except OSError:
                if process.poll() is not None:
                    raise RuntimeError("Test server exited; inspect server.log")
                time.sleep(.1)
        else:
            raise RuntimeError("Test server did not start")
        with sync_playwright() as playwright:
            options = {"headless": True, "args": ["--no-sandbox"]}
            if args.chromium:
                options["executable_path"] = args.chromium
            browser = playwright.chromium.launch(**options)
            for rater in range(quota):
                context = browser.new_context(viewport={"width": 1440, "height": 1450})
                page = context.new_page()
                page.on("pageerror", lambda error: receipt["browser_errors"].append(str(error)))
                page.goto(url)
                page.locator("#name").fill(f"流程测试-{rater + 1}")
                page.locator("#login-form button[type=submit]").click()
                page.wait_for_selector("#workspace:not([hidden])")
                page.locator(".session-details").evaluate("node => node.open = false")
                for item in range(expected):
                    expect(page.locator("#media-state")).to_have_text("4 段视频已就绪", timeout=60000)
                    response = page.request.get(url + "/api/next").json()
                    task = response["task"]
                    assert task, (rater, item, response)
                    aid = task["assignment_id"]
                    assert all(m not in json.dumps(task) for m in study["models"])
                    assert page.locator("#common-prompt").inner_text()
                    assert abs(float(page.locator("#read-interval").get_attribute("x")) - 1000 * task["read_start_seconds"] / task["timeline_seconds"]) < .01
                    assert abs(float(page.locator("#read-interval").get_attribute("width")) - 1000 * ((task["read_end_seconds"] or task["timeline_seconds"]) - task["read_start_seconds"]) / task["timeline_seconds"]) < .01
                    sizes = page.evaluate("Object.fromEntries([...document.querySelectorAll('video')].map(v=>[v.id,[v.videoWidth,v.videoHeight,v.readyState]]))")
                    assert all(width > 0 and height > 0 and state >= 2 for width, height, state in sizes.values())
                    assert page.locator('#vote-form input[type="radio"]').count() == 3
                    assert page.locator('#vote-form input[type="radio"]:checked').count() == 0
                    choice = ("A", "B", "tie")[(rater + item) % 3]
                    page.locator(f'input[name="choice"][value="{choice}"]').check()
                    expect(page.locator("#save-state")).to_have_text("草稿已保存")
                    if item == 0:
                        original = page.locator("#video-A").get_attribute("src")
                        page.reload()
                        expect(page.locator("#media-state")).to_have_text("4 段视频已就绪", timeout=60000)
                        assert page.locator("#video-A").get_attribute("src") == original
                        assert page.locator(f'input[name="choice"][value="{choice}"]').is_checked()
                        if rater == 0:
                            page.set_viewport_size({"width": 390, "height": 844})
                            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                            page.locator("#vote-form").screenshot(path=str(args.output / "mobile-voting.png"))
                            page.set_viewport_size({"width": 1440, "height": 1450})
                            page.locator("#vote-form").screenshot(path=str(args.output / "voting-form.png"))
                    if rater == 0 and item < min(3, len(study["cases"])):
                        page.locator("#timeline").fill("0.5")
                        page.locator("#timeline").dispatch_event("input")
                        page.locator("#play-all").click()
                        page.wait_for_timeout(600)
                        page.locator("#play-all").click()
                        times = page.evaluate("['gt','A','B'].map(s=>document.querySelector('#video-'+s).currentTime)")
                        offsets = [task["clips"][s]["start_seconds"] for s in ("gt", "A", "B")]
                        relative = [a - b for a, b in zip(times, offsets)]
                        assert max(relative) - min(relative) < .25, relative
                        assert all(t > .65 for t in relative), relative
                        page.locator(".session-details").evaluate("node => node.open = false")
                        page.evaluate("window.scrollTo(0, 260)")
                        screenshot = args.output / f"example-{item + 1}.png"
                        page.screenshot(path=str(screenshot))
                        with sqlite3.connect(db_path) as db:
                            task_id = db.execute("SELECT task FROM assignments WHERE id=?", (aid,)).fetchone()[0]
                        case_id = None
                        for case in study["cases"]:
                            for a, b in itertools.combinations(sorted(study["models"]), 2):
                                hashed = hashlib.sha256(json.dumps([case["case_id"], a, b], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]
                                if hashed == task_id:
                                    case_id = case["case_id"]
                        receipt["examples"].append({"case_id": case_id, "screenshot": str(screenshot),
                                                    "decoded_sizes": sizes, "synchronized_times": relative})
                    with page.expect_response(lambda r: r.url.endswith(f"/api/task/{aid}/vote")) as vote:
                        page.locator("#submit-vote").click()
                    assert vote.value.status == 200, vote.value.text()
                    receipt["submitted"] += 1
                    expect(page.locator("#completed")).to_have_text(str(item + 1))
                    if item + 1 < expected:
                        expect(page.locator("#video-A")).not_to_have_attribute("src", task["clips"]["A"]["url"])
                page.wait_for_selector("#finished:not([hidden])")
                context.close()
            # Also inspect the mobile layout without adding another vote.
            context = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True)
            page = context.new_page(); page.goto(url)
            page.wait_for_selector("#welcome:not([hidden])")
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.screenshot(path=str(args.output / "mobile-welcome.png"), full_page=True)
            context.close(); browser.close()
        assert not receipt["browser_errors"], receipt["browser_errors"]
        assert len({e["case_id"] for e in receipt["examples"]}) == min(3, len(study["cases"]))
        subprocess.run([args.server_python, "-m", "evaluation.pairwise.report", "--study", str(args.study.resolve()),
                        "--db", str(db_path), "--output", str(args.output / "report"), "--bootstrap", "1000"], cwd=root, check=True)
        summary = json.loads((args.output / "report/summary.json").read_text())
        assert summary["coverage"]["complete"]
        assert receipt["submitted"] == expected * quota
        overall = summary["groups"]["Overall"]
        assert "dimensions" not in overall and "gates" not in overall
        assert sum(pair["games"] for pair in overall["pairs"]) == receipt["submitted"]
        assert sum(pair["ties"] for pair in overall["pairs"]) == receipt["submitted"] // 3
        receipt["status"] = "passed"
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
    finally:
        (args.output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
        log.close()


if __name__ == "__main__":
    main()
