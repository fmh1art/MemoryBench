"""Playwright client for full_smoke.py's isolated validation server only."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, required=True)
    args = parser.parse_args()
    receipt = {'validation_only': True, 'browser_errors': [], 'samples': []}
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=['--no-sandbox'])
            context = browser.new_context(viewport={'width': 1440, 'height': 1100})
            page = context.new_page()
            page.on('pageerror', lambda error: receipt['browser_errors'].append(str(error)))
            page.goto(args.url)
            page.locator('#name').fill('SCRIPTED-BROWSER-VALIDATION')
            page.locator('#login-form button[type=submit]').click()
            page.wait_for_selector('#workspace:not([hidden])')
            for i in range(args.count):
                task = page.request.get(args.url + '/api/next').json()['task']
                assert task
                required = 4 + len(task['references'])
                expect(page.locator('#video-A')).to_have_attribute('src', task['clips']['A']['url'])
                expect(page.locator('#media-state')).to_have_text(f'{required} 段视频已就绪', timeout=90000)
                sizes = page.evaluate("Object.fromEntries([...document.querySelectorAll('video')].filter(v=>v.getAttribute('src')).map(v=>[v.id,[v.videoWidth,v.videoHeight,v.readyState]]))")
                assert len(sizes) == required
                assert all(width > 0 and height > 0 and state >= 2 for width, height, state in sizes.values())
                assert page.locator('#sampling-note').is_visible()
                page.locator('input[name=choice][value=tie]').check()
                expect(page.locator('#save-state')).to_have_text('草稿已保存', timeout=15000)
                if i == 1:
                    page.reload()
                    expect(page.locator('#media-state')).to_have_text(f'{required} 段视频已就绪', timeout=90000)
                    assert page.locator('input[name=choice][value=tie]').is_checked()
                    assert page.request.get(args.url + '/api/next').json()['task']['assignment_id'] == task['assignment_id']
                page.locator('#play-all').click()
                page.wait_for_timeout(500)
                page.locator('#play-all').click()
                real_end = task['clips']['gt']['duration_seconds'] - task['clips']['gt']['start_seconds']
                if task['references'] and task['timeline_seconds'] > real_end + 1:
                    page.locator('#timeline').fill(str(round(real_end + .1, 2)))
                    page.locator('#timeline').dispatch_event('input')
                    page.wait_for_timeout(300)
                    page.locator('#play-all').click()
                    page.wait_for_timeout(600)
                    page.locator('#play-all').click()
                    master = task['playback_master']
                    elapsed = page.locator('#video-' + master).evaluate('video => video.currentTime') - task['clips'][master]['start_seconds']
                    assert elapsed > real_end + .3, 'Long simulation prediction stopped at real GT end'
                assert page.locator('#finished').is_hidden()
                if i == 2:
                    page.screenshot(path=str(args.output / 'simulation-references.png'), full_page=True)
                    page.set_viewport_size({'width': 390, 'height': 844})
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                    page.screenshot(path=str(args.output / 'mobile.png'), full_page=True)
                    page.set_viewport_size({'width': 1440, 'height': 1100})
                with page.expect_response(lambda response: response.url.endswith(f"/api/task/{task['assignment_id']}/vote")) as vote:
                    page.locator('#submit-vote').click()
                assert vote.value.status == 200
                expect(page.locator('#completed')).to_have_text(str(i + 1))
                receipt['samples'].append({'references': len(task['references']), 'decoded_videos': sizes})
            assert not receipt['browser_errors']
            receipt['status'] = 'passed'
            browser.close()
    finally:
        (args.output / 'browser-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
