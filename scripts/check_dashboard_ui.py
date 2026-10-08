"""Use the existing local Playwright installation: ../project/.venv/bin/python scripts/check_dashboard_ui.py."""
from pathlib import Path
import hashlib,json,math
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/ui';url='http://127.0.0.1:8088'


def main():
    statefile=OUT/'operations/runtime_state.json';before=hashlib.sha256(statefile.read_bytes()).hexdigest()
    with sync_playwright() as pw:
        browser=pw.chromium.launch();page=browser.new_page(viewport={'width':1440,'height':1100});errors=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(url);page.wait_for_function("document.getElementById('k-version').textContent.includes('v')")
        assert page.locator('#global-error').is_hidden(),page.locator('#global-error').inner_text()
        assert page.locator('#k-gate').inner_text()=='FAIL'
        assert page.locator('#k-streak').inner_text()=='3일'
        page.screenshot(path=str(OUT/'dashboard_desktop.png'),full_page=True)
        page.get_by_role('tab',name='Prediction',exact=True).click();page.get_by_role('button',name='실험 입력 샘플 불러오기',exact=True).click()
        page.wait_for_function("document.getElementById('sequence').value.startsWith('[')")
        assert len(json.loads(page.locator('#sequence').input_value()))==168
        for _ in range(2):
            with page.expect_response(lambda r:r.url.endswith('/dashboard/predict-preview'),timeout=180000) as pending:page.get_by_role('button',name='예측 테스트',exact=True).click()
            r=pending.value;assert r.status==200,r.text();result=r.json()
            assert len(result['predictions'])==24 and all(math.isfinite(x['mwh']) and x['mwh']>=0 for x in result['predictions'])
            page.wait_for_function("document.getElementById('predict-status').textContent.includes('HTTP 200')")
            page.wait_for_function("!document.getElementById('preview-button').disabled")
        assert hashlib.sha256(statefile.read_bytes()).hexdigest()==before,'Preview mutated recorded operational forecasts'
        (OUT/'prediction_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        page.screenshot(path=str(OUT/'prediction_desktop.png'),full_page=True)
        values=json.loads(page.locator('#sequence').input_value());page.locator('#sequence').fill(json.dumps(values[:-1]))
        page.get_by_role('button',name='예측 테스트',exact=True).click();page.wait_for_function("document.getElementById('predict-status').textContent.includes('422')")
        page.get_by_role('tab',name='Drift Lab',exact=True).click();page.get_by_role('button',name='자동 실행 기록 재생',exact=True).click()
        page.wait_for_function("document.getElementById('replay-status').textContent.includes('Gate FAIL')")
        assert '-15%' in page.locator('#sensitivity').inner_text();assert 'Not detected' in page.locator('#sensitivity').inner_text()
        page.screenshot(path=str(OUT/'drift_lab_desktop.png'),full_page=True)
        page.get_by_role('tab',name='Logs',exact=True).click();page.get_by_role('button',name='FAIL',exact=True).click()
        assert page.locator('#logs-content .badge').all_text_contents()==['FAIL']
        page.screenshot(path=str(OUT/'logs_desktop.png'),full_page=True)
        for name in ['Overview','Prediction','Drift Lab','Datasets','Logs']:
            page.get_by_role('tab',name=name,exact=True).click();page.set_viewport_size({'width':390,'height':844})
            assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),name
            if name=='Overview':page.screenshot(path=str(OUT/'dashboard_mobile.png'),full_page=True)
        page.get_by_role('tab',name='Overview',exact=True).focus();page.keyboard.press('ArrowRight');assert page.get_by_role('tab',name='Prediction',exact=True).get_attribute('aria-selected')=='true'
        assert not errors,errors
        (OUT/'browser_validation.json').write_text(json.dumps({'status':'PASS','actual_preview_http':200,'prediction_count':24,'preview_repeatable_without_state_mutation':True,'length_167_422':True,'frozen_experiment_replay':True,'failure_is_visible':True,'log_severity_filter':True,'mobile_all_tabs_no_overflow':True,'keyboard_tabs':True,'javascript_errors':errors},indent=2))
        browser.close()
    print('Actual desktop/mobile/preview/replay/log/keyboard checks PASS')

if __name__=='__main__':main()
