#!/usr/bin/env python3
"""Exercise the standalone viewer in an isolated local Firefox session."""
import argparse
import json
from pathlib import Path
import tempfile
import time


def audit(page, *, firefox, geckodriver):
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.firefox.service import Service
    from selenium.webdriver.support.ui import WebDriverWait

    page = Path(page).resolve()
    options = Options()
    options.add_argument('-headless')
    options.binary_location = firefox
    # A snap browser cannot access the host's isolated /tmp namespace.
    cache = Path(__file__).resolve().parents[1] / '.cache'
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='viewer-browser-', dir=cache) as profile_root:
        service = Service(geckodriver, service_args=['--profile-root', profile_root],
                          log_output=str(page.parent/'browser_audit.log'))
        driver = webdriver.Firefox(options=options, service=service)
        try:
            driver.set_window_size(1600,1050)
            driver.get(page.as_uri())
            wait = WebDriverWait(driver,20)
            wait.until(lambda d:d.execute_script('return typeof diagnosticViewer !== "undefined" && document.getElementById("video").readyState >= 2'))
            def state():
                return driver.execute_script('return diagnosticViewer.state()')
            def seek(t):
                driver.execute_script('diagnosticViewer.seek(arguments[0])',t)
                wait.until(lambda d:abs(state()['time']-t)<.035)
                return state()
            assert state()['index']==0
            assert 'Placement XY error' not in driver.find_element(By.ID,'facts').text
            assert not driver.find_element(By.ID,'episode-summary').get_attribute('open')
            assert not driver.find_element(By.ID,'event-list').get_attribute('open')
            cases = []
            for t, expected in ((5,'arm'),(10,'grasp'),(45,'navigation'),(124.5,'settling'),(130,'settled')):
                current=seek(t)
                assert current['context']==expected,(t,current)
                cases.append(current)
                if expected=='navigation':
                    assert 'Base translation speed' in driver.find_element(By.ID,'title-a').text
                    assert 'Placement XY error' not in driver.find_element(By.ID,'facts').text
                    driver.save_screenshot(str(page.parent/'preview.png'))
                if expected=='settled':
                    assert 'Placement XY error' in driver.find_element(By.ID,'facts').text
                    driver.save_screenshot(str(page.parent/'placement_preview.png'))
            # Backward seeking must revoke the displayed later outcome.
            seek(5)
            assert 'placed and settled' not in driver.find_element(By.ID,'status').text
            assert 'Placement XY error' not in driver.find_element(By.ID,'facts').text
            # Manual inspection does not change context or evaluator state.
            driver.execute_script('document.getElementById("mode").value="navigation"; document.getElementById("mode").dispatchEvent(new Event("change"))')
            assert 'Base translation speed' in driver.find_element(By.ID,'title-a').text
            assert state()['context']=='arm'
            driver.execute_script('document.getElementById("mode").value="auto"; document.getElementById("mode").dispatchEvent(new Event("change"))')
            # Navigate a warning, then check frame stepping and playback.
            driver.find_element(By.CSS_SELECTOR,'#event-list summary').click()
            warning=driver.find_element(By.CSS_SELECTOR,'.event.warning')
            warning_time=float(warning.get_attribute('data-time'))
            warning.click()
            wait.until(lambda d:abs(state()['time']-warning_time)<.035)
            assert driver.find_element(By.ID,'advanced').get_attribute('open')
            assert driver.find_element(By.ID,'active-warning').is_displayed()
            before=state()['index']
            driver.find_element(By.ID,'forward').click()
            wait.until(lambda d:state()['index']==before+1)
            driver.find_element(By.ID,'back').click()
            wait.until(lambda d:state()['index']==before)
            driver.execute_script('document.getElementById("speed").value="0.5"; document.getElementById("speed").dispatchEvent(new Event("change"))')
            driver.find_element(By.ID,'play').click()
            wait.until(lambda d:state()['index']>before+2)
            driver.find_element(By.ID,'play').click()
            assert driver.execute_script('return document.getElementById("video").playbackRate')==.5
            assert not driver.execute_script('return document.getElementById("video").error')
            # Responsive layout has no horizontal overflow.
            driver.set_window_size(720,1000)
            driver.execute_script('window.dispatchEvent(new Event("resize"))')
            assert driver.execute_script('return document.documentElement.scrollWidth <= window.innerWidth')
            report={'passed':True,'offline_file_playback':True,'context_checks':cases,
                    'backward_seek_hides_future_outcomes':True,'manual_plot_selection_is_display_only':True,
                    'warning_seek_and_frame_step':True,'half_speed_playback':True,
                    'responsive_layout':True,'video_error':None}
            (page.parent/'browser_audit.json').write_text(json.dumps(report,indent=2)+'\n')
            return report
        finally:
            driver.quit()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('page',type=Path)
    parser.add_argument('--firefox',default='/snap/firefox/current/usr/lib/firefox/firefox')
    parser.add_argument('--geckodriver',default='/snap/bin/geckodriver')
    args=parser.parse_args()
    print(json.dumps(audit(args.page,firefox=args.firefox,geckodriver=args.geckodriver),indent=2))
