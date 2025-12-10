from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

chrome_options = Options()
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")

for year in range(6, 20):
    game = f"nba-2k{year}"
    url = f"https://eu.hoopshype.com/nba-2k/players/?game={game}"

    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)

    wait = WebDriverWait(driver, 10)
    all_players = []

    try:
        wait.until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))).click()
        time.sleep(1)
    except:
        pass

    last_first_player = None

    while True:
        rows = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "table tbody tr")))

        first_player = rows[0].text

        if first_player == last_first_player:
            break

        last_first_player = first_player

        for row in rows:
            try:
                name = row.find_element(By.CSS_SELECTOR, "div._0cD6l-__0cD6l-").text.strip()
                rating = row.find_element(By.CSS_SELECTOR, "td.RLrCiX__RLrCiX").text.strip()
                all_players.append({"Name": name, "Rating": rating})
            except:
                pass

        try:
            next_button_span = driver.find_element(By.CSS_SELECTOR,"button.hd3Vfp__hd3Vfp._3JhbLM__3JhbLM span.icon-caret-down-1")
            driver.execute_script("arguments[0].click();", next_button_span)
            time.sleep(2)
        except:
            break

    driver.quit()

    df = pd.DataFrame(all_players)
    df.to_csv(f"nba2k{year}_ratings.csv", index=False)
    print(df.head())
    print(f'Rating nba2k{year}: saved')

print('All csv files are saved')