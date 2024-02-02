from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import argparse
from bs4 import BeautifulSoup
import requests


WEB_URL = 'https://github.com/SimplifyJobs/Summer2024-Internships'


def main():
    driver = launchBrower()

    with open('table.html', 'r', encoding='utf-8') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')

    table = soup.find('table')

    table_body = table.find('tbody')

    tr_elements = table_body.find_all('tr')

    companies = []
    roles = []
    locations = []
    apply_links = []
    date_posted_arr = []
    job_descriptions = []

    prev_company = ""

    for i in range(400, 600):
        tr = tr_elements[i]

        company = tr.find_all('td')[0].text
        role = tr.find_all('td')[1].text
        location = tr.find_all('td')[2].text
        apply_area = tr.find_all('td')[3]
        date_posted = tr.find_all('td')[4].text

        try:
            apply_link = apply_area.find_all('a')[0].get('href')
        except:
            break

        if '↳' in company:
            company = prev_company

        prev_company = company

        driver.get(apply_link)
        time.sleep(10)
        try:
            body = driver.find_element_by_xpath(("//body"))
        except:
            continue

        job_description = body.text.lower()

        companies.append(company)
        roles.append(role)
        locations.append(location)
        apply_links.append(apply_link)
        date_posted_arr.append(date_posted)
        job_descriptions.append(job_description)

    df = pd.DataFrame({"Company": companies, "Role": roles,
                       "Location": locations, "Apply link": apply_links, "Date posted": date_posted_arr, "Job Description": job_descriptions})

    # df = pd.DataFrame({"Company": companies, "Role": roles,
    #                    "Location": locations, "Apply link": apply_links, "Date posted": date_posted_arr})

    df.to_csv("job_with_description_400_600.csv", index=False)
    df.to_json("job_without_description_400_600.json", orient="records")

    driver.quit()


def launchBrower():
    options = Options()
    options.headless = True
    options.add_argument('window-size=1920x1080')

    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    # driver.get(WEB_URL)
    # driver.maximize_window()  # Don't work well with headless = True

    return driver


if __name__ == '__main__':
    main()
