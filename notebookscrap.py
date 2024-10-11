import sys
sys.path.insert(0, 'chromedriver.exe')
import requests
import time
from flask import Flask, jsonify, request
from bs4 import BeautifulSoup
from selenium import webdriver
import chromedriver_autoinstaller
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
from neo4j import GraphDatabase
from apscheduler.schedulers.background import BackgroundScheduler

# Neo4j configuration
URI = "neo4j://localhost"
AUTH = ("neo4j", "12345678")

# Function to run queries against Neo4j
def run_query(query, parameters=None):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        with driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

# Function to save notebook data to Neo4j
def save_notebook_to_neo4j(name, price, view, link):
    query = """
    MERGE (n:Notebook {name: $name})
    SET n.price = $price, n.view = $view, n.link = $link
    """
    parameters = {
        'name': name,
        'price': price,
        'view': view,
        'link': link
    }
    run_query(query, parameters)

# Setup Chrome options
chrome_options = webdriver.ChromeOptions()
# chrome_options.add_argument('--headless')  # Uncomment if you want to run without GUI
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

# Install chromedriver
chromedriver_autoinstaller.install()

# Create Flask app
app = Flask(__name__)

# Scheduler setup
scheduler = BackgroundScheduler()

# Function to trigger the /api route
def scrape_via_api():
    try:
        # Call the /api route on localhost
        response = requests.get('http://127.0.0.1:7777/api')
        if response.status_code == 200:
            print("Scraping successful!")
        else:
            print(f"Failed to scrape, status code: {response.status_code}")
    except Exception as e:
        print(f"Error occurred while calling /api: {e}")

# Add the scheduled job to run every 5 minutes
scheduler.add_job(func=scrape_via_api, trigger="interval", minutes=5)
scheduler.start()

# Route for testing the API
@app.route('/')
def index():
    return "<h1>Test API</h1>"

# Route for scraping data
@app.route('/api', methods=['GET'])
def api():
    if request.method == 'GET':
        # set the target URL
        url = "https://www.advice.co.th/product/notebooks"

        # set up the webdriver
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        driver.implicitly_wait(10)  # Wait for the page to load
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        # Wait until the first product is visible
        wait = WebDriverWait(driver, 10)
        wait.until(EC.visibility_of_element_located((By.CLASS_NAME, "item")))

        # Scroll to the end of the page and keep loading more products
        products_list = []
        prices_list = []
        views_list = []
        links_list = []
        old_items_size = 0
        notebook_data = []  # Declare an empty list for the notebook data

        while True:
            # Scroll to the bottom of the page
            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
            time.sleep(2)  # Wait for new elements to load
            products = driver.find_elements(By.CLASS_NAME, "item")  # Retrieve all products
            
            new_items_size = len(products)
            # Break the loop if no new items were loaded
            if old_items_size == new_items_size:
                break
            old_items_size = new_items_size

        # Scrape product information
        for product in products:
            try:
                name = product.find_element(By.CLASS_NAME, "product-name.product-name-font").text
                price = product.find_element(By.CLASS_NAME, "sale.sale-font").text
                views = product.find_element(By.CLASS_NAME, "product_view").text
                link_element = product.find_element(By.CLASS_NAME, "product-item-link")
                link = link_element.get_attribute('href') if link_element else None

                # Remove currency symbols and commas, and convert price to float for sorting
                price_numeric = float(re.sub(r"[^\d.]", "", price))
                
                products_list.append(name)
                prices_list.append(price)
                views_list.append(views)
                links_list.append(link)
                
                # Append the data as a dictionary into notebook_data list
                notebook_data.append({
                    'Product': name,
                    'Price': price_numeric,  # Store the numeric price for sorting
                    'View': views,
                    'Link': link
                })

                # Save the notebook information into Neo4j
                save_notebook_to_neo4j(name, price_numeric, views, link)
                
            except Exception as e:
                print(f"Error occurred: {e}")
                continue

        driver.close()

        # Sort the notebook_data by price
        notebook_data_sorted = sorted(notebook_data, key=lambda x: x['Price'])

        # Return the sorted data as JSON
        return jsonify(notebook_data_sorted)

# Route for manual trigger of data scraping (optional)
@app.route('/scrape', methods=['GET'])
def manual_scrape():
    scrape_via_api()
    return jsonify({"message": "Scraping manually triggered!"})

# Start the Flask server
if __name__ == '__main__':
    # Start the Flask server
    app.run(port=7777)

    try:
        # Keep the app running
        while True:
            time.sleep(2)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()