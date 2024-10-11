from flask import Flask, request
from linebot import LineBotApi, WebhookHandler
from linebot.models import TextSendMessage, QuickReply, QuickReplyButton, MessageAction
import json
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import faiss
import re
import random
import numpy as np
from linebot import LineBotApi
from linebot.models import ImageSendMessage

# Neo4j configuration
URI = "neo4j://localhost"
AUTH = ("neo4j", "12345678")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
app = Flask(__name__)

def run_query(query, parameters=None):
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]
    driver.close()


cypher_query = '''
MATCH (n) WHERE (n:Greeting OR n:Questionadv) RETURN n.name as name, n.msg_reply as reply;
'''
greeting_corpus = []
greeting_vec = None
results = run_query(cypher_query)
for record in results:
    greeting_corpus.append(record['name'])
    #greeting_corpus = ["สวัสดีครับ","ดีจ้า"]
greeting_corpus = list(set(greeting_corpus))
#print(greeting_corpus)  

notebook_cypher_query = '''
MATCH (n:Notebook) RETURN n.name as name, n.price as price, n.link as link;
'''
notebook_corpus = []
notebook_vec = None
notebook_results = run_query(notebook_cypher_query)
for record in notebook_results:
    notebook_corpus.append((record['name']))  # Store as tuple
notebook_corpus = list(set(notebook_corpus))
print(notebook_corpus)


def compute_similar(corpus, sentence):
    a_vec = model.encode([corpus],convert_to_tensor=True,normalize_embeddings=True)
    b_vec = model.encode([sentence],convert_to_tensor=True,normalize_embeddings=True)
    similarities = util.cos_sim(a_vec, b_vec)
    return similarities

def neo4j_search(neo_query):
    results = run_query(neo_query)
    # Print results
    for record in results:
        response_msg = record['reply']
    return response_msg

def neo4j_searchnb(neo_query):
    results = run_query(neo_query)
    # Print results
    for record in results:
        response_msg = record['reply']
    return response_msg       


# Function to build FAISS index
def build_faiss_index(corpus):
    # Encode the corpus using the SentenceTransformer model
    corpus_embeddings = model.encode(corpus, convert_to_tensor=False, normalize_embeddings=True)
    # Initialize FAISS index (Flat index for L2 similarity)
    index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
    # Add embeddings to the index
    index.add(corpus_embeddings)
    return index, corpus_embeddings

# Function to compute similar sentences using FAISS
def compute_similar_faiss(corpus_index, sentence, k=1):
    ask_vec = model.encode([sentence], convert_to_tensor=False, normalize_embeddings=True)
    distances, indices = corpus_index.search(ask_vec, k)
    return distances[0], indices[0]

# Save chat history in Neo4j with a single node "Chat"
def save_chat_history(user_id, message, response):
    query = """
    MERGE (u:User {id: $user_id})
    CREATE (u)-[:HAS_CHAT]->(c:Chat {user_message: $message, bot_response: $response, timestamp: datetime()})
    """
    parameters = {
        'user_id': user_id,
        'message': message,
        'response': response
    }
    run_query(query, parameters)

# Building FAISS index for the greeting corpus
greeting_index, greeting_embeddings = build_faiss_index(greeting_corpus)
notebook_index, notebook_embeddings = build_faiss_index(notebook_corpus)

def compute_response(sentence, user_id): 
    # First check for greeting similarity
    distances_greeting, indices_greeting = compute_similar_faiss(greeting_index, sentence)
    
    if distances_greeting[0] < 0.4:  # FAISS uses L2 distances, so lower is better
        Match_greeting = greeting_corpus[indices_greeting[0]]
        My_cypher = f"MATCH (n) WHERE (n:Greeting OR n:Questionadv) AND n.name = '{Match_greeting}' RETURN n.msg_reply AS reply"
        my_msg = neo4j_search(My_cypher)
        
    
    # Now check for Notebook match if greeting didn't result in a response
    elif "Notebook" in sentence:  # Modify keywords as needed
        distances_notebook, indices_notebook = compute_similar_faiss(notebook_index, sentence)
        if distances_notebook[0] < 0.6:  # Only proceed if the distance is valid
            Match_notebook = notebook_corpus[indices_notebook[0]]
            My_cypher = f"MATCH (n:Notebook) WHERE n.name = '{Match_notebook}' RETURN n.link AS reply"
            my_msg = neo4j_searchnb(My_cypher)
            print(my_msg)

    else:
     # Check if the sentence is a legal-related question
        legal_keywords = ["โน๊ตบุ๊ค", "ร้าน", "คอม", "Advice", "แอดไวซ์"]
        if all(keyword not in sentence for keyword in legal_keywords):
            # Respond with a message that legal questions cannot be answered
            my_msg = "ขออภัยด้วยครับ แชทบอทตัวนี้สามารถตอบคำถามได้แค่ในส่วนที่เกี่ยวกับการแนะนำโน๊ตบุ๊คท่านั้น กรุณาถามคำถามที่เฉพาะเจาะจงมากกว่านี้หรือลองเปลี่ยนคำถามดูครับ"
        else:
            OLLAMA_API_URL = "http://localhost:11434/api/generate"
            headers = {
                "Content-Type": "application/json"
            }

            # Prepare the request payload for the TinyLLaMA model
            payload = {
                "model": "supachai/llama-3-typhoon-v1.5",
                "prompt": sentence + "ตอบสั้นๆไม่เกิน 30 คำ",
                "stream": False
            }

            # Send the POST request to the Ollama API
            response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(payload))

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the response JSON
                response_data = response.text
                data = json.loads(response_data)
                my_msg = data["response"] # Extract the response from the API
            else:
                # If Ollama API fails, return a default error message
                my_msg = "เกิดข้อผิดพลาดทาง API ลองพิมพ์ใหม่ดูนะครับ"

    return my_msg

def get_notebook_models():
    # Query to fetch the notebooks with the highest view count, limited to 10
    query = """
    MATCH (n:Notebook)
    RETURN n.name, n.price, n.view
    ORDER BY n.price
    LIMIT 10
    """
    result = run_query(query)
    # Format the notebook list to show name and price
    notebook_list = [f"{record['n.name']}: {record['n.price']} ยอดการชม: {record['n.view']}" for record in result]
    return "\n".join(notebook_list)

def get_notebook_models_by_price_range(min_price, max_price):
    # Query to fetch the notebooks within the specified price range
    query = """
    MATCH (n:Notebook)
    WHERE n.price >= $min_price AND n.price <= $max_price
    RETURN n.name, n.price, n.view
    ORDER BY n.price
    LIMIT 5
    """
    # Set the parameters for the query
    params = {
        "min_price": min_price,
        "max_price": max_price
    }
    
    # Run the query with the parameters
    result = run_query(query, params)
    
    # Format the notebook list to show name, price, and views
    notebook_list = [f"{record['n.name']}: {record['n.price']} บาท ยอดการชม: {record['n.view']}" for record in result]
    numbered_notebook_list = [f"{i+1}. {notebook}" for i, notebook in enumerate(notebook_list)]
    return "\n".join(numbered_notebook_list)

def scrape_promotion():
    url = "https://www.advice.co.th/activity-promotion?page=1"
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    driver.implicitly_wait(15)  # Adjusted wait time

    wait = WebDriverWait(driver, 15)
    wait.until(EC.visibility_of_element_located((By.CLASS_NAME, "col-xs-12.col-sm-4")))

    time.sleep(5)  # Add some delay to allow content to load completely

    products = driver.find_elements(By.CLASS_NAME, "list-items.list-items-no-action")
    promotion_data = []
    for product in products[:5]:  # Get only the top 5 products for simplicity
        try:
            title = product.find_element(By.CLASS_NAME, "entry-date").text
            desc_element = product.find_element(By.CLASS_NAME, "name-news")
            desc = desc_element.text
            link = desc_element.find_element(By.TAG_NAME, 'a').get_attribute('href')  # Get the link
            promotion_data.append(f"{title}: {desc} - รายละเอียดเพิ่มเติม: {link}")
        except Exception as e:
            print(f"Error: {e}")  # Print the error for debugging
            continue
    driver.close()

    return "\n".join(promotion_data)

def scrape_notebook_spec(model_name):
    # Query Neo4j to get the price for the notebook
    query = """
    MATCH (n:Notebook {name: $model_name})
    RETURN n.price
    """
    result = run_query(query, parameters={"model_name": model_name})
    
    # If the result exists, get the price
    if result and len(result) > 0:
        price = result[0]['n.price']
    else:
        price = "ไม่พบราคาในฐานข้อมูล"
    
    # Continue scraping the notebook specifications from the website
    response_msg = compute_response(model_name, "system")  # "system" user_id is used for internal processing
    url = str(response_msg)
    
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    driver.implicitly_wait(25)  # Adjusted wait time

    wait = WebDriverWait(driver, 15)
    wait.until(EC.visibility_of_element_located((By.CLASS_NAME, "sub-description.all-fontstyle")))
    
    # Fetch the image element
    img_url = None
    try:
        img_element = driver.find_element(By.CSS_SELECTOR, "img.img-fluid.detail-img.xzoom")
        img_url = img_element.get_attribute("src")
    except Exception as e:
        print(f"Error fetching image: {e}")

    # Fetch the notebook specifications
    products = driver.find_elements(By.CLASS_NAME, "sub-description.all-fontstyle")
    notebook_data = []

    for product in products:
        try:
            spec = product.text
            notebook_data.append(f"{spec}")
        except Exception as e:
            print(f"Error extracting specification: {e}")
            continue
    
    driver.close()

    # Return image URL, price from Neo4j, and notebook specs together
    return img_url, price, "\n".join(notebook_data)

selected_notebook = None

def random_fromneo():
    global selected_notebook
    # Query to fetch 5 random notebooks
    query = """
    MATCH (n:Notebook)
    WITH n, rand() AS r
    RETURN n.name, n.price, n.view
    ORDER BY r
    LIMIT 5 
    """
    result = run_query(query)
    if result:
        notebooks_info = []
        selected_notebook = []  # Clear the previous selected notebooks
        for i, record in enumerate(result):
            notebook_info = f"{i + 1}. {record['n.name']}: {record['n.price']} ยอดการชม: {record['n.view']}"
            notebooks_info.append(notebook_info)
            selected_notebook.append(record['n.name'])  # Store each notebook name
        return "\n".join(notebooks_info)  # Return all notebooks info as a single string
    else:
        return "ไม่พบข้อมูลโน๊ตบุ๊ค"

# Functions to define quick replies
def quick_reply_menu():
    quick_reply_buttons = [
        QuickReplyButton(action=MessageAction(label="รุ่นโน๊ตบุ๊ค", text="รุ่นโน๊ตบุ๊คแนะนำ")),
        QuickReplyButton(action=MessageAction(label="โปรโมชัน", text="โปรโมชันมีอะไรบ้าง")),
        QuickReplyButton(action=MessageAction(label="สุ่มโน๊ตบุ๊ค", text="สุ่มโน๊ตบุ๊คให้หน่อย")),
        QuickReplyButton(action=MessageAction(label="เกี่ยวกับ", text="เกี่ยวกับ"))
    ]
    quick_reply = QuickReply(items=quick_reply_buttons)
    return quick_reply

def quick_reply_random():
    quick_reply_buttons_nb = []
    
    # Create quick reply buttons with numbers as labels
    if selected_notebook:
        # Create quick reply buttons with numbers as labels
        for i, notebook_name in enumerate(selected_notebook):
            quick_reply_buttons_nb.append(
                QuickReplyButton(action=MessageAction(label=str(i + 1), text=notebook_name))
            )

        quick_reply_buttons_nb.append(QuickReplyButton(action=MessageAction(label="สุ่ม", text="สุ่มโน๊ตบุ๊คให้หน่อย")))
        quick_reply_buttons_nb.append(QuickReplyButton(action=MessageAction(label="เกี่ยวกับ", text="เกี่ยวกับ")))
        quick_reply_buttons_nb.append(QuickReplyButton(action=MessageAction(label="ย้อนกลับ", text="ย้อนกลับ")))

    else :
        quick_reply_buttons_nb.append(QuickReplyButton(action=MessageAction(label="กดสุ่มตรงนี้หรือพิมพ์", text="สุ่มโน๊ตบุ๊คให้หน่อย")))
        quick_reply_buttons_nb.append(QuickReplyButton(action=MessageAction(label="ถึงจะเลือกหมายเลขได้ครับ", text="สุ่มโน๊ตบุ๊คให้หน่อย")))
        quick_reply_buttons_nb.append(QuickReplyButton(action=MessageAction(label="ย้อนกลับ", text="ย้อนกลับ")))
    quick_reply = QuickReply(items=quick_reply_buttons_nb)
    return quick_reply


# Linebot handler
@app.route("/", methods=['POST'])
def linebot():
    body = request.get_data(as_text=True)
    try:
        json_data = json.loads(body)
        access_token = ''
        secret = ''
        line_bot_api = LineBotApi(access_token)
        handler = WebhookHandler(secret)
        signature = request.headers['X-Line-Signature']
        handler.handle(body, signature)
        
        event = json_data['events'][0]
        tk = event['replyToken']
        user_id = event['source']['userId']  # Get the user ID for history

        # Quick reply and messages
        if event['type'] == 'message':
            msg = event['message']['text']

            if msg == "เกี่ยวกับ":
                response_msg = "เราเป็นแชทบอทที่ไว้คอยแนะนำเกี่ยวกับโน๊ตบุ๊คในร้าน Advice \nโดยจะบอกข้อมูลเกี่ยวกับสเปคและราคา สามารถกดปุ่มด้านล่างเพื่อเช็คข้อมูลได้เลย"
                quick_reply = quick_reply_menu()
                line_bot_api.reply_message(tk, TextSendMessage(text=response_msg, quick_reply=quick_reply))

            elif msg == "รุ่นโน๊ตบุ๊คแนะนำ":
                # Use get_notebook_models to fetch notebook data from Neo4j
                notebook_info = get_notebook_models()
                response_msg = f"แนะนำโน๊ตบุ๊คโดยเรียงราคาจากต่ำที่สุดไปสูง: \n{notebook_info}"
                quick_reply = quick_reply_menu()
                line_bot_api.reply_message(tk, TextSendMessage(text=response_msg, quick_reply=quick_reply))

            elif "Notebook" in msg:  
                # Scrape notebook specs and image URL based on model name
                img_url, price, spec_info = scrape_notebook_spec(msg)  

                # Construct the response message for specs
                response_msg = f"สเปคของ {msg}:\nราคา: {price}\n{spec_info}"

                # Quick reply menu
                quick_reply = quick_reply_menu()

                # Prepare the image and spec messages
                if img_url:
                    image_message = ImageSendMessage(
                        original_content_url=img_url,
                        preview_image_url=img_url
                    )
                    text_message = TextSendMessage(text=response_msg, quick_reply=quick_reply)

                    # Send both the image and text messages together
                    line_bot_api.reply_message(tk, [image_message, text_message])
                else:
                    # If no image URL is found, send only the text message
                    line_bot_api.reply_message(tk, TextSendMessage(text=response_msg, quick_reply=quick_reply))
            
            elif "ไม่เกิน" in msg:
                # ใช้ regex เพื่อค้นหาตัวเลขในข้อความ
                price_match = re.search(r'(\d+)', msg)
                
                if price_match:
                    # แปลงตัวเลขที่พบจากข้อความให้เป็นจำนวนเต็ม
                    max_price = int(price_match.group(1))
                    
                    # เรียกฟังก์ชันเพื่อดึงโน๊ตบุ๊คในช่วงราคา
                    notebooks = get_notebook_models_by_price_range(0, max_price)
                    
                    # ตอบกลับด้วยรายการโน๊ตบุ๊ค
                    response_msg = f"นี่คือโน๊ตบุ๊คที่อยู่ในช่วงราคาที่กล่าวมาครับ: \n{notebooks}\nคุณสนใจรุ่นไหนเป็นพิเศษมั้ยครับ"
                    quick_reply = quick_reply_random()
                    line_bot_api.reply_message(tk, TextSendMessage(text=response_msg, quick_reply=quick_reply))
                    
                else:
                    # หากไม่พบตัวเลขให้ตอบกลับด้วยข้อความเตือน
                    response_msg =f"กรุณาระบุราคาที่ต้องการให้ชัดเจน"
                    quick_reply = quick_reply_menu()
                    line_bot_api.reply_message(tk, TextSendMessage(text=response_msg, quick_reply=quick_reply))

            elif "ในช่วง" in msg:
                # ใช้ regex เพื่อค้นหาตัวเลขสองจำนวนจากข้อความ
                price_match = re.findall(r'(\d+)', msg)
                
                if price_match and len(price_match) == 2:
                    # แปลงตัวเลขที่พบจากข้อความให้เป็นจำนวนเต็ม
                    min_price = int(price_match[0])
                    max_price = int(price_match[1])
                    
                    # เรียกฟังก์ชันเพื่อดึงโน๊ตบุ๊คในช่วงราคาที่กำหนด
                    notebooks = get_notebook_models_by_price_range(min_price, max_price)
                    
                    # ตอบกลับด้วยรายการโน๊ตบุ๊ค
                    response_msg = f"นี่คือโน๊ตบุ๊คที่อยู่ในช่วงราคาที่กล่าวมาครับ: \n{notebooks}\nคุณสนใจรุ่นไหนเป็นพิเศษมั้ยครับ"
                    quick_reply = quick_reply_random()
                    line_bot_api.reply_message(event['replyToken'], TextSendMessage(text=response_msg, quick_reply=quick_reply))

                else:
                    # หากไม่พบตัวเลขหรือจำนวนตัวเลขไม่ตรง ให้ตอบกลับด้วยข้อความเตือน
                    response_msg = "กรุณาระบุช่วงราคาที่ถูกต้อง"
                    quick_reply = quick_reply_menu()
                    line_bot_api.reply_message(event['replyToken'], TextSendMessage(text=response_msg, quick_reply=quick_reply))
                    

            elif msg == "โปรโมชันมีอะไรบ้าง":
                # Scrape promotion information
                promotion_info = scrape_promotion() 
                response_msg = f"โปรโมชันในช่วงนี้: \n{promotion_info}"
                quick_reply = quick_reply_menu()
                line_bot_api.reply_message(tk, TextSendMessage(text=response_msg, quick_reply=quick_reply))

            elif msg == "ย้อนกลับ":
                response_msg = f"สามารถกดปุ่มด้านล่างเพื่อเช็คข้อมูลได้เลย"
                quick_reply = quick_reply_menu()
                line_bot_api.reply_message(tk, TextSendMessage(text=response_msg, quick_reply=quick_reply))

            elif msg == "สุ่มโน๊ตบุ๊คให้หน่อย":
                # Scrape promotion information
                random_nb = random_fromneo()
                response_msg = f"นี่คือโน๊ตบุ๊คที่สุ่มออกมาได้ครับ: \n{random_nb}\nคุณสนใจรุ่นไหนเป็นพิเศษมั้ยครับ"
                quick_reply = quick_reply_random()
                line_bot_api.reply_message(tk, TextSendMessage(text=response_msg, quick_reply=quick_reply))
            
            else:
                quick_reply = quick_reply_menu()
                response_msg = compute_response(msg, user_id)
                line_bot_api.reply_message( tk, TextSendMessage(text=response_msg, quick_reply=quick_reply))

            save_chat_history(user_id, msg, response_msg)

    except Exception as e:
        print(e)
        print(body)
    return 'OK'

if __name__ == '__main__':
    app.run(port=5000)