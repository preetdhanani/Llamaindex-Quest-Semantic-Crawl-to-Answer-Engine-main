import concurrent.futures
import requests,os
from bs4 import BeautifulSoup
import json
import traceback
import logging
from urllib.parse import urljoin
import concurrent
import pickle

if not os.path.exists("static/"):
    os.makedirs("static/")
# Set up logging to a file
logging.basicConfig(
    filename='static/crawler_logs.log',  # Name of the log file
    level=logging.INFO,           # Log level (INFO, WARNING, ERROR, etc.)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S'   # Date format
)



class web_crawler:
    def __init__(self, start_url, max_depth=1, max_retries=1):
        self.start_url = start_url
        self.max_depth = max_depth
        self.max_retries = max_retries
        self.visited = set()
        self.all_text = []
        self.url_name = self.start_url.split("/")[2]
        self.successful_urls_file = f"static/{self.url_name}_successful_urls.txt"
        self.crawled_data_file = f"static/{self.url_name}_crawled_data.json"
        self.error_log_file = f"static/{self.url_name}_error_log.txt"
        
        


        
    # def get_page_content(self, url):
    #     for attempt in range(self.max_retries):
    #         try:
    #             logging.info(f"Processing initiated")
    #             response = requests.get(url, timeout=30)
    #             response.raise_for_status()
    #             soup = BeautifulSoup(response.content, 'html.parser')
    #             text_content = soup.get_text()
    #             text_content = " ".join(line.strip() for line in text_content.split("\n") if line.strip())

    #             with open(self.successful_urls_file, 'a') as f:
    #                 f.write(url + '\n')
    #             with open(self.crawled_data_file, 'a') as f:
    #                 json.dump({"url": url, "text": text_content}, f)
                
    #             logging.info(f"Successfully processed: {url}")


    #             return str(soup), text_content
            
    #         except Exception as e:
    #             logging.warning(f"Error processing{attempt +1}for url: {url}: {str(e)}")

    def get_page_content(self, url):
        for attempt in range(self.max_retries):
            try:
                logging.info(f"Processing initiated for URL: {url}")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                text_content = soup.get_text()
                text_content = " ".join(line.strip() for line in text_content.split("\n") if line.strip())

                with open(self.successful_urls_file, 'a') as f:
                    f.write(url + '\n')
                with open(self.crawled_data_file, 'a') as f:
                    json.dump({"url": url, "text": text_content}, f)
                
                logging.info(f"Successfully processed: {url}")

                return str(soup), text_content
            
            except Exception as e:
                logging.warning(f"Error processing attempt {attempt + 1} for URL {url}: {str(e)}")

        return None, None  # Explicitly return None after all retries fail


    def parse_links(self, url, html_content):
        links=[]
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            for a_tag in soup.find_all('a', href=True)  :
                href = a_tag['href']
                full_url = urljoin(url, href)
                links.append(full_url)          

        except Exception as e:
            logging.error(f"Error parsing links for {url}: {str(e)}")
          
        return links
        

    def recursive_crawl(self, url, depth=0):
             
        if depth > self.max_depth or url in self.visited:
            return
            
        self.visited.add(url)
        html_content, text_content = self.get_page_content(url)

        if html_content and text_content:
            self.all_text.append((text_content, url))
            links = self.parse_links(url, html_content)

            with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
                futures = [executor.submit(self.recursive_crawl, link, depth +1 ) for link in links]

    def crawl(self):
        logging.info(f"Starting crawl from: {self.start_url}")
        if not os.path.exists(f"static/{self.url_name}_successful_urls.txt"):
            self.recursive_crawl(self.start_url)
        else:
            print("Crawling process completed. Exiting.")
        logging.info("Crawling process completed.")
        if not os.path.exists(f"static/{self.url_name}_all_text.pkl"):
            with open(f"static/{self.url_name}_all_text.pkl", "wb") as f:
                pickle.dump(self.all_text, f)
        else:
            self.all_text = pickle.load(open(f"static/{self.url_name}_all_text.pkl", "rb"))
            print("All text variable loaded")

    def save_all_text(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for text, url in self.all_text:
                f.write(f"URL: {url}\n")
                f.write(f"{text}\n")
                f.write("\n" + "="*50 + "\n\n")
        logging.info(f"All text content has been saved to {filename}")

