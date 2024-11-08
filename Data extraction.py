import requests
from bs4 import BeautifulSoup
import re


urls = ["https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/"]


text_data = ""

for url in urls:
   
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
   
    text = soup.get_text()
    text = re.sub(r"\[.*?\]", "", text)  
    
    
    text = text.strip()
    text_data += text + "\n\n"


with open("tutorial_text.txt", "w", encoding="utf-8") as file:
    file.write(text_data)

print("HTML data extracted and saved as tutorial_text.txt.")