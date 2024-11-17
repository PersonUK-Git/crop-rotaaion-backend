import google.generativeai as genai
import os

genai.configure(api_key='AIzaSyCC4m4NoNGu3QgHv7pXjz4QMnWXzOUsr7I')

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Write a story about a magic backpack.")
print(response.text)