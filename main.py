import openai

openai.api_key = '****'

with open('document.txt', 'r') as file:
    document_text = file.read()

response = openai.Embedding.create(
  input = document_text,
  model = 'text-embedding-ada-002'
)

vector = response['data'][0]['embedding']

print(vector)
