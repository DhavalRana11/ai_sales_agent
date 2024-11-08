
import os
from langchain.chains import RetrievalQA
import langchain_openai
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma


# ADD YOUR API KEY Accordingly
# os.environ['OPENAI_API_KEY'] = 'YOUR API KEY'


llm = OpenAI(temperature=0.9, verbose=True)



loader = TextLoader("tutorial_text.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)


embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)


qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())


query = "What is CIMA?"
response = qa.invoke(query)
print(response['result'])