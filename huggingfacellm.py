
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


loader = TextLoader("tutorial_text.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

llm = HuggingFacePipeline.from_model_id(model_id="google/flan-t5-base", task="text2text-generation", pipeline_kwargs={"max_length":1000})

directory='db'
embeddings = HuggingFaceEmbeddings()
docsearch = Chroma(persist_directory=directory, embedding_function=embeddings)


qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

query = "What is the cost of your data science course?"
response = qa.invoke(query)
print(response['result'])