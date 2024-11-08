
import langchain
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import pipeline
import transformers


loader = TextLoader("tutorial_text.txt")
documents = loader.load()

r_split = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap=0,
    separator="\n\n",
    length_function=len
)
text = r_split.split_documents(documents)

embedding = HuggingFaceEmbeddings()
vectordb = Chroma.from_documents(text, embedding=embedding)


model_name = 'declare-lab/flan-alpaca-base'
generate_text = transformers.pipeline(
    model=model_name,
    task='text2text-generation',
    max_length=1100, 
    repetition_penalty=1.1 
)
llm = HuggingFacePipeline(pipeline=generate_text)


memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)

qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectordb.as_retriever(),memory=memory)


def loop():
    global response
    x = True
    while x==True:
        prompt = input('Input your prompt here: ')
        if prompt=='exit':
            return x==False
        else:
           
            response = qa.invoke(prompt)
            print("AI Said: ")
          
            print(response['answer'])

    print("You have exited")   

loop()


