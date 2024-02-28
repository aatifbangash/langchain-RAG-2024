from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from decouple import config
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

try:
        
    # llm = Ollama(model="llama2")
    # chat_model = ChatOllama(verbose = True)

    # text = "What would be a good company name for a company that makes colorful socks?"
    # # messages = [HumanMessage(content=text)]
    # # print(messages)

    # res = llm.invoke(text)
    # # res = chat_model.invoke(messages)
    # print(res)
    # chat_model.invoke(messages)
    # print(chat_model)
    SECRET_KEY = config('OPENAI_API_KEY')

    # load OpenAI LLM model (gpt-3.5)
    llm = OpenAI(openai_api_key=SECRET_KEY)
    # , model_name="gpt-3.5-turbo-instruct"

    # load website into document using Langchain loader (webloader)
    loader = WebBaseLoader("https://www.codenterprise.com/")
    docs = loader.load()

    # instantiate the OpenAIEmbedding - used to create embedding for the document to store in vectorDB
    embeddings = OpenAIEmbeddings()

    # text spliter instance
    text_splitter = RecursiveCharacterTextSplitter()

    # split document into chunks using TextSplitter
    documents = text_splitter.split_documents(docs)

    # Store documents (embedding form of splitted documents) into the local vector db. 
    # Embeddings (numeric representaion of text/paragraphs)
    vector = FAISS.from_documents(documents, embeddings)
    
    # Prepare the promt to fromat the question and its context (from vectorDB)
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
    <context>
    {context}
    </context>
    Question: {input}""")

    # create a chain for query
    query_chain = create_stuff_documents_chain(llm, prompt)

    # retriever = instance of vectorDB
    retriever = vector.as_retriever()

    # create a retrieval chain/instance using retriever object and query chain
    retrieval_chain = create_retrieval_chain(retriever, query_chain)

    # pass input/query to retrieval chain to search in the vectorDB to create the promt to be processed by LLM model
    response = retrieval_chain.invoke({"input": "What is the Role of Zahoor ur rahman at codenterprise?"})
    print(response["answer"])

    # # method 1
    # noInputPrompt = PromptTemplate(
    #     input_variables=[],
    #     template="What is Laravel? explain in one line."
    # )

    # # method 2
    # inputParam = PromptTemplate.from_template("What is {language}? explain it to me in one line.") # {language} is considered an input_variable here
    # formattedInputPrompt = inputParam.format(language="NodeJs")
    # print(inputParam);
    # response = llm.invoke(formattedInputPrompt)
    # print(response)


    # formattedNoInputPrompt = noInputPrompt.format()
    # response = llm.invoke(formattedNoInputPrompt)
    # print(response)

except Exception as e:
        print(e)
