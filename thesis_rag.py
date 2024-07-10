import getpass
import os

from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader


os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")


loader = PyPDFLoader("./DSWW_Thesis_Final.pdf")
docs = loader.load_and_split()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

question = "What is distributional shift?"

output = rag_chain_with_source.invoke(question)

print(f"Question:\n{output['question']}\n\n")

print(f"Answer:\n{output['answer']}\n\n")


context = output["context"]
print("\n\nContext:\n")
for doc in context:
    page_no = doc.metadata["page"]
    source = doc.metadata["source"]
    content = doc.page_content

    print(f"Page {page_no} from {source}:\n{content}\n\n")
