# Install packages 
!pip install -U langchain langchain-community langchain-huggingface transformers sentence-transformers faiss-cpu

# Import packages
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

# Study notes 
with open("llm_notes.txt", "w") as f:
    f.write("""
      Agent is a class that uses an LLM to choose a sequence of actions to take.
      In Chains, a sequence of actions is hardcoded. 
      In Agents, a language model is used as a reasoning engine to determine which actions to take and in which order.
      Agents select and use Tools and Toolkits for actions.
      Chains are easily reusable components linked together.
      Chains encode a sequence of calls to components like models, document retrievers, other Chains, etc., and provide 
      a simple interface to this sequence.
      Memory maintains Chain state, incorporating context from past runs.
    """)

loader = TextLoader("llm_notes.txt")
docs = loader.load()

# Split into chunks
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# Create vector database
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(split_docs, embedding_model)

# Load model
hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Build Question and Answer system
feynman_bot = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever()
)

# Prompt
question = "Give me a thought-provoking quiz question based on my notes."
response = feynman_bot.invoke({"query": question})

print("Feynman Bot:", response["result"])

