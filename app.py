import os

import gradio as gr
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Custom ensemble retriever with LLM rephrasing and document filtering
class CustomEnsembleRetriever(EnsembleRetriever):
    def invoke(self, query: str, *args, **kwargs) -> list[Document]:
        """
        Rephrase the query using LLM call and judge the documents returned by the superclass 
        EnsembleRetriever using judge_documents()
        """
        documents = super().invoke(query, *args, **kwargs)

        # Rephrase if applicable
        print("Original question:", query)
        if rephrase:
            rephrased_query = llm.invoke(rephrase_template.format(query=query), {"temperature": 0}).content
            print("Rephrased question:", rephrased_query)
            documents += super().invoke(rephrased_query, *args, **kwargs)

        return self.judge_documents(query, documents)

    def judge_documents(self, query: str, documents: list[Document]) -> list[Document]:
        """
        Filter documents by relevance using LLM call
        """
        if judge:
            docs_str = ""
            for index, doc in enumerate(documents):
                docs_str += f"\n{index}. {doc}"

            filtered_doc_nums = llm.invoke(judge_template.format(query=query, docs_to_judge=docs_str), {"temperature": 0}).content.split()

            if not filtered_doc_nums or filtered_doc_nums[0] == "0":
                documents = [Document(page_content="No documents found!")]
            else:
                temp = list(documents)
                documents = []
                for num in filtered_doc_nums:
                    documents.append(temp[int(num)-1])

        return documents

# Prompts
system_prompt = """<|start_header_id|>user<|end_header_id|>
You are an assistant for discussing Wings of Fire using the provided context.
Your response should be under 250 tokens.
You are given the extracted parts of a long document and a question. Anwser the question as thoroughly as possible with the tone and form of an objective analytical essay.
You must use only the provided context to answer the question. Do not make up an answer.
WHEN ANSWERING THE QUESTION, DO NOT MENTION THE CONTEXT.
If the user is asking a question and there are no relevant documents, say that you don't know.
If the user is not asking a question, you may discuss Wings of Fire with them only.
You can only discuss Wings of Fire. If the user is not talking about Wings of Fire, inform them that you can only discuss Wings of Fire and suggest potential Wings of Fire related questions that they can ask instead.
Question: {question}
Context: {context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

rephrase_template = """Rephrase this query to be more easily searchable in a google search. Do not add things to the query. Just rephrase the query to be clearer and simpler.
Do not respond with ANYTHING EXCEPT THE REPHRASED QUERY.

Example query: whats morrowseer like and what does he want
Response: What is Morrowseer's personality and what are his motivations?

Query to process: {query}
"""

judge_template = """Provide the numbers of the documents that are EXTREMELY LIKELY to be relevant to the given query. Seperate the numbers by spaces.
Do not respond with ANYTHING EXCEPT THE NUMBERS.
If there are no relevant documents, then respond with a 0.
If there is an exact duplicate of a document, only return the number of one of them.

Example query: What is Morrowseer's personality and what are his motivations?
Example documents:
1. Morrowseer is a NightWing antagonist in the book series Wings of Fire
2. the NightWings plotted to take over the rainforest
3. charming and charismatic. darkstalker's traits allowed him to make friends and allies easily
4. The NightWings created a false prophecy in order to help them take over the rainforest
5. in the ancient days, NightWings were known to be wise, spreading knowledge across the continent
6. Morrowseer was scheming, as he was involved in creating the false prophecy
Response: 1 2 4 6

Query to process: {query}
Documents: {docs_to_judge}
"""

# Load data from chromadb
print("Loading data from chromadb...")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.environ['GEMINI'])
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chromadb")

# Instantiate model
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.7,
    api_key=os.environ['GROQ'],
    model_kwargs={"top_p": 0.65}
)

# Get documents and instantiate BM25Retriever
docs = vectorstore.get()["documents"]
bm25_retriever = BM25Retriever.from_texts(docs)

# Generate chatbot response based on user question
def chatbot_response(question, history, prompt_template, bm25_k, vs_k, _rephrase, _judge):
    global judge
    judge = _judge
    global rephrase
    rephrase = _rephrase

    # Set k values and instantiate EnsembleRetriever
    bm25_retriever.k = bm25_k
    vs_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": vs_k})
    retriever = CustomEnsembleRetriever(retrievers=[bm25_retriever, vs_retriever], weights=[0.5, 0.5])  

    # Prompt template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    # Instantiate and invoke retriever and chain
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type_kwargs={"prompt": prompt}, return_source_documents=True)
    response = qa_chain.invoke({"query": question})

    # Print debug
    print("Response:", response["result"], "\n\n")
    print("Rephrase?", rephrase, "\nJudge?", judge)
    for index, document in enumerate(response["source_documents"]):
        try:
            print(f'*{str(index+1)}. {document.metadata["source"]}*')
        except:
            print(f'*(metadata not found)*')
        print(f'Quote: "{document.page_content}"\n\n')

    return response["result"]

# Instantiate and start the demo
print("Starting gradio...")
demo = gr.ChatInterface(
    chatbot_response,
    title="🐲 WoF RAG Q&A Bot",
    description="A Llama3 8b Q&A bot powered by Groq, using RAG (Retrieval Augmented Generation) on documents from the Wings of Fire wiki. It utilizes LLMs to rephrase the user's query and judge and filter retrieved documents for relevance. Note that this is just a demo; the bot knows a decent amount but is still prone to hallucination or saying that it doesn't know. It performs best with Q&A and analyzing canon characters or events. If responses are unsatisfactory, try tweaking the values in the additional inputs section at the bottom.",
    additional_inputs=[
        gr.Textbox(value=system_prompt, label="System message"),
        gr.Slider(minimum=0, maximum=3, value=2, step=1, label="Number of documents to retrieve for bm25 "),
        gr.Slider(minimum=0, maximum=3, value=2, step=1, label="Number of documents to retrieve for vectorstore similarity"),
        gr.Checkbox(label="Rephrase query?", value=True),
        gr.Checkbox(label="Judge returned documents?", value=True),
    ],
    examples=[
        ["What is Wings of Fire"], 
        ["What are the implications of the dragonet prophecy"], 
        ["What are the motivations of Queen Scarlet"], 
        ["Write an essay about the role Qibli plays in Wings of Fire"], 
        ["Who is Foxglove"]
    ],
    cache_examples=False,
)
demo.launch(show_api=False)