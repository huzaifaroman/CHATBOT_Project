# # from PyPDF2 import PdfReader
# # from langchain.text_splitter import CharacterTextSplitter
# # from langchain_community.vectorstores import FAISS  # Updated import
# # from sentence_transformers import SentenceTransformer
# # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
# # import torch

# # # Load PDF and extract text
# # reader = PdfReader("gpt.pdf")
# # raw_text = ''
# # for i, page in enumerate(reader.pages):
# #     text = page.extract_text()
# #     if text:
# #         raw_text += text

# # # Split the text into chunks
# # text_splitter = CharacterTextSplitter(
# #     separator="\n",
# #     chunk_size=1000,
# #     chunk_overlap=200,
# #     length_function=len,
# # )
# # texts = text_splitter.split_text(raw_text)

# # # Use SentenceTransformer for embeddings (Hugging Face)
# # embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# # embeddings = embedding_model.encode(texts)

# # # Create FAISS index with embeddings (pass embeddings and texts separately)
# # docsearch = FAISS.from_embeddings(embeddings=embeddings, texts=texts)

# # # Load the model for Question Answering and Chat from Hugging Face
# # qa_model_name = "google/flan-t5-large"
# # chat_model_name = "google/flan-t5-large"

# # qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
# # qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

# # chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_name)
# # chat_model = AutoModelForSeq2SeqLM.from_pretrained(chat_model_name)

# # # Example question-answering process
# # def answer_pdf_query(question):
# #     docs = docsearch.similarity_search(question)  # Search for the most similar document
# #     context = ' '.join([doc.page_content for doc in docs])  # Merge relevant chunks

# #     inputs = qa_tokenizer(question, context, return_tensors='pt', truncation=True, max_length=512, padding=True)
    
# #     with torch.no_grad():
# #         outputs = qa_model(**inputs)
# #         answer_start = outputs.start_logits.argmax()
# #         answer_end = outputs.end_logits.argmax() + 1
        
# #         answer = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
        
# #     return answer

# # # Function for chat-based generation
# # def chat_response(question):
# #     inputs = chat_tokenizer(question, return_tensors='pt', truncation=True)
    
# #     outputs = chat_model.generate(**inputs, max_length=64)
# #     response = chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
# #     return response

# # # Example of querying a PDF
# # query = "Who are the authors of the article?"
# # answer = answer_pdf_query(query)
# # print(f"Answer: {answer}")

# # # Example of chatting
# # user_query = "Tell me about yourself"
# # chat_reply = chat_response(user_query)
# # print(f"Chatbot: {chat_reply}")



# # import os
# # import fitz
# # import threading
# # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
# # from langchain import LLMChain
# # from langchain.prompts import PromptTemplate
# # from langchain_community.llms import HuggingFaceHub
# # from langchain.memory import ConversationBufferMemory
# # import torch
# # import logging
# # import pytesseract
# # from pdf2image import convert_from_path



# # # Configure logging
# # logging.basicConfig(level=logging.DEBUG)

# # # Set Hugging Face API Token
# # os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

# # # Model names
# # chat_model_name = "google/flan-t5-large"
# # qa_model_name = "google/flan-t5-large"
# # # Load models
# # chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_name, trust_remote_code=True)
# # chat_model = AutoModelForSeq2SeqLM.from_pretrained(chat_model_name)
# # qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)

# # qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

# # # Prompt templates
# # chat_prompt_template = PromptTemplate(
# #     input_variables=["question"],
# #     template="Question: {question}\nAnswer: Let's think step by step."
# # )

# # qa_prompt_template = PromptTemplate(
# #     input_variables=["context", "question"],
# #     template="Context: {context}\nQuestion: {question}\nAnswer:"
# # )

# # # Memory for conversation
# # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# # # Global variable to store extracted PDF context
# # pdf_context_storage = {}

# # def create_chat_chain():
# #     try:
# #         return LLMChain(
# #             llm=HuggingFaceHub(
# #                 repo_id=chat_model_name,
# #                 model_kwargs={"temperature": 0, "max_length": 64}
# #             ),
# #             memory=memory,
# #             prompt=chat_prompt_template,
# #         )
# #     except Exception as e:
# #         logging.error(f"Error initializing Chat LLM Chain: {e}")
# #         exit(1)

# # def create_qa_chain():
# #     try:
# #         return LLMChain(
# #             llm=HuggingFaceHub(
# #                 repo_id=qa_model_name,
# #                 model_kwargs={"temperature": 0, "max_length": 512}
# #             ),
# #             memory=memory,
# #             prompt=qa_prompt_template,
# #         )
# #     except Exception as e:
# #         logging.error(f"Error initializing QA LLM Chain: {e}")
# #         exit(1)



# # def extract_pdf_context(pdf_path):
# #     try:
# #         document = fitz.open(pdf_path)
# #         text = ""
# #         for page in document:
# #             # Get text directly if available
# #             text += page.get_text()
        
       
# #         # if not text.strip():
# #         #     images = convert_from_path(pdf_path)
# #         #     for image in images:
# #         #         text += pytesseract.image_to_string(image)

# #         # document.close()
        
# #         # logging.debug(f"Extracted text: {text[:500]}...")  
# #         return text
# #     except Exception as e:
# #         logging.error(f"Error extracting PDF context: {e}")
# #         return ""

# # def handle_chat_query(user_input):
# #     chat_chain = create_chat_chain()
# #     try:
# #         response = chat_chain.run({"question": user_input})

# #         if isinstance(response, str) and response.strip():
# #             return {"response": response}
# #         else:
# #             return {"response": "Invalid response format from the chat model."}

# #     except Exception as e:
# #         logging.error(f"Error during chat generation: {e}")
# #         return {"response": "An error occurred while processing your request."}

# # def process_pdf_query(pdf_path, question):
# #     try:
# #         context = extract_pdf_context(pdf_path)
        
# #         logging.debug(f"Extracted context: {context}")

# #         if not context:
# #             return {"error": "Failed to extract context from PDF."}

# #         logging.debug(f"Question: {question}")
# #         logging.debug(f"Context Length: {len(context)}")

# #         # Tokenize input with truncation
# #         inputs = qa_tokenizer(question, context, return_tensors='pt', truncation=True, max_length=512, padding=True)
        
# #         with torch.no_grad():
# #             outputs = qa_model(**inputs)
# #             answer_start = outputs.start_logits.argmax()
# #             answer_end = outputs.end_logits.argmax() + 1
            
# #             logging.debug(f"Start index: {answer_start}, End index: {answer_end}")
            
# #             if answer_start < answer_end:  # Ensure valid indices
# #                 answer = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
# #             else:
# #                 answer = "Unable to determine the answer from the provided context."
                
# #         logging.debug(f"Answer: {answer}")
# #         return {"answer": answer}
# #     except Exception as e:
# #         logging.error(f"Error during PDF query processing: {e}")
# #         return {"error": str(e)}





# # if __name__ == "__main__":
# #     print("Welcome to the Chatbot! Type 'exit' to quit.")
# #     while True:
# #         user_input = input("User: ")
# #         if user_input.lower() == "exit":
# #             break
# #         elif user_input.lower().startswith("pdf:"):
# #             pdf_path = user_input.split(":")[1].strip()
# #             question = input("Question about the PDF: ")
# #             # Use a callback function to handle responses from the thread
# #             def pdf_callback(response):
# #                 if 'answer' in response:
# #                     print(f"Bot: {response['answer']}\n")
# #                 else:
# #                     print(f"Bot: {response['error']}\n")

# #             process_pdf_query(pdf_path, question)
# #             print("Processing your question about the PDF...")  # Inform the user that processing is ongoing
# #         else:
# #             response = handle_chat_query(user_input)
# #             print(f"Bot: {response['response']}\n")

# # import os
# # import fitz
# # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, AutoModel
# # from langchain import LLMChain
# # from langchain.prompts import PromptTemplate
# # from langchain_community.llms import HuggingFaceHub
# # from langchain.memory import ConversationBufferMemory
# # import torch
# # import logging
# # from langchain.text_splitter import CharacterTextSplitter
# # from langchain.vectorstores import FAISS
# # from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# # from langchain.docstore import InMemoryDocstore
# # from langchain.document_loaders import PyPDFLoader
# # from pdf2image import convert_from_path
# # import pytesseract


# # # Configure logging
# # logging.basicConfig(level=logging.DEBUG)

# # # Hugging Face API Token
# # os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

# # # Model and embedding details
# # chat_model_name = "google/flan-t5-large"
# # qa_model_name = "google/flan-t5-large"
# # embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# # # Load models
# # chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_name, trust_remote_code=True)
# # chat_model = AutoModelForSeq2SeqLM.from_pretrained(chat_model_name)
# # qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
# # qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

# # # Load embedding model
# # embedder = HuggingFaceEmbeddings(model_name=embedding_model_name)

# # # Text splitter for splitting PDF content
# # text_splitter = CharacterTextSplitter(
# #     separator="\n",
# #     chunk_size=300,
# #     chunk_overlap=200,
# #     length_function=len,
# # )

# # # Prompt templates
# # chat_prompt_template = PromptTemplate(
# #     input_variables=["question"],
# #     template="Question: {question}\nAnswer: Let's think step by step."
# # )

# # qa_prompt_template = PromptTemplate(
# #     input_variables=["context", "question"],
# #     template="Context: {context}\nQuestion: {question}\nAnswer:"
# # )

# # # Memory for conversation
# # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# # # Global variable to store vector store and document context
# # vector_store = None
# # pdf_context_storage = {}

# # def create_chat_chain():
# #     try:
# #         return LLMChain(
# #             llm=HuggingFaceHub(
# #                 repo_id=chat_model_name,
# #                 model_kwargs={"temperature": 0, "max_length": 64}
# #             ),
# #             memory=memory,
# #             prompt=chat_prompt_template,
# #         )
# #     except Exception as e:
# #         logging.error(f"Error initializing Chat LLM Chain: {e}")
# #         exit(1)

# # def create_qa_chain():
# #     try:
# #         return LLMChain(
# #             llm=HuggingFaceHub(
# #                 repo_id=qa_model_name,
# #                 model_kwargs={"temperature": 0, "max_length": 512}
# #             ),
# #             memory=memory,
# #             prompt=qa_prompt_template,
# #         )
# #     except Exception as e:
# #         logging.error(f"Error initializing QA LLM Chain: {e}")
# #         exit(1)

# # # PDF context extraction
# # def extract_pdf_context(pdf_path):
# #     try:
# #         document = fitz.open(pdf_path)
# #         text = ""
# #         for page in document:
# #             text += page.get_text()
        
# #         document.close()
        
# #         # Fallback to OCR if text extraction fails
# #         if not text.strip():
# #             images = convert_from_path(pdf_path)
# #             for image in images:
# #                 text += pytesseract.image_to_string(image)
                
# #         logging.debug(f"Extracted text: {text[:500]}...")  
# #         return text
       
# #     except Exception as e:
# #         logging.error(f"Error extracting PDF context: {e}")
# #         return ""

# # # Assuming you have a pre-trained embeddings model
# # embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # def create_vector_store(pdf_path):
# #     global vector_store

# #     # Load the PDF and split into documents
# #     loader = PyPDFLoader(pdf_path)
# #     docs = loader.load_and_split()

# #     # Debug the structure of docs to see what is returned
# #     logging.debug(f"Loaded documents: {docs}")

# #     # Convert documents to embeddings
# #     doc_embeddings = embeddings_model.embed_documents([doc.page_content for doc in docs if hasattr(doc, 'page_content')])

# #     # Create a mapping between document IDs and documents (for FAISS)
# #     docstore = InMemoryDocstore({i: doc for i, doc in enumerate(docs)})

# #     # Initialize FAISS with the embeddings and the document store
# #     vector_store = FAISS.from_documents(doc_embeddings, docstore)
# #     logging.debug("FAISS vector store created successfully.")

# # def process_pdf_query(pdf_path, question):
# #     global vector_store
# #     try:
# #         if vector_store is None:
# #             # Create a vector store if not already created
# #             create_vector_store(pdf_path)

# #         # Perform similarity search
# #         docs = vector_store.similarity_search(question)

# #         logging.debug(f"Docs: {docs}")

# #         # Check if we have any relevant documents
# #         if not docs:
# #             return {"error": "No relevant context found for your query."}

# #         # Run the QA chain with the retrieved documents and the question
# #         qa_chain = create_qa_chain()
# #         answer = qa_chain.run(input_documents=docs, question=question)

# #         logging.debug(f"Answer: {answer}")
# #         return {"answer": answer}

# #     except Exception as e:
# #         logging.error(f"Error during PDF query processing: {e}")
# #         return {"error": str(e)}

# # # Handle chat queries
# # def handle_chat_query(user_input):
# #     chat_chain = create_chat_chain()
# #     try:
# #         response = chat_chain.run({"question": user_input})

# #         if isinstance(response, str) and response.strip():
# #             return {"response": response}
# #         else:
# #             return {"response": "Invalid response format from the chat model."}

# #     except Exception as e:
# #         logging.error(f"Error during chat generation: {e}")
# #         return {"response": "An error occurred while processing your request."}
    

# # if __name__ == "__main__":
# #     print("Welcome to the Chatbot! Type 'exit' to quit.")
# #     while True:
# #         user_input = input("User: ")
        
# #         if user_input.lower() == "exit":
# #             print("Exiting the chatbot. Goodbye!")
# #             break
        
# #         # Check if user wants to ask a question about a PDF
# #         elif user_input.lower().startswith("pdf:"):
# #             pdf_path = user_input.split(":")[1].strip()
            
# #             if os.path.exists(pdf_path):
# #                 question = input("Question about the PDF: ")
                
# #                 # Process the PDF query and print the response
# #                 response = process_pdf_query(pdf_path, question)
                
# #                 if 'answer' in response:
# #                     print(f"Bot: {response['answer']}\n")
# #                 else:
# #                     print(f"Bot: {response['error']}\n")
# #             else:
# #                 print("Invalid PDF path. Please provide a valid file path.\n")

# #         else:
# #             # Handle general chat queries
# #             response = handle_chat_query(user_input)
# #             print(f"Bot: {response['response']}\n")


# # import os
# # import fitz
# # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering, AutoModel
# # from langchain import LLMChain
# # from langchain.prompts import PromptTemplate
# # from langchain_community.llms import HuggingFaceHub
# # from langchain.memory import ConversationBufferMemory
# # import torch
# # import logging
# # from langchain.text_splitter import CharacterTextSplitter
# # from langchain.vectorstores import FAISS
# # from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# # from langchain.docstore import InMemoryDocstore
# # from langchain.document_loaders import PyPDFLoader
# # from pdf2image import convert_from_path
# # import pytesseract
# # from langchain.docstore.document import Document
# # from sentence_transformers import SentenceTransformer




# # # Configure logging
# # logging.basicConfig(level=logging.DEBUG)

# # # Hugging Face API Token
# # os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

# # # Model and embedding details
# # chat_model_name = "google/flan-t5-large"
# # qa_model_name = "google/flan-t5-large"
# # embedding_model_name  = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

# # # Load models
# # chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_name, trust_remote_code=True)
# # chat_model = AutoModelForSeq2SeqLM.from_pretrained(chat_model_name)
# # qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
# # qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

# # # Load embedding model
# # embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model_name)


# # # Text splitter for splitting PDF content
# # text_splitter = CharacterTextSplitter(
# #     separator="\n",
# #     chunk_size=300,
# #     chunk_overlap=200,
# #     length_function=len,
# # )

# # # Prompt templates
# # chat_prompt_template = PromptTemplate(
# #     input_variables=["question"],
# #     template="Question: {question}\nAnswer: Let's think step by step."
# # )

# # qa_prompt_template = PromptTemplate(
# #     input_variables=["context", "question"],
# #     template="Context: {context}\nQuestion: {question}\nAnswer:"
# # )

# # # Memory for conversation
# # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# # # Global variable to store vector store and document context
# # vector_store = None
# # pdf_context_storage = {}

# # def create_chat_chain():
# #     try:
# #         return LLMChain(
# #             llm=HuggingFaceHub(
# #                 repo_id=chat_model_name,
# #                 model_kwargs={"temperature": 0, "max_length": 64}
# #             ),
# #             memory=memory,
# #             prompt=chat_prompt_template,
# #         )
# #     except Exception as e:
# #         logging.error(f"Error initializing Chat LLM Chain: {e}")
# #         exit(1)

# # def create_qa_chain():
# #     try:
# #         return LLMChain(
# #             llm=HuggingFaceHub(
# #                 repo_id=qa_model_name,
# #                 model_kwargs={"temperature": 0, "max_length": 512}
# #             ),
# #             memory=memory,
# #             prompt=qa_prompt_template,
# #         )
# #     except Exception as e:
# #         logging.error(f"Error initializing QA LLM Chain: {e}")
# #         exit(1)

# # # PDF context extraction
# # def extract_pdf_context(pdf_path):
# #     try:
# #         text_parts = []  # Initialize an empty list to hold parts of the extracted text
# #         images = convert_from_path(pdf_path)  # Convert PDF pages to images
        
# #         for image in images:
# #             # Extract text from each image using OCR
# #             text_parts.append(pytesseract.image_to_string(image, lang='eng'))  # Specify the language as needed

# #         # Join all the extracted text parts into a single string
# #         text = ''.join(text_parts)        
# #         logging.debug(f"Extracted text: {text[:500]}...")  # Log the first 500 characters of the extracted text
# #         return text
       
# #     except Exception as e:
# #         logging.error(f"Error extracting PDF context: {e}")
# #         return ""






# # def create_vector_store(pdf_path):
# #     global vector_store

# #     # Extract text from the PDF
# #     full_text = extract_pdf_context(pdf_path)
# #     if not full_text.strip():
# #         logging.error("No text extracted from the PDF.")
# #         return

# #     # Split the extracted text into chunks
# #     split_docs = text_splitter.split_text(full_text)
# #     logging.debug(f"Split documents: {split_docs}")

# #     # Create document objects with 'page_content' attribute
# #     documents = [Document(page_content=doc) for doc in split_docs]

# #     # Convert document texts to embeddings
# #     # Use embed_documents for batch embedding
# #     doc_embeddings = embeddings_model.embed_documents([doc.page_content for doc in documents])

# #     # Initialize FAISS with the embeddings and the document objects
# #     vector_store = FAISS.from_documents(documents, doc_embeddings)
# #     logging.debug("FAISS vector store created successfully.")

# # # Process the PDF query
# # def process_pdf_query(pdf_path, question):
# #     global vector_store
# #     try:
# #         if vector_store is None:
# #             # Create the vector store if not already created
# #             create_vector_store(pdf_path)

# #         # Perform similarity search
# #         docs = vector_store.similarity_search(question)

# #         if docs:
# #             # Combine the contents of the retrieved documents
# #             context = " ".join([doc.page_content for doc in docs if hasattr(doc, 'page_content')])
# #             logging.debug(f"Retrieved document context: {context[:500]}...")
# #         else:
# #             context = ""

# #         if not context:
# #             return {"error": "No relevant context found for your query."}

# #         # Answer the question using the QA model
# #         inputs = qa_tokenizer(question, context, return_tensors='pt', truncation=True, max_length=512, padding=True)

# #         with torch.no_grad():
# #             outputs = qa_model(**inputs)
# #             answer_start = outputs.start_logits.argmax()
# #             answer_end = outputs.end_logits.argmax() + 1

# #             if answer_start < answer_end:
# #                 answer = qa_tokenizer.convert_tokens_to_string(
# #                     qa_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end])
# #                 )
# #             else:
# #                 answer = "Unable to determine the answer from the provided context."

# #         logging.debug(f"Answer: {answer}")
# #         return {"answer": answer}

# #     except Exception as e:
# #         logging.error(f"Error during PDF query processing: {e}")
# #         return {"error": str(e)}
# #     # Handle chat queries
# # def handle_chat_query(user_input):
# #     chat_chain = create_chat_chain()
# #     try:
# #         response = chat_chain.run({"question": user_input})

# #         if isinstance(response, str) and response.strip():
# #             return {"response": response}
# #         else:
# #             return {"response": "Invalid response format from the chat model."}

# #     except Exception as e:
# #         logging.error(f"Error during chat generation: {e}")
# #         return {"response": "An error occurred while processing your request."}


# # if __name__ == "__main__":
    
    
# #     fun2 = extract_pdf_context(pdf_path='gpt.pdf')
# #     fun = create_vector_store(pdf_path='gpt.pdf')
# #     print(fun2)
# #     print(fun)
# #     # print("Welcome to the Chatbot! Type 'exit' to quit.")
# #     # while True:
# #     #     user_input = input("User: ")
        
# #     #     if user_input.lower() == "exit":
# #     #         print("Exiting the chatbot. Goodbye!")
# #     #         break
        
# #     #     # Check if user wants to ask a question about a PDF
# #     #     elif user_input.lower().startswith("pdf:"):
# #     #         pdf_path = user_input.split(":")[1].strip()
            
# #     #         if os.path.exists(pdf_path):
# #     #             question = input("Question about the PDF: ")
                
# #     #             # Process the PDF query and print the response
# #     #             response = process_pdf_query(pdf_path, question)
                
# #     #             if 'answer' in response:
# #     #                 print(f"Bot: {response['answer']}\n")
# #     #             else:
# #     #                 print(f"Bot: {response['error']}\n")
# #     #         else:
# #     #             print("Invalid PDF path. Please provide a valid file path.\n")

# #     #     else:
# #     #         # Handle general chat queries
# #     #         response = handle_chat_query(user_input)
# #     #         print(f"Bot: {response['response']}\n")










# import os
# import fitz
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSeq2SeqLM,
#     AutoModelForQuestionAnswering,
#     AutoModel,
# )
# from langchain import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain_community.llms import HuggingFaceHub
# from langchain.memory import ConversationBufferMemory
# import torch
# import logging
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain.docstore.document import Document
# from pdf2image import convert_from_path
# import pytesseract
# from langchain.embeddings import HuggingFaceEmbeddings
# import PyPDF2
# import torch.nn.functional as F

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)

# # Hugging Face API Token
# os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

# # Model and embedding details
# chat_model_name = "google/flan-t5-large"
# qa_model_name = "google/flan-t5-large"
# embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'

# # Load models
# chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_name, trust_remote_code=True)
# chat_model = AutoModelForSeq2SeqLM.from_pretrained(chat_model_name)
# qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
# qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

# # Load embedding model
# embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
# embedding_model = AutoModel.from_pretrained(embedding_model_name)

# # Text splitter for splitting PDF content
# text_splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_size=300,
#     chunk_overlap=200,
#     length_function=len,
# )

# # Prompt templates
# chat_prompt_template = PromptTemplate(
#     input_variables=["question"],
#     template="Question: {question}\nAnswer: Let's think step by step."
# )

# qa_prompt_template = PromptTemplate(
#     input_variables=["context", "question"],
#     template="Context: {context}\nQuestion: {question}\nAnswer:"
# )

# # Memory for conversation
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# # Global variable to store vector store and document context
# vector_store = None

# # Mean Pooling Function
# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output.last_hidden_state
#     input_mask_expanded = (
#         attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     )
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
#         input_mask_expanded.sum(1), min=1e-9
#     )

# # Function to get embeddings
# def get_embeddings(text_list):
#     encoded_input = embedding_tokenizer(
#         text_list, padding=True, truncation=True, return_tensors="pt"
#     )
#     with torch.no_grad():
#         model_output = embedding_model(**encoded_input)
#     return mean_pooling(model_output, encoded_input["attention_mask"])

# def create_chat_chain():
#     try:
#         return LLMChain(
#             llm=HuggingFaceHub(
#                 repo_id=chat_model_name,
#                 model_kwargs={"temperature": 0, "max_length": 64}
#             ),
#             memory=memory,
#             prompt=chat_prompt_template,
#         )
#     except Exception as e:
#         logging.error(f"Error initializing Chat LLM Chain: {e}")
#         exit(1)

# def create_qa_chain():
#     try:
#         return LLMChain(
#             llm=HuggingFaceHub(
#                 repo_id=qa_model_name,
#                 model_kwargs={"temperature": 0, "max_length": 512}
#             ),
#             memory=memory,
#             prompt=qa_prompt_template,
#         )
#     except Exception as e:
#         logging.error(f"Error initializing QA LLM Chain: {e}")
#         exit(1)

# def extract_pdf_context(pdf_path):
#     try:
#         text_parts = []

#         # Use PyPDF2 to extract text first
#         raw_text = ''
#         with open(pdf_path, 'rb') as file:
#             reader = PyPDF2.PdfReader(file)
#             for i, page in enumerate(reader.pages):
#                 text = page.extract_text()
#                 if text:
#                     raw_text += text
#                     logging.debug(f"Extracted text from page {i + 1}: {text[:100]}...")  # Log the first 100 characters from each page
#                 else:
#                     logging.debug(f"No text found on page {i + 1}.")

#         # If PyPDF2 extracted any text, append it to text_parts
#         if raw_text.strip():
#             text_parts.append(raw_text)
#             logging.debug(f"Total extracted text from PDF: {raw_text[:500]}...")  # Log the first 500 characters of the extracted text
#         else:
#             logging.debug("No text found using PyPDF2, proceeding with image extraction.")

#             # Convert PDF pages to images and use pytesseract for OCR
#             images = convert_from_path(pdf_path)
#             for i, image in enumerate(images):
#                 ocr_text = pytesseract.image_to_string(image, lang='eng')
#                 if ocr_text.strip():  # Check if any text was extracted from the image
#                     text_parts.append(ocr_text)
#                     logging.debug(f"Extracted text from image of page {i + 1}: {ocr_text[:100]}...")  # Log the first 100 characters from OCR
#                 else:
#                     logging.debug(f"No text found in image of page {i + 1}.")

#         # Combine all extracted text parts
#         combined_text = ''.join(text_parts)
#         logging.debug(f"Combined extracted text: {combined_text[:500]}...")  # Log the first 500 characters of the combined text
#         return combined_text

#     except Exception as e:
#         logging.error(f"Error extracting PDF context: {e}")
#         return ""

# def create_vector_store(pdf_path):
#     global vector_store

#     # Extract text from the PDF
#     full_text = extract_pdf_context(pdf_path)
#     if not full_text.strip():
#         logging.error("No text extracted from the PDF.")
#         return

#     # Split the extracted text into chunks
#     split_docs = text_splitter.split_text(full_text)
#     logging.debug(f"Split documents: {split_docs}")

#     # Create document objects with 'page_content' attribute
#     documents = [Document(page_content=doc) for doc in split_docs]

#     try:
#         # Generate embeddings for the documents
#         embeddings = get_embeddings([doc.page_content for doc in documents])

#         # Initialize Chroma with the embeddings
#         vector_store = Chroma.from_documents(documents=documents, embedding_function=lambda x: embeddings)

#         logging.debug("Chroma vector store created successfully.")
#     except Exception as e:
#         logging.error(f"Error initializing Chroma vector store: {e}")

# # Process the PDF query
# def process_pdf_query(pdf_path, question):
#     global vector_store
#     try:
#         if vector_store is None:
#             create_vector_store(pdf_path)

#         # Generate embedding for the question
#         question_embedding = get_embeddings([question]).cpu().detach().numpy()

#         # Perform similarity search
#         docs = vector_store.similarity_search(query_embeddings=question_embedding)

#         if docs:
#             context = " ".join([doc.page_content for doc in docs if hasattr(doc, 'page_content')])
#             logging.debug(f"Retrieved document context: {context[:500]}...")
#         else:
#             context = ""

#         if not context:
#             return {"error": "No relevant context found for your query."}

#         # Answer the question using the QA model
#         inputs = qa_tokenizer(question, context, return_tensors='pt', truncation=True, max_length=512, padding=True)

#         with torch.no_grad():
#             outputs = qa_model(**inputs)
#             answer_start = outputs.start_logits.argmax()
#             answer_end = outputs.end_logits.argmax() + 1

#             if answer_start < answer_end:
#                 answer = qa_tokenizer.convert_tokens_to_string(
#                     qa_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end])
#                 )
#             else:
#                 answer = "Unable to determine the answer from the provided context."

#         logging.debug(f"Answer: {answer}")
#         return {"answer": answer}

#     except Exception as e:
#         logging.error(f"Error during PDF query processing: {e}")
#         return {"error": str(e)}

# # Handle chat queries
# def handle_chat_query(user_input):
#     chat_chain = create_chat_chain()
#     try:
#         response = chat_chain.run({"question": user_input})

#         if isinstance(response, str) and response.strip():
#             return {"response": response}
#         else:
#             return {"response": "Invalid response format from the chat model."}

#     except Exception as e:
#         logging.error(f"Error during chat generation: {e}")
#         return {"response": "An error occurred while processing your request."}

# if __name__ == "__main__":
#     # Uncomment the following lines to test PDF processing and chat
#     # fun2 = extract_pdf_context(pdf_path='gpt.pdf')
#     # fun = create_vector_store(pdf_path='gpt.pdf')
#     # print(fun2)
#     # print(fun)

    

#     # Uncomment the following lines to enable user interaction
#     print("Welcome to the Chatbot! Type 'exit' to quit.")
#     while True:
#         user_input = input("User: ")
        
#         if user_input.lower() == "exit":
#             print("Exiting the chatbot. Goodbye!")
#             break
        
#         elif user_input.lower().startswith("pdf:"):
#             pdf_path = user_input.split(":")[1].strip()

#             if os.path.exists(pdf_path):
#                 question = input("Question about the PDF: ")

#                 response = process_pdf_query(pdf_path, question)

#                 if 'answer' in response:
#                     print(f"Bot: {response['answer']}\n")
#                 else:
#                     print(f"Bot: {response['error']}\n")
#             else:
#                 print("Invalid PDF path. Please provide a valid file path.\n")
        
#         else:
#             response = handle_chat_query(user_input)
#             print(f"Bot: {response['response']}\n")


















# # import os
# # import fitz
# # from transformers import (
# #     AutoTokenizer,
# #     AutoModelForSeq2SeqLM,
# #     AutoModelForQuestionAnswering,
# #     AutoModel,
# # )
# # from langchain import LLMChain
# # from langchain.prompts import PromptTemplate
# # from langchain_community.llms import HuggingFaceHub
# # from langchain.memory import ConversationBufferMemory
# # import torch
# # import logging
# # from langchain.text_splitter import CharacterTextSplitter
# # from langchain.docstore.document import Document
# # from pdf2image import convert_from_path
# # import pytesseract
# # import PyPDF2  # Import PyPDF2
# # from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer



# # # Hugging Face API Token
# # os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

# # # Model details
# # chat_model_name = "google/flan-t5-large"
# # qa_model_name = "facebook/m2m100_418M"

# # # Load models
# # chat_tokenizer = AutoTokenizer.from_pretrained(chat_model_name, trust_remote_code=True)
# # chat_model = AutoModelForSeq2SeqLM.from_pretrained(chat_model_name)
# # qa_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
# # qa_model =  M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

# # # Text splitter for splitting PDF content
# # text_splitter = CharacterTextSplitter(
# #     separator="\n",
# #     chunk_size=300,
# #     chunk_overlap=200,
# #     length_function=len,
# # )

# # # Prompt templates
# # chat_prompt_template = PromptTemplate(
# #     input_variables=["question"],
# #     template="Question: {question}\nAnswer: Let's think step by step."
# # )

# # qa_prompt_template = PromptTemplate(
# #     input_variables=["context", "question"],
# #     template="Context: {context}\nQuestion: {question}\nAnswer:"
# # )

# # # Memory for conversation
# # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# # def create_chat_chain():
# #     try:
# #         return LLMChain(
# #             llm=HuggingFaceHub(
# #                 repo_id=chat_model_name,
# #                 model_kwargs={"temperature": 0, "max_length": 64}
# #             ),
# #             memory=memory,
# #             prompt=chat_prompt_template,
# #         )
# #     except Exception as e:
# #         logging.error(f"Error initializing Chat LLM Chain: {e}")
# #         exit(1)

# # def create_qa_chain():
# #     try:
# #         return LLMChain(
# #             llm=HuggingFaceHub(
# #                 repo_id=qa_model_name,
# #                 model_kwargs={"temperature": 0, "max_length": 512}
# #             ),
# #             memory=memory,
# #             prompt=qa_prompt_template,
# #         )
# #     except Exception as e:
# #         logging.error(f"Error initializing QA LLM Chain: {e}")
# #         exit(1)

# # def extract_pdf_context(pdf_path):
# #     try:
# #         text_parts = []

# #         # Use PyPDF2 to extract text first
# #         raw_text = ''
# #         with open(pdf_path, 'rb') as file:
# #             reader = PyPDF2.PdfReader(file)
# #             for i, page in enumerate(reader.pages):
# #                 text = page.extract_text()
# #                 if text:
# #                     raw_text += text
# #                     logging.debug(f"Extracted text from page {i + 1}: {text[:100]}...")  # Log the first 100 characters from each page
# #                 else:
# #                     logging.debug(f"No text found on page {i + 1}.")

# #         # If PyPDF2 extracted any text, append it to text_parts
# #         if raw_text.strip():
# #             text_parts.append(raw_text)
# #             logging.debug(f"Total extracted text from PDF: {raw_text[:500]}...")  # Log the first 500 characters of the extracted text
# #         else:
# #             logging.debug("No text found using PyPDF2, proceeding with image extraction.")

# #             # Convert PDF pages to images and use pytesseract for OCR
# #             images = convert_from_path(pdf_path)
# #             for i, image in enumerate(images):
# #                 ocr_text = pytesseract.image_to_string(image, lang='eng')
# #                 if ocr_text.strip():  # Check if any text was extracted from the image
# #                     text_parts.append(ocr_text)
# #                     logging.debug(f"Extracted text from image of page {i + 1}: {ocr_text[:100]}...")  # Log the first 100 characters from OCR
# #                 else:
# #                     logging.debug(f"No text found in image of page {i + 1}.")

# #         # Combine all extracted text parts
# #         combined_text = ''.join(text_parts)
# #         logging.debug(f"Combined extracted text: {combined_text[:500]}...")  # Log the first 500 characters of the combined text
# #         return combined_text

# #     except Exception as e:
# #         logging.error(f"Error extracting PDF context: {e}")
# #         return ""


# # def process_pdf_query(pdf_path, question):
# #     try:
# #         # Extract text from the PDF
# #         full_text = extract_pdf_context(pdf_path)

# #         # Ensure full_text is valid
# #         if full_text is None or not isinstance(full_text, str) or not full_text.strip():
# #             logging.error("No valid text extracted from the PDF.")
# #             return {"error": "No text extracted from the PDF."}

# #         # Use the QA model to answer the question based on the entire PDF content
# #         qa_chain = create_qa_chain()
        
# #         # Prepare inputs for the QA model
# #         # Tokenize the full_text and question together
# #         inputs = qa_tokenizer(
# #             full_text,
# #             question,
# #             return_tensors='pt',
# #             truncation=True,
# #             max_length=512,
# #             padding="max_length"
# #         )

# #         # Add decoder_input_ids for generation if required
# #         decoder_input_ids = qa_tokenizer.encode(question, return_tensors='pt')
        
# #         with torch.no_grad():
# #             outputs = qa_model(input_ids=inputs['input_ids'], decoder_input_ids=decoder_input_ids)

# #             answer_start = outputs.start_logits.argmax()
# #             answer_end = outputs.end_logits.argmax() + 1

# #             if answer_start < answer_end:
# #                 answer = qa_tokenizer.convert_tokens_to_string(
# #                     qa_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end])
# #                 )
# #             else:
# #                 answer = "Unable to determine the answer from the provided context."

# #         logging.debug(f"Answer: {answer}")
# #         return {"answer": answer}

# #     except Exception as e:
# #         logging.error(f"Error during PDF query processing: {e}")
# #         return {"error": str(e)}


# # # Handle chat queries
# # def handle_chat_query(user_input):
# #     chat_chain = create_chat_chain()
# #     try:
# #         response = chat_chain.run({"question": user_input})

# #         if isinstance(response, str) and response.strip():
# #             return {"response": response}
# #         else:
# #             return {"response": "Invalid response format from the chat model."}

# #     except Exception as e:
# #         logging.error(f"Error during chat generation: {e}")
# #         return {"response": "An error occurred while processing your request."}

# # if __name__ == "__main__":
# #     # Uncomment the following lines to test PDF processing and chat
# #     # fun2 = extract_pdf_context(pdf_path='gpt.pdf')
# #     # print(fun2)

# #     # Uncomment the following lines to enable user interaction
# #     print("Welcome to the Chatbot! Type 'exit' to quit.")
# #     while True:
# #         user_input = input("User: ")
        
# #         if user_input.lower() == "exit":
# #             print("Exiting the chatbot. Goodbye!")
# #             break
        
# #         elif user_input.lower().startswith("pdf:"):
# #             pdf_path = user_input.split(":")[1].strip()

# #             if os.path.exists(pdf_path):
# #                 question = input("Question about the PDF: ")

# #                 response = process_pdf_query(pdf_path, question)

# #                 if 'answer' in response:
# #                     print(f"Bot: {response['answer']}\n")
# #                 else:
# #                     print(f"Bot: {response['error']}\n")
# #             else:
# #                 print("Invalid PDF path. Please provide a valid file path.\n")
        
# #         else:
# #             response = handle_chat_query(user_input)
# #             print(f"Bot: {response['response']}\n")




















import os
import PyPDF2
import logging
import google.generativeai as genai
from dotenv import load_dotenv

# Configure logging
load_dotenv()

logging.basicConfig(level=logging.DEBUG)
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Generation configuration for Gemini
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize the Gemini model with custom generation config for PDF and chat
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Directory to save uploaded files
# UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyPDF2."""
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            extracted_text = ""
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text
            logging.debug(f"Extracted PDF text: {extracted_text[:500]}...")  # Log a snippet of the extracted text
            return extracted_text
    except Exception as e:
        logging.error(f"Error extracting PDF context: {e}")
        return ""

def process_pdf_query(pdf_path, question):
    """Processes the PDF query by uploading the PDF and asking a question."""
    try:
        # Extract context from PDF
        context = extract_text_from_pdf(pdf_path)

        if not context:
            return {"error": "Failed to extract context from the PDF."}

        # Start a chat session for PDF interaction
        chat_session = model.start_chat(
            history=[
                {"role": "user", "parts": [{"text": f"Context: {context}"}]},
                {"role": "user", "parts": [{"text": f"Question: {question}"}]}
            ]
        )

        # Get the response from the chat session
        response = chat_session.send_message(question)
        
        return {"text": response.text}
    
    except Exception as e:
        logging.error(f"Error processing PDF query: {e}")
        return {"error": f"An error occurred while processing the PDF query: {str(e)}"}


def handle_chat_query(user_input):
    """Handles the chat query using the Gemini chat model."""
    try:
        # Start a chat session
        chat_session = model.start_chat(
            history=[]
        )
        
        # Send the user input and get a response
        response = chat_session.send_message(user_input)
        
        return {"text": response.text}
    
    except Exception as e:
        logging.error(f"Error during chat query: {e}")
        return {"error": f"An error occurred while processing the chat input: {str(e)}"}

# if __name__ == "__main__":
#     print("Welcome to the Chatbot! Type 'exit' to quit.")
#     while True:
#         user_input = input("User: ")
#         if user_input.lower() == "exit":
#             break
#         elif user_input.lower().startswith("pdf:"):
#             pdf_path = user_input.split(":")[1].strip()
#             question = input("Question about the PDF: ")
#             result = process_pdf_query(pdf_path, question)
#             if 'text' in result:
#                 print(f"Bot: {result['text']}\n")
#             else:
#                 print(f"Bot: {result['error']}\n")
#         else:
#             response = handle_chat_query(user_input)
#             print(f"Bot: {response['text'] if 'text' in response else response['error']}\n")
