#database hosted locally on computer, to really quickly look up info
#info is passed onto model to give context to replies. 
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings #imports embeddings model, turns text into values readable my model
from langchain_chroma import Chroma #chroma is a database for storing embeddings
from langchain_core.documents import Document #to create documents that store raw text and metadata
import os 
import pandas as pd #pandas handles csv files

load_dotenv()

dataframe = pd.read_csv("realistic_restaurant_reviews.csv") #loads the csv file into a dataframe using pandas
embeddings = OpenAIEmbeddings(model="text-embedding-3-large") #embedding model to convert text to vectors

db_location = "./chroma_langchain_db" #tells computer where to look for vector db
add_documents = not os.path.exists(db_location) #if the db doesnt exist, we add documents to it.

if add_documents: #if db deosnt exist
    documents = []
    ids = []
    for i, row in dataframe.iterrows(): #loops each row in pandas df
        document = Document(
            #creates documents, with page content from each column plus additional metadata (more columns).
            page_content= row["Title"] + " " + row["Review"], 
            metadata = {"rating" : row["Rating"], "date": row["Date"]},
            id = str(i)#index of value in the row, assigns unique id
        )
        ids.append(str(i)) #list of unique ids
        documents.append(document) #appends to documents list

vector_store = Chroma( #storing the embeddings
    collection_name = "restaurant_reviews",
    persist_directory= db_location, #stores it to disk, so easier access to data in runs of program
    embedding_function= embeddings #converts text to vectors

)

if add_documents: 
    #if we need to add documents, we can add them into the chroma db using the embeddings model to convert text to vectors
    vector_store.add_documents(documents = documents, ids = ids) #stores them under "restaurant_reviews" collection
    
retriever = vector_store.as_retriever(
    #makes a retriever object, whicch can be used by model to access the chroma vectors
    search_kwargs = {"k":10} #10 most relevant reviews to pass to model
)