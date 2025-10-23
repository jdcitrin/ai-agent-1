from dotenv import load_dotenv #imports api key
from langchain_openai import OpenAI #imports model
from langchain_core.prompts import ChatPromptTemplate #imports template to prompt model
from vector import retriever #imports the retriever function, which gets relevant reviews from database.


load_dotenv()

model = OpenAI(model_name="gpt-4o-mini") #loads model

template = """
You are an expert food reviewer rating the pizza restaurant. Answer the following question.
You are speaking to a person who is looking for honest reviews about pizza restaurants, you can also chat with them about non-related questions, but steer it back to pizza.
Take the question as a full statement, do not try to finish the users sentence. If you do not understand, ask them to rephrase.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template) #loads prompt for ai, using given template above.

chain = prompt | model #feeds prompt into model

while True:
    print("=================================")
    question = input("Ask your question: (q to quit): ")
    if question == "q":
        break
    print("=================================")
    #formatting user question

   
    reviews = retriever.invoke(question) #gets relevant reviews from vector database
    result = chain.invoke({"reviews": reviews, "question": question}) #feeds reviews and user question into model chain
    print(result)