from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_intro.chatbot import chat_model
from langchain_intro.chatbot import review_chain, hospital_agent_executor

messages = [
    SystemMessage(
        content="""
            You're an assistant knowledgeable about healthcare.
            Only answer healthcare-related questions.
            """
    ),
    HumanMessage(content="How do I change tire?"),   
]

# print(chat_model.invoke(messages))

# print(chat_model.invoke("give me a list of systoms for diabetes"))

# for s in chat_model.stream("give me a list of systoms for diabetes"):
#     print(s.content, end="", flush=True)

# print("\n")
# for c in chat_model.batch(["What is an aveoli?", "what is a penicillin allergy?"]):
#     print(c.content)


# context = "I had a great stay!"
# question = "Did anyone have a positive experience?"
# print(review_chain.invoke({"context": context, "question": question}))

# question = """Has anyone complained about
#             communication with the hospital staff?"""
# print(review_chain.invoke(question))

print(
    hospital_agent_executor.invoke(
        {
            "input": "What is the current wait time at hospital C?"
        }
    )
)

print(
    hospital_agent_executor.invoke(
        {
            "input": "What have patients said about their comfort at the hospital?"
        }
    )
)