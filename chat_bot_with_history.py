import os
# load .env
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage

MENU = "Please select your Expert that you want." \
       "\n 1. Health" \
       "\n 2. Finance" \
       "\n 3. Technolog"


def convert_chat_string_to_prompt(chat_string, type="ai"):
    if type == "user":
        return HumanMessage(content=chat_string)
    elif type == "ai":
        return AIMessage(content=chat_string)
    else:
        raise ValueError("Invalid type")


def load_llm(provider="google"):
    # GeminiAI
    if provider == "google":
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    elif provider == "openai":
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")
    else:
        raise ValueError("Invalid provider")
    return llm


def get_prompt(character="healthy-care"):
    if character == "health":
        system_prompt = "You are a highly knowledgeable and experienced health specialist. You will confidently and " \
                        "compassionately answer the user's questions about health-related issues, drawing upon your " \
                        "extensive medical expertise "
    elif character == "finance":
        system_prompt = "You are a distinguished financial expert, renowned for your rigorous analysis and sound " \
                        "advice. You will provide the user with authoritative and insightful responses to their " \
                        "questions about personal finance matters. "
    elif character == "technology":
        system_prompt = "You are a leading-edge technology specialist, at the forefront of industry innovations. You " \
                        "will address the user's questions about technology-related issues with a keen, innovative, " \
                        "and solution-focused mindset "
    else:
        raise ValueError("Invalid character")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    return prompt


def simple_chat_bot(user_input, chat_history, provider="google", character="healthy-care"):
    # load llm
    llm = load_llm(provider)
    # get prompt
    prompt = get_prompt(character)
    # init chain
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    # invoke
    output = chain.invoke({"input": user_input, 'chat_history': chat_history})
    return output


def main():
    provider = "google"
    chat_history = []
    print(MENU)
    while True:
        try:
            user_choice = int(input(">> "))
            if user_choice == 1:
                character = "health"
                print(f"You are choosing the {character} expert")
                break
            elif user_choice == 2:
                character = "finance"
                print(f"You are choosing the {character} expert")
                break
            elif user_choice == 3:
                character = "technology"
                print(f"You are choosing the {character} expert")
                break
            else:
                print("Invalid choice. Please try again.")
                print(MENU)
        except ValueError:
            print("Invalid input. Please enter a number.")

    while True:
        user_input = input("Enter your message (press 'enter' to exit): ")
        if user_input.lower() == "":
            break
        output = simple_chat_bot(user_input,chat_history, provider, character)
        chat_history.append(convert_chat_string_to_prompt(user_input, type="user"))
        chat_history.append(convert_chat_string_to_prompt(output, type="ai"))
        print(output)


if __name__ == "__main__":
    main()

