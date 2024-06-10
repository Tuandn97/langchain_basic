import os
import json
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
       "\n 3. Technology"

CHAT_HISTORY_FILE = "chat_history.json"


def convert_chat_string_to_prompt(chat_string, type="ai"):
    if type == "user":
        return HumanMessage(content=chat_string)
    elif type == "ai":
        return AIMessage(content=chat_string)
    else:
        raise ValueError("Invalid type")


def load_llm(provider="google"):
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
                        "extensive medical expertise. "
    elif character == "finance":
        system_prompt = "You are a distinguished financial expert, renowned for your rigorous analysis and sound " \
                        "advice. You will provide the user with authoritative and insightful responses to their " \
                        "questions about personal finance matters. "
    elif character == "technology":
        system_prompt = "You are a leading-edge technology specialist, at the forefront of industry innovations. You " \
                        "will address the user's questions about technology-related issues with a keen, innovative, " \
                        "and solution-focused mindset. "
    else:
        raise ValueError("Invalid character")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    return prompt


def simple_chat_bot(user_input, chat_history, provider="google", character="healthy-care"):
    llm = load_llm(provider)
    prompt = get_prompt(character)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    output = chain.invoke({"input": user_input, 'chat_history': chat_history})
    return output


def save_chat_history(chat_history, character):
    try:
        with open(CHAT_HISTORY_FILE, "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {}

    # Convert chat_history to a list of dicts
    chat_history_dicts = [
        {"role": "user", "content": message.content} if isinstance(message, HumanMessage)
        else {"role": "assistant", "content": message.content}
        for message in chat_history
    ]

    data[character] = chat_history_dicts

    with open(CHAT_HISTORY_FILE, "w") as file:
        json.dump(data, file, indent=4)


def load_chat_history(character):
    try:
        with open(CHAT_HISTORY_FILE, "r") as file:
            data = json.load(file)
            if character in data:
                # Convert the list of dicts back to HumanMessage and AIMessage objects
                chat_history = [
                    HumanMessage(content=message["content"]) if message["role"] == "user"
                    else AIMessage(content=message["content"])
                    for message in data[character]
                ]
                return chat_history
            else:
                return []
    except FileNotFoundError:
        return []


def main():
    provider = "google"
    print(MENU)
    while True:
        try:
            user_choice = int(input(">> "))
            if user_choice == 1:
                character = "health"
                print(f"You are choosing the {character} expert")
                chat_history = load_chat_history(character)
                print(chat_history)
                break
            elif user_choice == 2:
                character = "finance"
                print(f"You are choosing the {character} expert")
                chat_history = load_chat_history(character)
                break
            elif user_choice == 3:
                character = "technology"
                print(f"You are choosing the {character} expert")
                chat_history = load_chat_history(character)
                break
            else:
                print("Invalid choice. Please try again.")
                print(MENU)
        except ValueError:
            print("Invalid input. Please enter a number.")

    while True:
        user_input = input("Enter your message (press '0' to exit): ")
        output = simple_chat_bot(user_input, chat_history, provider, character)
        chat_history.append(convert_chat_string_to_prompt(user_input, type="user"))
        chat_history.append(convert_chat_string_to_prompt(output, type="ai"))
        print(output)
        save_chat_history(chat_history, character)
        if user_input.lower() == "0":
            break


if __name__ == "__main__":
    main()
