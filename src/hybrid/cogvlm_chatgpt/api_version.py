import os

import time

from dotenv import load_dotenv

from langchain.globals import set_debug

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_openai import ChatOpenAI

from gradio_client import Client, file

from src.utils.memory_history_util import InMemoryHistory


def main():

    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    set_debug(False)

    # Intialize the store for session IDs
    store = {}

    def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
        """Get the message history for the given session ID."""

        if session_id not in store:
            store[session_id] = InMemoryHistory()
        return store[session_id]

    def get_image_features(info_dict: dict) -> dict:
        """Get the image features."""

        prompt = f"""
        You are a robotic arm positioned facing the image.
        Examine the given image and answer the following questions.
        {info_dict["relevant_qns"]}
        
        """

        client = Client("THUDM/CogVLM-CogAgent")
        result = client.predict(
            input_text=prompt,
            temperature=0,
            image_prompt=file(info_dict["image_path"]),
            result_previous=[],
            hidden_image="Hello!!",
            is_english=True,
            api_name="/post",
        )

        response = result[1][0][1]

        if response == "Timeout! Please wait a few minutes and retry.":
            print(response)
            exit()
        else:
            info_dict["image_features"] = response

        return info_dict

    def get_instructions(info_dict: dict) -> dict:
        """Get the instructions for a robot to perform the required task given an image."""

        llm = ChatOpenAI(temperature=0.2, model="gpt-4o", max_tokens=4096)

        runnable_with_history = RunnableWithMessageHistory(
            llm,
            get_by_session_id,
        )

        prompt1 = f"""
        Imagine you are in control of a robotic arm with the following commands: {info_dict["bot_commands"]}
        Given the task of: {info_dict["task"]}, what are some information essential to completing the task?
        Generate questions to obtain the desired information.
        Be extremely specific in your phrasing, ensuring that the questions are understandable by a child.
        Give only the questions; give them in a numbered list.
        """

        msg = runnable_with_history.invoke(
            [
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": prompt1,
                        },
                    ]
                )
            ],
            config={"configurable": {"session_id": "abc"}},
        )

        info_dict["relevant_qns"] = msg.content

        info_dict = get_image_features(info_dict)

        prompt2 = f"""
        Here are the answers to the questions you have generated earlier: {info_dict["image_features"]}
        Using the answers and the available robot commands, provide a detailed step-by-step guide on how the robot would complete the task.
        Note that the robot is in the position of the 'viewer'.
        Give a reason for each step.
        """

        msg = runnable_with_history.invoke(
            [
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": prompt2,
                        },
                    ]
                )
            ],
            config={"configurable": {"session_id": "abc"}},
        )

        info_dict["bot_inst"] = msg.content

        return info_dict

    def get_code_summary(info_dict: dict) -> dict:
        """Get the code commands given the instructions."""

        llm = ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=4096)

        prompt = f"""
        Instructions: {info_dict["bot_inst"]}
        Given the instructions, provide the code commands to execute the task and concise comments only.
        """

        msg = llm.invoke(
            [
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ]
                )
            ]
        )

        info_dict["code_summary"] = msg.content

        return info_dict

    # Robot commands available
    bot_commands = """
        1. move_to(x, y)
        2. grab(object)
        3. release(object)
        4. push(object)
        5. pull(object)
        6. rotate(angle)
    """

    # [DOORS]
    # image_path = r"images/autodoor.jpg"
    # image_path = r"images/blackdoor_handle_push.jpg"
    # image_path = r"images/bluedoor_knob_push.jpg"
    # image_path = r"images/browndoor_knob_pull.jpg"
    image_path = r"images/glassdoor_sliding.jpg"
    # image_path = r"images/housedoor_knob_push.jpg"
    # image_path = r"images/labdoor_lever_pull.jpg"
    # image_path = r"images/metaldoor_lever_pull.jpg"
    # image_path = r"images/pinkdoor_knob_pull.jpg"
    # image_path = r"images/pvcdoor_folding.jpg"

    # [MISC]
    # image_path = r"images/whitetable.jpg"
    # image_path = r"images/threat_detection.jpg"
    # image_path = r"images/fridge_lefthandle.jpg"

    # resize_image(image_path, image_path)

    # Define the task to be performed
    task = input("Enter the task to be performed: ")

    # Define the initial info_dict
    info_dict = {
        "image_path": image_path,
        "task": task,
        "bot_commands": bot_commands,
    }

    start = time.time()

    # Run the chain
    info_dict = get_instructions(info_dict)
    info_dict = get_code_summary(info_dict)

    end = time.time()

    print("\n=== RELEVANT QUESTIONS ===\n\n" + info_dict["relevant_qns"])
    print("\n=== IMAGE FEATURES ===\n\n" + info_dict["image_features"])
    print("\n=== ROBOT INSTRUCTIONS ===\n\n" + info_dict["bot_inst"])
    print("\n=== CODE SUMMARY ===\n\n" + info_dict["code_summary"])
    print("\n===\n\nTIME TAKEN (s):", (end - start))

    # Clear the session IDs
    store = {}


if __name__ == "__main__":
    main()
