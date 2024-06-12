import os
from dotenv import load_dotenv

from langchain.chains import TransformChain
from langchain.globals import set_debug

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_openai import ChatOpenAI

from src.utils.image_util import load_image, resize_image
from src.utils.memory_history_util import InMemoryHistory

"""
Author: Dayna Chia
Date: 2024-06-11
Notes: 
- Uses the method of Self-Consistency (Prompt Design and Engineering: Introduction and Advanced Methods, 2024)
"""


def main():

    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    set_debug(True)

    load_image_chain = TransformChain(
        input_variables=["image_path"],
        output_variables=["image"],
        transform=load_image,
    )

    # Intialize the store for session IDs
    store = {}

    def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
        """Get the message history for the given session ID."""

        if session_id not in store:
            store[session_id] = InMemoryHistory()
        return store[session_id]

    def get_instructions(inputs: dict, session_id: str) -> dict:
        """Get the instructions for a robot to perform the required task given an image."""

        # llm = ChatOpenAI(temperature=0, model="gpt-4-turbo-2024-04-09", max_tokens=4096)
        llm = ChatOpenAI(temperature=0.5, model="gpt-4o", max_tokens=4096)

        runnable_with_history = RunnableWithMessageHistory(
            llm,
            get_by_session_id,
        )

        prompt1 = f"""
        Observe the given image and its details.
        Imagine you are in control of a robotic arm with the following commands: {inputs["bot_commands"]}
        Provide a detailed step-by-step guide on how the robot would complete the task of: {inputs["task"]}
        """

        runnable_with_history.invoke(
            [
                HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{inputs['image']}"
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt1,
                        },
                    ]
                )
            ],
            config={"configurable": {"session_id": session_id}},
        )

        prompt2 = f"""
        By referencing an observation in the image, ensure each instruction is accurate. Do not make assumptions.
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
            config={"configurable": {"session_id": session_id}},
        )

        return {"bot_inst": msg.content}

    def get_n_instructions(inputs: dict) -> dict:

        output1 = get_instructions(inputs, "1")
        output2 = get_instructions(inputs, "2")
        output3 = get_instructions(inputs, "3")

        return {
            "image": inputs["image"],
            "bot_inst1": output1["bot_inst"],
            "bot_inst2": output2["bot_inst"],
            "bot_inst3": output3["bot_inst"],
        }

    def refine_instructions(inputs: dict) -> dict:

        llm = ChatOpenAI(temperature=0.2, model="gpt-4o", max_tokens=4096)

        prompt = f"""
        Instructions 1: {inputs["bot_inst1"]}
        Instructions 2: {inputs["bot_inst2"]}
        Instructions 3: {inputs["bot_inst3"]}
        Refer to the image again. Compare the different sets of instructions and list down the discrepancies.
        Make necessary corrections to ensure consistency in the instructions and provide a final set of instructions.
        """

        msg = llm.invoke(
            [
                HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{inputs['image']}"
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ]
                )
            ]
        )

        return {"bot_inst": msg.content}

    def get_code_summary(inputs: dict) -> dict:
        """Get the code commands given the instructions."""

        llm = ChatOpenAI(temperature=0.2, model="gpt-4", max_tokens=4096)

        prompt = f"""
        Instructions: {inputs["bot_inst"]}
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

        return {"code_summary": msg.content}

    def run_chain(image_path: dict, task: str, bot_commands: str) -> str:
        """Run the chain."""

        chain = (
            load_image_chain
            | get_n_instructions
            | refine_instructions
            | get_code_summary
        )
        return chain.invoke(
            {
                "image_path": f"{image_path}",
                "task": task,
                "bot_commands": bot_commands,
            }
        )

    # Robot commands available
    bot_commands = """
        1. move_to(x, y)
        2. grab(object)
        3. release(object)
        4. push(object)
        5. pull(object)
        6. rotate(angle)
    """

    # image_path = input("Enter the path of the image: ")
    # image_path = r"images\fridge_lefthandle.jpg"
    # image_path = r"images\housedoor_knob_push.jpg"
    # image_path = r"images\labdoor_straighthandle_pull.jpg"
    # image_path = r"images\browndoor_knob_pull.jpg"
    image_path = r"images\bluedoor_knob_push.jpg"
    # image_path = r"images\whitetable.jpg"

    resize_image(image_path, image_path)

    # Define the task to be performed
    task = input("Enter the task to be performed: ")

    result = run_chain(image_path, task, bot_commands)
    print("\n==========\n")
    print(result["code_summary"])

    # Clear the session IDs
    store = {}


if __name__ == "__main__":
    main()
