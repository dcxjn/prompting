import os
from dotenv import load_dotenv

from langchain.chains import TransformChain
from langchain.globals import set_debug

from langchain_core.messages import HumanMessage

from langchain_openai import ChatOpenAI

from src.utils.image_util import load_image, resize_image

"""
Author: Dayna Chia
Date: 2024-06-04
Note: 
- Uses the method of Chain of Thought (Prompt Design and Engineering: Introduction and Advanced Methods, 2024)
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

    def get_human_instructions(inputs: dict) -> dict:
        """Get the instructions for a human to perform the required task given an image."""

        prompt = f"""
        Observe the given image and its details. Pick out only the relevant details for the task of: {inputs["task"]}.
        Provide a detailed step-by-step guide on how a human would complete the task.
        Link each instruction to an observation in the image in this format: "Observation: Instruction".
        """

        llm = ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=4096)
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

        return {"human_inst": msg.content, "bot_commands": inputs["bot_commands"]}

    def get_robot_instructions(inputs: dict) -> dict:
        """Get the instuctions for the bot to perform the required task given the human instructions."""

        prompt = f"""
        Imagine you are in control of a robotic arm with the following commands: {inputs["bot_commands"]}
        Human instuctions: {inputs["human_inst"]}
        Given the human instructions you have been provided and the commands you are able to execute, provide a guide on how the robot would complete the task.
        """

        llm = ChatOpenAI(temperature=0.2, model="gpt-4o", max_tokens=4096)
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
            | get_human_instructions
            | get_robot_instructions
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


if __name__ == "__main__":
    main()
