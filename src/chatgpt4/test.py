# To test:
# 1. Get description of image
# 2. Set the role
# 3. Give the list of available commands
# 4. Provide the action to be taken alongside the prompt.

import os
from dotenv import load_dotenv

from langchain.chains import TransformChain
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.globals import set_debug

from src.utils import image_utils


def main():

    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    set_debug(True)

    load_image_chain = TransformChain(
        input_variables=["image_path"],
        output_variables=["image"],
        transform=image_utils.load_image,
    )

    def human_instructions_model(inputs: dict) -> str:
        """Get the instructions for a human to perform the required task given an image."""

        prompt = f"""
        Observe the given image and its details. Provide a step-by-step guide on how a human would complete the task of: {inputs['task']}. 
        Link each instruction to an observation in the image in this format: "Observation: Instruction".
        Think of any secondary tasks that may need to be performed after completing the primary task.
        """

        model = ChatOpenAI(temperature=0.2, model="gpt-4o", max_tokens=4096)
        msg = model.invoke(
            [
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{inputs['image']}"
                            },
                        },
                    ]
                )
            ]
        )
        return msg.content

    def robot_instructions_model(human_inst: str) -> str:
        """Get the instuctions for the bot to perform the required task given the human instructions."""

        prompt = f"""
        Imagine you are in control of a robotic arm with the following commands:
        1. move_to(x, y)
        2. grab(object)
        3. release(object)
        4. push(object)
        5. pull(object)
        6. rotate(angle)

        Human instuctions: {human_inst}

        Given the human instructions you have been provided and the commands you are able to execute, provide a step-by-step guide on how the robot would complete the task.
        """

        model = ChatOpenAI(temperature=0.2, model="gpt-4o", max_tokens=4096)
        msg = model.invoke(
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
        return msg.content

    def run_chain(image_path: dict, task: str) -> str:

        chain = load_image_chain | human_instructions_model | robot_instructions_model
        return chain.invoke({"image_path": f"{image_path}", "task": task})

    # image_path = input("Enter the path of the image: ")
    # image_path = "images\housedoor_knob_push.jpg"
    image_path = "images\labdoor_straighthandle_pull.jpg"
    task = input("Enter the task to be performed: ")
    result = run_chain(image_path, task)
    print(result)


if __name__ == "__main__":
    main()
