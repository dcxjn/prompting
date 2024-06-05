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
from langchain import globals
from langchain_core.runnables import chain


def main():

    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    def load_image(inputs: dict) -> dict:
        """Load image from file and encode it as base64."""

        image_path = inputs["image_path"]

        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        image_base64 = encode_image(image_path)
        return {"image": image_base64}

    load_image_chain = TransformChain(
        input_variables=["image_path"], output_variables=["image"], transform=load_image
    )

    def image_description_model(inputs: dict) -> str | dict:
        """Get verbose description of the image."""

        model = ChatOpenAI(temperature=0.2, model="gpt-4o", max_tokens=4096)
        msg = model.invoke(
            [
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "Describe the image in a factual manner, using only the features seen in the image.",
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

    def get_bot_instructions(image_path: str, task: str) -> str:
        """Get the instuctions for the bot to perform the required task given an image."""

        prompt = """
        Imagine you are in control of a robotic arm with the following commands:
        1. move_to(x, y)
        2. grab(object)
        3. release(object)
        4. push(object)
        5. pull(object)

        Given the image, the commands you have been provided and the assigned task, please provide a step-by-step guide on how to complete the task.
        """

        chain = load_image_chain | image_description_model
        return chain.invoke({"image_path": f"{image_path}", "prompt": prompt})

    image_path = input("Enter the path of the image: ")
    task = input("Enter the task to be performed: ")
    result = get_bot_instructions(image_path, task)
    print(result)


if __name__ == "__main__":
    main()
