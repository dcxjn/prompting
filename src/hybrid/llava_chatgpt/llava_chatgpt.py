import os

import time

from dotenv import load_dotenv

from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
)

from langchain.globals import set_debug

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_openai import ChatOpenAI

import torch

from PIL import Image

from src.utils.image_util import load_image, resize_image
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

        quant_config = 4

        # processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-34b-hf")
        processor = LlavaNextProcessor.from_pretrained(
            "llava-hf/llava-v1.6-vicuna-13b-hf"
        )

        if quant_config == 4:
            model = LlavaNextForConditionalGeneration.from_pretrained(
                # "llava-hf/llava-v1.6-34b-hf",
                "llava-hf/llava-v1.6-vicuna-13b-hf",
                torch_dtype=torch.bfloat16,
                do_sample=True,
                temperature=0.2,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                device_map="cuda",
            )
        elif quant_config == 8:
            model = LlavaNextForConditionalGeneration.from_pretrained(
                # "llava-hf/llava-v1.6-34b-hf",
                "llava-hf/llava-v1.6-vicuna-13b-hf",
                torch_dtype=torch.bfloat16,
                do_sample=True,
                temperature=0.2,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map="cuda",
            )
        else:
            model = LlavaNextForConditionalGeneration.from_pretrained(
                # "llava-hf/llava-v1.6-34b-hf",
                "llava-hf/llava-v1.6-vicuna-13b-hf",
                torch_dtype=torch.bfloat16,
                do_sample=True,
                temperature=0.2,
                device_map="cuda",
            )

        # Load image
        image = Image.open(info_dict["image_path"])

        prompt = f"""
        Given the image, answer the questions.
        Give a reason for each of your answers.
        Questions: {info_dict["relevant_qns"]}
        """

        prompt = "<image>" + f"USER: {prompt}\nASSISTANT:"
        input = processor(prompt, image, return_tensors="pt").to("cuda")
        outputs = model.generate(**input, max_new_tokens=2048)

        output = processor.decode(outputs[0], skip_special_tokens=True)

        info_dict["image_features"] = output

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
        Given the task of: {info_dict["task"]}, think of all the relevant information that is required to complete the task.
        Generate the relevant questions in bullet point form.
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
        You are given the image features: {info_dict["image_features"]}
        Provide a detailed step-by-step guide on how the robot would complete the task.
        Link each instruction to an observation in the image in this format: "Observation: Instruction"
        Ensure that the CORRECT observation is linked to the instruction.
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

    # image_path = input("Enter the path of the image: ")
    # image_path = r"images/fridge_lefthandle.jpg"
    # image_path = r"images/housedoor_knob_push.jpg"
    # image_path = r"images/browndoor_knob_pull.jpg"
    # image_path = r"images/labdoor_straighthandle_pull.jpg"
    image_path = r"images/bluedoor_knob_push.jpg"
    # image_path = r"images/whitetable.jpg"

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

    print("\n=== RELEVANT QUESTIONS ===\n\n", info_dict["relevant_qns"])
    print("\n=== IMAGE FEATURES ===\n\n", info_dict["image_features"])
    print("\n=== ROBOT INSTRUCTIONS ===\n\n", info_dict["bot_inst"])
    print("\n=== CODE SUMMARY ===\n\n", info_dict["code_summary"])
    print("\n===\n\nTIME TAKEN (s): ", (end - start))

    # Clear the session IDs
    store = {}


if __name__ == "__main__":
    main()
