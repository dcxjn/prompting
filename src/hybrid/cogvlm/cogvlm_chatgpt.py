import os
from dotenv import load_dotenv

from transformers import AutoModelForCausalLM, LlamaTokenizer

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

         # Set tokenizer
        tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            'THUDM/cogvlm-chat-hf',
            do_sample=False,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to('cuda').eval()

        # Load image
        image = Image.open(info_dict['image_path']).convert('RGB')

        prompt = f"""
        Given the task: {info_dict["task"]}, describe all RELEVANT features in the image.
        Be as detailed and specific as possible. Are these features applicable to the task?
        Give your observations in the format of bullet points.
        """

        input = model.build_conversation_input_ids(tokenizer, query=prompt, history=None, images=[image])
        input = {
            'input_ids': input['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': input['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': input['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[input['images'][0].to('cuda').to(torch.bfloat16)]],
        }
        gen_kwargs = {"max_length": 2048, "do_sample": False}

        with torch.no_grad():
            output = model.generate(**input, **gen_kwargs)
            output = output[:, input['input_ids'].shape[1]:]

        output = tokenizer.decode(output[0])

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
        You are given the image features: {info_dict["image_features"]}
        Provide a detailed step-by-step guide on how a human would complete the task of: {info_dict["task"]}
        Link each instruction to an observation in the image in this format: "Observation: Instruction"
        Ensure that the CORRECT observation is linked to the instruction.
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

        info_dict["human_inst"] = msg.content

        prompt2 = f"""
        Imagine you are in control of a robotic arm with the following commands: {info_dict["bot_commands"]}
        Given the human instructions you have generated, provide a guide on how the robot would complete the task.
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

        prompt3 = f"""
        Reflect on the instructions produced. Are they accurate and LOGICAL given the image features?
        Give precise instructions, avoid giving 'or' options.
        """

        msg = runnable_with_history.invoke(
            [
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": prompt3,
                        },
                    ]
                )
            ],
            config={"configurable": {"session_id": "abc"}},
        )

        info_dict["refined_bot_inst"] = msg.content

        return info_dict

    def get_code_summary(info_dict: dict) -> dict:
        """Get the code commands given the instructions."""

        llm = ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=4096)

        prompt = f"""
        Instructions: {info_dict["refined_bot_inst"]}
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
    image_path = r"images/browndoor_knob_pull.jpg"
    # image_path = r"images/labdoor_straighthandle_pull.jpg"
    # image_path = r"images/bluedoor_knob_push.jpg"
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

    # Run the chain
    info_dict = get_image_features(info_dict)
    info_dict = get_instructions(info_dict)
    info_dict = get_code_summary(info_dict)

    print("\n=== IMAGE FEATURES ===\n")
    print(info_dict["image_features"])
    print("\n=== HUMAN INSTRUCTIONS ===\n")
    print(info_dict["human_inst"])
    print("\n=== ROBOT INSTRUCTIONS ===\n")
    print(info_dict["bot_inst"])
    print("\n=== REFINED ROBOT INSTRUCTIONS ===\n")
    print(info_dict["refined_bot_inst"])
    print("\n=== CODE SUMMARY ===\n")
    print(info_dict["code_summary"])

    # Clear the session IDs
    store = {}


if __name__ == "__main__":
    main()

