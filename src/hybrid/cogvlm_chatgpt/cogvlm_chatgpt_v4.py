import os

import time

from dotenv import load_dotenv

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    
    def get_image_description(info_dict: dict) -> dict:
        """Get the image features."""

        quant_config = 8

        if quant_config == 4:
            tokenizer = AutoTokenizer.from_pretrained("THUDM/cogvlm2-llama3-chat-19B-int4", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                "THUDM/cogvlm2-llama3-chat-19B-int4",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).eval()
        if quant_config == 8:
            tokenizer = AutoTokenizer.from_pretrained("THUDM/cogvlm2-llama3-chat-19B", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                    "THUDM/cogvlm2-llama3-chat-19B",
                    torch_dtype=torch.bfloat16,
                    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
            ).eval()
        else:
            tokenizer = AutoTokenizer.from_pretrained("THUDM/cogvlm2-llama3-chat-19B", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                    "THUDM/cogvlm2-llama3-chat-19B",
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
            ).eval().to("cuda")

        # Load image
        image = Image.open(info_dict["image_path"]).convert("RGB")

        query = "Describe the image."

        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=query,
            history=None,
            images=[image],
            template_version="chat",
        )

        input = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to("cuda"),
            "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to("cuda"),
            "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to("cuda"),
            "images": [[input_by_model["images"][0].to("cuda").to(torch.bfloat16)]],
        }
        gen_kwargs = {
            "max_new_tokens": 512,
            "pad_token_id": 128002,
            "do_sample": True,
            # "temperature": 0.6,
            # "top_p": 0.4,
            "top_k": 1,
        }

        with torch.no_grad():
            output = model.generate(**input, **gen_kwargs)
            output = output[:, input["input_ids"].shape[1] :]
            response = tokenizer.decode(output[0], skip_special_tokens=True)

        info_dict["image_desc"] = response

        return info_dict

    def get_image_features(info_dict: dict) -> dict:
        """Get the image features."""

        quant_config = 8

        if quant_config == 4:
            tokenizer = AutoTokenizer.from_pretrained("THUDM/cogvlm2-llama3-chat-19B-int4", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                "THUDM/cogvlm2-llama3-chat-19B-int4",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).eval()
        if quant_config == 8:
            tokenizer = AutoTokenizer.from_pretrained("THUDM/cogvlm2-llama3-chat-19B", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                    "THUDM/cogvlm2-llama3-chat-19B",
                    torch_dtype=torch.bfloat16,
                    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
            ).eval()
        else:
            tokenizer = AutoTokenizer.from_pretrained("THUDM/cogvlm2-llama3-chat-19B", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                    "THUDM/cogvlm2-llama3-chat-19B",
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
            ).eval().to("cuda")

        # Load image
        image = Image.open(info_dict["image_path"]).convert("RGB")

        # Create questions list
        questions = info_dict["relevant_qns"].split("\n")

        # Create answers list
        answers = []

        for question in questions:

            query = f"""
            You are a robotic arm positioned facing the image.
            Examine the given image and answer the following question.
            Answer according to the features present in the image.
            Ensure that your answer is ACCURATE.
            Question: {question}
            """

            input_by_model = model.build_conversation_input_ids(
                tokenizer,
                query=query,
                history=None,
                images=[image],
                template_version="chat",
            )

            input = {
                "input_ids": input_by_model["input_ids"].unsqueeze(0).to("cuda"),
                "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to("cuda"),
                "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to("cuda"),
                "images": [[input_by_model["images"][0].to("cuda").to(torch.bfloat16)]],
            }
            gen_kwargs = {
                "max_new_tokens": 256,
                "pad_token_id": 128002,
                "do_sample": True,
                # "temperature": 0.6,
                # "top_p": 0.4,
                "top_k": 1,
            }

            with torch.no_grad():
                output = model.generate(**input, **gen_kwargs)
                output = output[:, input["input_ids"].shape[1] :]
                response = tokenizer.decode(output[0], skip_special_tokens=True)

            answers.append(response)
        
        answers_str = "\n".join(answers)

        info_dict["image_features"] = answers_str

        return info_dict

    def get_instructions(info_dict: dict) -> dict:
        """Get the instructions for a robot to perform the required task given an image."""

        llm = ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=4096)

        runnable_with_history = RunnableWithMessageHistory(
            llm,
            get_by_session_id,
        )

        info_dict = get_image_description(info_dict)

        prompt1 = f"""
        You have an image with this description: {info_dict["image_desc"]}
        The image depicts a scenario in which you are supposed to complete a task in.
        Given the task of: {info_dict["task"]}, what are some information essential to completing the task?
        Generate questions to obtain the desired information.
        Be extremely specific in your phrasing, ensuring that the questions are understandable by a child.
        Give only the questions; give them in a bulleted list.
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
        Imagine you are in control of a robotic arm with the following commands: {info_dict["bot_commands"]}
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

        # prompt3 = f"""
        # By referencing an observation in the image, ensure each instruction is accurate.
        # Check that each instruction is logical.
        # Does the overall flow of instructions make sense?
        # """

        # msg = runnable_with_history.invoke(
        #     [
        #         HumanMessage(
        #             content=[
        #                 {
        #                     "type": "text",
        #                     "text": prompt3,
        #                 },
        #             ]
        #         )
        #     ],
        #     config={"configurable": {"session_id": "abc"}},
        # )

        # info_dict["refined_bot_inst"] = msg.content

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
        1. move_to(x, y) # robot body moves to the coordinates
        2. arm_move_to(x, y) # robot arm moves to the coordinates
        3. grab(object) # robot arm grabs object
        4. release(object) # robot arm releases object
        5. push(object) # robot arm pushes object
        6. pull(object) # robot arm pulls object
        7. arm_rotate(angle) # robot arm rotates at that angle
    """

    # [DOORS]
    # image_path = r"images/autodoor.jpg"
    # image_path = r"images/blackdoor_handle_push.jpg"
    image_path = r"images/bluedoor_knob_push.jpg"
    # image_path = r"images/browndoor_knob_pull.jpg"
    # image_path = r"images/glassdoor_sliding.jpg"
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

    print("\n=== IMAGE DESC ===\n\n" + info_dict["image_desc"])
    print("\n=== RELEVANT QUESTIONS ===\n\n" + info_dict["relevant_qns"])
    print("\n=== IMAGE FEATURES ===\n\n" + info_dict["image_features"])
    print("\n=== ROBOT INSTRUCTIONS ===\n\n" + info_dict["bot_inst"])
    # print("\n=== REFINED ROBOT INSTRUCTIONS ===\n\n" + info_dict["refined_bot_inst"])
    print("\n=== CODE SUMMARY ===\n\n" + info_dict["code_summary"])
    print("\n===\n\nTIME TAKEN (s):", (end - start))

    # Clear the session IDs
    store = {}


if __name__ == "__main__":
    main()
