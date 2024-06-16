import os

from transformers import pipeline, BitsAndBytesConfig
import torch

from src.utils.image_util import load_image, resize_image


def main():

    def query(inputs: dict) -> dict:

        model_id = "liuhaotian/llava-v1.6-34b"

        # Configure BitsAndBytesConfig based on GPU availability
        if torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
        else:
            quantization_config = None  # or configure for CPU if necessary

        # Initialize pipeline based on GPU availability
        if torch.cuda.is_available():
            pipe = pipeline(
                "image-to-text",
                model=model_id,
                model_kwargs={"quantization_config": quantization_config},
                device=0,  # use the first GPU
            )
        else:
            pipe = pipeline(
                "image-to-text",
                model=model_id,
                model_kwargs={"quantization_config": quantization_config},
            )

        image = inputs["image"]

        prompt1 = f"""
        Observe the given image and its details.
        Provide a detailed step-by-step guide on how a human would complete the task of: {inputs["task"]}.
        Link each instruction to an observation in the image in this format: Observation - Instruction.
        """

        prompt2 = f"""
        Imagine you are in control of a robotic arm with the following commands: {inputs["bot_commands"]}
        Given the human instructions you have generated, provide a guide on how the robot would complete the task.
        """

        prompt3 = f"""
        By referencing an observation in the image, ensure each instruction is accurate. Do not make assumptions.
        Check that each instruction is logical.
        """

        user_prompt1 = "USER: <image>\n" + prompt1 + "​\nASSISTANT: "

        output1 = pipe(
            image, prompt=user_prompt1, generate_kwargs={"max_new_tokens": 4096}
        )

        user_prompt2 = (
            user_prompt1
            + output1[0]["generated_text"]
            + "\nUSER: "
            + prompt2
            + "​\nASSISTANT: "
        )

        output2 = pipe(
            image, prompt=user_prompt2, generate_kwargs={"max_new_tokens": 4096}
        )

        user_prompt3 = (
            user_prompt2
            + output2[0]["generated_text"]
            + "\nUSER: "
            + prompt3
            + "​\nASSISTANT: "
        )

        output3 = pipe(
            image, prompt=user_prompt3, generate_kwargs={"max_new_tokens": 4096}
        )

        return {"bot_inst": output3[0]["generated_text"]}

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
    # image_path = r"images\browndoor_knob_pull.jpg"
    # image_path = r"images\labdoor_straighthandle_pull.jpg"
    image_path = r"images\bluedoor_knob_push.jpg"
    # image_path = r"images\whitetable.jpg"

    resize_image(image_path, image_path)
    image = load_image({"image_path": image_path})["image"]

    # Define the task to be performed
    task = input("Enter the task to be performed: ")

    result = query(
        {
            "image": image,
            "task": task,
            "bot_commands": bot_commands,
        }
    )

    print("\n==========\n")
    print(result["bot_inst"])


if __name__ == "__main__":
    main()
