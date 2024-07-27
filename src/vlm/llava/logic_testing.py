from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

import torch

from PIL import Image

import re


def main():

    # [DOORS]
    # image_path = r"images/autodoor.jpg"
    # image_path = r"images/blackdoor_handle_push.jpg"
    # image_path = r"images/bluedoor_knob_push.jpg"
    # image_path = r"images/browndoor_knob_pull.jpg"
    # image_path = r"images/glassdoor_sliding.jpg"
    # image_path = r"images/housedoor_knob_push.jpg"
    # image_path = r"images/labdoor_lever_pull.jpg"
    # image_path = r"images/metaldoor_lever_pull.jpg"
    # image_path = r"images/pinkdoor_knob_pull.jpg"
    image_path = r"images/pvcdoor_folding.jpg"

    # [MISC]
    # image_path = r"images/whitetable.jpg"
    # image_path = r"images/threat_detection.jpg"
    # image_path = r"images/fridge_lefthandle.jpg"

    model_size = 13
    quant_config = 8

    if model_size == 13:
        model_path = "llava-hf/llava-v1.6-vicuna-13b-hf" # 13b model
    elif model_size == 34:
        model_path = "llava-hf/llava-v1.6-34b-hf" # 34b model
    
    processor = LlavaNextProcessor.from_pretrained(model_path)

    if quant_config == 4:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            do_sample=True,
            temperature=0.6,
            top_p=0.4,
            top_k=1,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            device_map="cuda",
        )
    elif quant_config == 8:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            do_sample=True,
            temperature=0.6,
            top_p=0.4,
            top_k=1,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="cuda",
        )
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            do_sample=True,
            temperature=0.6,
            top_p=0.4,
            top_k=1,
            device_map="cuda",
        )

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Create questions list
    questions = [
        "I am facing the image. Is this a push or pull door or something else?",
        "Does the door have a handle? If so, what type of handle does the door have and how do I open it?",
    ]

    # Create answers list
    answers = []

    for question in questions:

        user_prompt = question

        # conversation = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": user_prompt},
        #             {"type": "image"},
        #         ]
        #     }
        # ]

        # prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        # inputs = processor(prompt, image, return_tensors="pt").to("cuda")
        # output = model.generate(**inputs, max_new_tokens=1024)
        # response = processor.decode(output[0], skip_special_tokens=True)

        if model_size == 13:
            prompt = "<image>\n" + f"USER: {user_prompt}\nASSISTANT:"
        elif model_size == 34:
            prompt = (
                "<|im_start|>user\n<image>\n"
                + user_prompt
                + "<|im_end|><|im_start|>assistant\n"
            )

        input = processor(prompt, image, return_tensors="pt").to("cuda")
        input["input_ids"][input["input_ids"] == 64003] = 64000  # temp solution
        outputs = model.generate(**input, max_new_tokens=1024)

        output = processor.decode(outputs[0], skip_special_tokens=True)

        if model_size == 13:
            match = re.search(r"ASSISTANT:\s*(.*)", output, re.DOTALL)
        elif model_size == 34:
            match = re.search(r"assistant\s*\n(.*)", output, re.DOTALL)

        if match:
            response = match.group(1).strip()

        answers.append(response)

    answers_str = "\n".join(answers)

    print("\n=== ANSWERS ===\n\n" + answers_str)


if __name__ == "__main__":
    main()
