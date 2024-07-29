from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import torch

from PIL import Image


def main():

    # [DOORS]
    image_path = r"images/autodoor.jpg"
    # image_path = r"images/blackdoor_handle_push.jpg"
    # image_path = r"images/bluedoor_knob_push.jpg"
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

    quant_config = 8

    if quant_config == 4:
        tokenizer = AutoTokenizer.from_pretrained(
            "THUDM/cogvlm2-llama3-chat-19B-int4", trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            "THUDM/cogvlm2-llama3-chat-19B-int4",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()
    if quant_config == 8:
        tokenizer = AutoTokenizer.from_pretrained(
            "THUDM/cogvlm2-llama3-chat-19B", trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            "THUDM/cogvlm2-llama3-chat-19B",
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            "THUDM/cogvlm2-llama3-chat-19B", trust_remote_code=True
        )
        model = (
            AutoModelForCausalLM.from_pretrained(
                "THUDM/cogvlm2-llama3-chat-19B",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            .eval()
            .to("cuda")
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

        query = question

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
            "max_new_tokens": 1024,
            "pad_token_id": 128002,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.4,
            "top_k": 1,
        }

        with torch.no_grad():
            output = model.generate(**input, **gen_kwargs)
            output = output[:, input["input_ids"].shape[1] :]
            response = tokenizer.decode(output[0], skip_special_tokens=True)

        answers.append(response)

    answers_str = "\n".join(answers)

    print("\n=== ANSWERS ===\n\n" + answers_str)


if __name__ == "__main__":
    main()
