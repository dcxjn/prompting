import os

from dotenv import load_dotenv

from langchain.globals import set_debug

from langchain_core.messages import HumanMessage

from langchain_openai import ChatOpenAI

import base64


def main():

    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    set_debug(False)

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

    # Load image
    with open(image_path, "rb") as image_file:
        image = base64.b64encode(image_file.read()).decode("utf-8")

    # Create questions list
    questions = [
        "I am facing the image. Is this a push or pull door or something else?",
        "Does the door have a handle? If so, what type of handle does the door have and how do I open it?",
    ]

    # Create answers list
    answers = []

    for question in questions:

        llm = ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=1024)

        msg = llm.invoke(
            [
                HumanMessage(
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ]
                )
            ]
        )

        answers.append(msg.content)

    answers_str = "\n".join(answers)

    print("\n=== ANSWERS ===\n\n" + answers_str)


if __name__ == "__main__":
    main()
