import os
import base64
import json
from pathlib import Path
import openai
from openai import OpenAI
import time
from tqdm import tqdm
import random
from gpt_script.qwen_api import process_image_with_prompt

'''
生成的文件还需要清理后才可以使用
'''
class OpenAIImageProcessor:
    def __init__(self, api_key: str, base_url: str):
        """
        Initialize the OpenAI client with the API key and base URL.

        :param api_key: OpenAI API key.
        :param base_url: OpenAI base URL.
        """
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["BASE_URL"] = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def check_image_exists(self, image_name: str, json_path: Path) -> bool:
        """
        Check if image already exists in JSON file.
        
        :param image_name: Name of the image to check
        :param json_path: Path to the JSON file
        :return: True if image exists, False otherwise
        """
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                return any(item["image_name"] == image_name for item in existing_data)
        except (json.JSONDecodeError, FileNotFoundError):
            return False
        
    def process_image(self, image_path: Path, prompt: str, model: str) -> dict:
        """
        Process the given image and return a description using OpenAI.

        :param image_path: Path to the image file.
        :param prompt: Text prompt for the model.
        :param model: Model name to use for processing.
        :return: Dictionary containing image name and model result.
        """
        with open(image_path, "rb") as img_file:
            img_b64_str = base64.b64encode(img_file.read()).decode("utf-8")

        img_type = f"image/{image_path.suffix.lstrip('.')}"

        chat_completion = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"},
                        },
                    ],
                }
            ],
        )

        result = chat_completion.choices[0].message.content

        return {
            "image_name": image_path.name,
            "result": result,
        }

    def save_to_json(self, question_type_name, data: list, json_path: Path):
        """
        Save the processed data to a JSON file.

        :param data: List of data to save.
        :param json_path: Path to the JSON file.
        """
        json_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_data = []
        data[0]["category"] = question_type_name
        #import ipdb; ipdb.set_trace()
        existing_data.extend(data)

        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(existing_data, json_file, ensure_ascii=False, indent=4)

        print(f"Data successfully saved to {json_path}")
prompt_list = [
    'Please generate a question and answer from the picture, the question type is "functional reasoning". The following are templates you can refer to: "question": "What can I use to water my plants?", "answer": "The green hose", "question": "What device can I use to send an email?", "answer": "computer", "question": "Where can I get water?", "answer": "waterbottle", "question": "Where can I store my undergarments?", "answer": "in the drawers next to the clothing rack stand", "question": "Where I can show my presentation?", "answer": "on the television", "question": "I need to carry things to class, what can I use?", "answer": "The two black backpacks by the couch or the blue duffle bag"',
    'Please generate a question and answer from the picture, the question type is "object state recognition". The following are templates you can refer to: "question": "Is the entrance door to the room open or closed?", "answer": "open", "question": "Is the kitchen sink tap open or closed?", "answer": "closed", "question": "Is the filing cabinet closest to the white board open or closed?", "answer": "It is open", "question": "Is the closet door fully closed?", "answer": "No", "question": "Is the window open to let in light?", "answer": "Yes",',
    'Please generate a question and answer from the picture, the question type is "spatial understanding". The following are templates you can refer to: "question": "Is there space under the kitchen bar area?", "answer": "No.", "question": "What is in between the two white cabinets in the kitchen?", "answer": "The microwave", "question": "What is in between the two couches?", "answer": "A coffee table", "question": "What is above the piano?", "answer": "A painting", "question": "What appliance is below the hanging cabinets on the left side of the kitchen?", "answer": "A toaster oven", "question": "what is above the markers on the whiteboard?", "answer": "eraser"',
    'Please generate a question and answer from the picture, the question type is "attribute recognition". The following are templates you can refer to:"question": "what color are the dining room walls?", "answer": "brown", "question": "Is the room well-lit?", "answer": "No", "question": "What is the color of the computer mouse?", "answer": "white", "question": "What is the color of the vacuum cleaner?", "answer": "black", "question": "What color pattern is on the pillow? ", "answer": "checkerboard pattern", "question": "What is the color of non-black chairs?", "answer": "red",',
    'Please generate a question and answer from the picture, the question type is "object recognition". The following are templates you can refer to: "question": "Whats under the sink?", "answer": "Trash bin", "question": "What is the colorful object on the bed?", "answer": "a blue towel", "question": "What is located in the center of the room?", "answer": "tennis table", "question": "What is located on the counter on the left?", "answer": "a printer", "question": "What is located on the white desk?", "answer": "computer",',
    'Please generate a question and answer from the picture, the question type is "object localization". The following are templates you can refer to: "question": "what room is the surface cleaner in?", "answer": "the room with the whiteboard", "question": "where is the fan?", "answer": "in the dining room besides the kitchen", "question": "Where did I leave my water bottle?", "answer": "On the table next to the desk in the study.", "question": "Where is the closet with the mirror?", "answer": "In the bedroom downstairs", "question": "Where is the white sofa?", "answer": "In the living room",',
    'Please generate a question and answer from the picture, the question type is "world knowledge". The following are templates you can refer to: "question": "Is my backyard safe to let me dog out in?", "answer": "Yes, its fenced.", "question": "Can this home be used for a large dinner party?", "answer": "Yes.", "question": "Are the walls painted?", "answer": "No", "question": "Where can I get a drink of water?", "answer": "From the water dispenser in the fridge", "question": "I need to blow my nose while taking a bath, what can I use?", "answer": "There is toilet paper next to the tub.",'     
]

question_types = [
    "functional reasoning",
    "object state recognition",
    "spatial understanding",
    "attribute recognition", 
    "object recognition",
    "object localization",
    "world knowledge"
]

qa_prompt = ""

def main():
    # Configuration
    # API_KEY = "sk-IxyZ12cYdsxsvUCnD31eC59aFc1546Df8378302237125401"
    # BASE_URL = "https://api3.apifans.com/v1"

    API_KEY = "sk-KlQFn0xmaR52YmJNwH4hXjLbK6m4kPJo8J8JXtjAI213qgr0" # expensive
    BASE_URL = "https://lonlie.plus7.plus/v1"
    split_set = "train"
    IMAGE_FOLDER = f"/mnt/data5/ghx/workplace/habitat-lab/data/collect_data/last_frames_folder_rgb_semantic/last_frame_{split_set}_rgb_semantic"
    JSON_PATH = f"/mnt/data5/ghx/workplace/habitat-lab/data/collect_data/last_frames_folder_rgb_semantic/gpt_json/gpt4o_gen_last_frame_{split_set}_rgb_semantic.json"
    PROMPT = qa_prompt
    MODEL = "gpt-4o-2024-11-20"
    START_IDX = 0
    END_IDX = 99999
    # Init
    processor = OpenAIImageProcessor(api_key=API_KEY, base_url=BASE_URL)
    #semantic filter
    
    file_list = sorted(os.listdir(IMAGE_FOLDER))
    filter_list = [f for f in file_list if not f.startswith("semantic")]
    filter_list = filter_list[START_IDX:END_IDX]
    #import ipdb; ipdb.set_trace()
    # Process all images in the folder
    for item in tqdm(filter_list, desc="Processing images"):
        if processor.check_image_exists(item, Path(JSON_PATH)):
            print(f"Skipping {item} - already processed")
            continue
        time.sleep(1)
        PROMPT = random.choice(prompt_list)
        question_index = prompt_list.index(PROMPT)

        start_time = time.time()  # Start timing
        #import ipdb; ipdb.set_trace()
        question_type_name = question_types[question_index]
        image_path = os.path.join(IMAGE_FOLDER, item)
        # Process image
        image_data = processor.process_image(image_path=Path(image_path), prompt=PROMPT, model=MODEL)

        # Save results to JSON
        processor.save_to_json(question_type_name=question_type_name,data=[image_data], json_path=Path(JSON_PATH))
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time  # Calculate elapsed time
        
        print(f"Processed image: {image_path} | Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
