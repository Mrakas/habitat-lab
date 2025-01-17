import os
import base64
import json
from pathlib import Path
import openai
from openai import OpenAI
import time
from tqdm import tqdm
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

    def save_to_json(self, data: list, json_path: Path):
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

        existing_data.extend(data)

        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(existing_data, json_file, ensure_ascii=False, indent=4)

        print(f"Data successfully saved to {json_path}")

qa_prompt = """
    Please design seven questions based on the picture and the following template: only one question per type is needed.please return a json file for me:
"Functional reasoning":{
    "question": "Where can I put my hat?",
    "answer": "On the hat rack",
    "question": "Where can I put my apples?",
    "answer": "Use the basket on the kitchen counter"}

"Object state recognition":{
    "question": "Is the suitcase on the floor open or closed?",
    "answer": "Closed",
    "question": "Is the desk clean or full of things?",
    "answer": "Full of things"}

"Spatial Reasoning":{
    "question": "What is under the hand towel?",
    "answer": "a stool",
    "question": "What is behind to box of Coca Cola?",
    "answer": "a toaster"}

"Attribute Recognition":{
    "question": "What color is the car?",
    "answer": "Blue",
    "question": "What animal is shown in the picture in the bedroom?",
    "answer": "A bird"}

"Object Localization":{
    "question": "Where is my room key?",
    "answer": "on the desk",
    "question": "Where is the Kleenex box?",
    "answer": "Over the chest of drawers"}

"Object Recognition":{
    "question": "what is on the chair?",
    "answer": "a soft pillow",
    "question": "What is the black object next to the black chair?",
    "answer": "A filing cabinet"}

"World Knowledge":{
    "question": "I don't have a news paper, how can I check the news",
    "answer": "use the tv",
    "question": "How should I clean up a water spill?",
    "answer": "use the paper towel on the dining table"}"
    """

def main():
    # Configuration
    API_KEY = "sk-IxyZ12cYdsxsvUCnD31eC59aFc1546Df8378302237125401"
    BASE_URL = "https://api3.apifans.com/v1"
    IMAGE_FOLDER = "/mnt/data5/ghx/workplace/habitat-lab/data/collect_data/last_frame_test"
    JSON_PATH = "/mnt/data5/ghx/workplace/habitat-lab/data/collect_data/json/QA_data_val_unseen.json"
    PROMPT = qa_prompt
    MODEL = "gpt-4o-2024-08-06"
    START_IDX = 0
    END_IDX = 2000
    # Init
    processor = OpenAIImageProcessor(api_key=API_KEY, base_url=BASE_URL)

    # Process all images in the folder
    for item in tqdm(os.listdir(IMAGE_FOLDER)[START_IDX:END_IDX], desc="Processing images"):
        start_time = time.time()  # Start timing
        #import ipdb; ipdb.set_trace()
        image_path = os.path.join(IMAGE_FOLDER, item)
        # Process image
        image_data = processor.process_image(image_path=Path(image_path), prompt=PROMPT, model=MODEL)
        # Save results to JSON
        processor.save_to_json(data=[image_data], json_path=Path(JSON_PATH))
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time  # Calculate elapsed time
        
        print(f"Processed image: {image_path} | Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
