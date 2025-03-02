import os
import base64
import json
from pathlib import Path
import openai
from openai import OpenAI
import time
from tqdm import tqdm
'''
v2:一条轨迹对应多个问题
v3:一条轨迹对应一个问题
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

qa_prompt = "Please describe this image in brief."

def main():
    # Configuration
    API_KEY = "sk-IxyZ12cYdsxsvUCnD31eC59aFc1546Df8378302237125401"
    BASE_URL = "https://api3.apifans.com/v1"
    IMAGE_FOLDER = "/mnt/data5/ghx/workplace/habitat-lab/AEQA/debug"
    JSON_PATH = "/mnt/data5/ghx/workplace/habitat-lab/AEQA/debugjson"
    PROMPT = qa_prompt
    MODEL = "gpt-4o-2024-08-06"
    START_IDX = 0
    END_IDX = 20000
    # Init
    processor = OpenAIImageProcessor(api_key=API_KEY, base_url=BASE_URL)

    # Process all images in the folder
    file_list = os.listdir(IMAGE_FOLDER)[START_IDX:END_IDX]
    filter_list = [f for f in file_list if not f.startswith("semantic")]
    import ipdb; ipdb.set_trace()
    
    for item in tqdm(filter_list, desc="Processing images"):
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
