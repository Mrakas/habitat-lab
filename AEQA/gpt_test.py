import os
import base64
from pathlib import Path
from openai import OpenAI

def get_gpt_response(image_path, prompt, api_key, base_url="https://api.openai.com/v1", model="gpt-4o"):
    """
    Process the given image and return a description using OpenAI.

    Args:
        image_path (str): Path to the image file
        prompt (str): Text prompt for the model
        api_key (str): OpenAI API key
        base_url (str, optional): OpenAI base URL. Defaults to standard OpenAI URL.
        model (str, optional): Model name to use. Defaults to "gpt-4o".

    Returns:
        str: Model's response to the image and prompt
    """
    # Initialize client
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # Read and encode image
    with open(image_path, "rb") as img_file:
        img_b64_str = base64.b64encode(img_file.read()).decode("utf-8")
    
    # Get image type from file extension
    img_type = f"image/{Path(image_path).suffix.lstrip('.')}"
    
    # Send request to OpenAI
    response = client.chat.completions.create(
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
    
    # Return text response
    return response.choices[0].message.content


#Example usage:
    # API_KEY = "sk-KlQFn0xmaR52YmJNwH4hXjLbK6m4kPJo8J8JXtjAI213qgr0" # expensive
    # BASE_URL = "https://lonlie.plus7.plus/v1"
result = get_gpt_response(
    image_path="/mnt/data5/ghx/workplace/habitat-lab/AEQA/debug_molmo/test.png",
    prompt="Describe what you see in this image",
    api_key="sk-KlQFn0xmaR52YmJNwH4hXjLbK6m4kPJo8J8JXtjAI213qgr0",
    base_url="https://lonlie.plus7.plus/v1",  # Optional
    model="gpt-4o-2024-11-20"  # Optional
)

print(result)