def process_image_with_prompt(files, api_key="sk-dnxvrnlexnzxlmskkgzqvvktwgbbbcmpldrudqazuaodwyqo", model="Qwen/Qwen2-VL-72B-Instruct"):
    """
    Process an image with a text prompt using the Silicon Flow API.
    
    Args:
        files: Dictionary containing the image file and prompt
        api_key: API key for authentication
        model: Model to use for processing
        
    Returns:
        The text response from the API
    """
    import requests
    import json
    from PIL import Image
    import io
    import base64
    
    # Extract image and prompt from files
    image_file = files['image'][1]
    prompt = files['prompt'][1]
    
    # Convert image to base64
    image_bytes = image_file.read()
    image_file.seek(0)  # Reset file pointer for potential reuse
    
    # Open image and convert to webp format
    img = Image.open(io.BytesIO(image_bytes))
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='webp')
    byte_arr = byte_arr.getvalue()
    base64_image = base64.b64encode(byte_arr).decode('utf-8')
    
    # Prepare API request
    url = "https://api.siliconflow.cn/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "stream": False,
        "max_tokens": 512,
        "stop": None,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"}
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Make API request
    response = requests.request("POST", url, json=payload, headers=headers)
    response_data = json.loads(response.text)
    
    # Extract and return content
    return response_data['choices'][0]['message']['content']

print(process_image_with_prompt)

