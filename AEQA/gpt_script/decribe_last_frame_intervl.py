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
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import os 

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
torch.cuda.set_device(2)  

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
path = '/mnt/data5/ghx/workplace/Huggingface/intervl'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)


# set the max number of tiles in `max_num`
# pixel_values = load_image('/mnt/data5/ghx/workplace/habitat-lab/AEQA/debug/semantic_t1192_f37.png', max_num=12).to(torch.bfloat16).cuda()
# generation_config = dict(max_new_tokens=1024, do_sample=True)

# # single-image single-round conversation (单图单轮对话)
# question = '<image>\nPlease describe the image shortly.'
# response = model.chat(tokenizer, pixel_values, question, generation_config)
# print(f'User: {question}\nAssistant: {response}')



def main():
    # Configuration
    IMAGE_FOLDER = "/mnt/data5/ghx/workplace/habitat-lab/data/collect_data/last_frames_folder_rgb_semantic/last_frame_train_rgb_semantic"
    JSON_FOLDER  = "/mnt/data5/ghx/workplace/habitat-lab/data/collect_data/last_frames_folder_rgb_semantic/jsons"
    START_IDX = 0
    END_IDX = 99999
    # Process all images in the folder
    file_list = sorted(os.listdir(IMAGE_FOLDER)[START_IDX:END_IDX])
    #file_list = sorted(os.listdir(IMAGE_FOLDER)[START_IDX:END_IDX])[3::4]
    filter_list = [f for f in file_list if not f.startswith("semantic")]
    #import ipdb; ipdb.set_trace()
    
    for item in tqdm(filter_list, desc="Processing images"):
        start_time = time.time()  # Start timing
        image_path = os.path.join(IMAGE_FOLDER, item)

        # 构建JSON文件路径 - 使用原始图片名称但改为.json后缀
        json_filename = os.path.splitext(item)[0] + '.json'
        json_path = os.path.join(JSON_FOLDER, json_filename)

        # 如果JSON文件已存在，跳过处理
        if os.path.exists(json_path):
            continue

        # Process image
        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=1024, do_sample=True)

        # single-image single-round conversation
        question = '<image>\nPlease describe the image shortly.'
        response = model.chat(tokenizer, pixel_values, question, generation_config)


        # 准备要保存的数据
        result_data = {
            "image_path": image_path,
            "response": response
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=4)

        #import ipdb; ipdb.set_trace()
        print(f'User: {question}\nAssistant: {response}')
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time  # Calculate elapsed time
        
        print(f"Processed image: {image_path} | Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
