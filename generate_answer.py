import cv2
import os
os.environ["OMP_NUM_THREADS"] = "16"
import json
import torch
import gc
import time
import string
import math
import random
import argparse
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import multiprocessing
from PIL import Image
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from modeling_llm_retrieval import RetrievalModel
from transformers import AutoTokenizer, AutoModel


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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


def get_top_k_indices(scores, k):
    sorted_indices = np.argsort(scores)[::-1]
    top_k_indices = sorted_indices[:k]
    return top_k_indices

def generate_retrieval_result(basemodel_path: str,
                              document_path: str,
                              question: str,
                              top_k: int,
                              device: str):
    llm_model = AutoModel.from_pretrained(
        basemodel_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval()
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["self_attn.qkv_proj", "self_attn.o_proj"],
        bias="none",
        task_type="SEQ_CLS"
    )
    llm_with_lora = get_peft_model(llm_model, lora_config)
    for param in llm_model.parameters():
        param.requires_grad = False
    for name, param in llm_model.named_parameters():
        if "lora" in name:
            param.requires_grad = True
    tokenizer = AutoTokenizer.from_pretrained(basemodel_path, trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = 'left'
    retrieval_model = RetrievalModel(llm_model,
                                     tokenizer,
                                     image_size=448,
                                     patch_size=14,
                                     downsample_ratio=0.5,
                                     ).to(device)
    state_dict = torch.load('model.pth', map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    retrieval_model.load_state_dict(state_dict)
    retrieval_model.eval()

    chunk_height = 500
    doc_images = os.listdir(document_path)
    all_page_clips = []
    clip_page_pos = []
    all_page_positions = []
    all_page_ids = []
    clip_page_ids = []
    clip_bounds = []
    for page_idx, page_id in enumerate(doc_images):
        image = cv2.imread(os.path.join(document_path, page_id))
        image_height = image.shape[0]
        page_clip_num = image_height // chunk_height
        for i in range(page_clip_num):
            all_page_clips.append(image[i * chunk_height:(i + 1) * chunk_height])
            clip_page_pos.append(page_idx + 1)
            all_page_positions.append(
                torch.tensor([[(i * chunk_height) / image_height,
                               (i + 1) * chunk_height / image_height]], dtype=torch.bfloat16))
            all_page_ids.append(torch.tensor([(page_idx + 1) / len(doc_images)], dtype=torch.bfloat16))
            clip_page_ids.append(page_id)
            clip_bounds.append((i * chunk_height, (i + 1) * chunk_height))
        if image_height - page_clip_num * chunk_height > 10:
            all_page_clips.append(image[page_clip_num * chunk_height:])
            all_page_positions.append(
                torch.tensor([[(page_clip_num * chunk_height) / image_height, 1.0]], dtype=torch.bfloat16))
            all_page_ids.append(torch.tensor([(page_idx + 1) / len(doc_images)], dtype=torch.bfloat16))
            clip_page_ids.append(page_id)
            clip_bounds.append((page_clip_num * chunk_height, image_height))
    scores = []
    # save image to tmp path
    temp_files = []
    timestamp = int(time.time())
    for i in range(len(all_page_clips)):
        random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        temp_file = f'{timestamp}_{i}_{random_str}.jpg'
        cv2.imwrite(f'/tmp/{temp_file}', all_page_clips[i])
        temp_files.append(temp_file)
    for i in range(len(all_page_clips)):
        images = [f"/tmp/{temp_files[i]}"]
        with torch.no_grad():
            batch_score = retrieval_model([question], images, device, all_page_positions[i].to(device),
                                          all_page_ids[i].to(device))
        scores.extend(batch_score.cpu().tolist())
    for temp_file in temp_files:
        os.remove(f'/tmp/{temp_file}')
    top_k_indices = get_top_k_indices(scores, top_k)
    top_k_page_ids = [clip_page_ids[i] for i in top_k_indices]
    return top_k_page_ids


def generate_answer(basemodel_path: str, document_path: str, top_k_page_ids: list, question: str, device: str):
    llm_qa = AutoModel.from_pretrained(
        basemodel_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(basemodel_path, trust_remote_code=True, use_fast=False)
    top_k_page_ids = list(set(top_k_page_ids))
    top_k_page_ids.sort()
    pixel_values = []
    for page_id in top_k_page_ids:
        pixel_values.append(load_image(os.path.join(document_path, page_id), max_num=12).to(torch.bfloat16).to(device))
    pixel_values = torch.cat(pixel_values, dim=0)
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    question_text = (f'<image>\n{question}, just output answer, no description.')
    response = llm_qa.chat(tokenizer, pixel_values, question_text, generation_config)
    return response



if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    ########### replace with your own data ############
    base_model_path = "/home/share_weight/InternVL2-4B"
    document_path = "./data/demo_doc"
    question = "What is the name of the institute mentioned in the title?"
    top_k = 5
    ########### replace with your own data ############

    top_k_page_ids = generate_retrieval_result(base_model_path, document_path, question, top_k, device)
    print(top_k_page_ids)
    answer = generate_answer(base_model_path, document_path, top_k_page_ids, question, device)
    print(answer)