import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import os
import sys
import math
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

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

def get_input_for_intern_vl_2_4b(tokenizer, images, questions, num_ques_image, device, image_size, patch_size, downsample_ratio):
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'


    image_features = []
    for image in images:
        image_features.append(load_image(image, input_size=image_size, max_num=12).to(torch.bfloat16).to(device))

    num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))
    image_idx = 0
    question_with_img_token = []
    for i, question in enumerate(questions):

        for _ in range(num_ques_image[i]):
            image_batch_size = image_features[image_idx].size(0)
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * image_batch_size + IMG_END_TOKEN
            question = question.replace('<image>', image_tokens, 1)
            image_idx += 1
        question_with_img_token.append(question)
    inputs = tokenizer(question_with_img_token, return_tensors='pt', padding=True).to(device)
    image_features = torch.cat(image_features, dim=0)
    image_flags = torch.ones((image_features.size(0), 1), dtype=torch.long, device=image_features.device)
    return inputs, image_features, image_flags


class RetrievalModel(nn.Module):
    def __init__(
        self,
        llm_model,
        tokenizer,
        image_size,
        patch_size,
        downsample_ratio,
        hidden_dim=3072,
        reduced_dim=64,
        col_calc_rule='sum',
        use_y_position=True,
        use_page_position=True,
        position_emb_size=3072,
        position_calc='add',
        position_mode='add_to_col_document'
    ):
        super(RetrievalModel, self).__init__()
        self.model_main_version = '2.3.0'
        self.IMG_START_TOKEN = '<img>'
        self.IMG_END_TOKEN = '</img>'
        self.IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.col_proj = torch.nn.Linear(hidden_dim, reduced_dim, bias=False, dtype=torch.bfloat16)
        self.col_calc_rule = col_calc_rule
        self.use_y_position = use_y_position
        self.use_page_position = use_page_position
        self.position_emb_size = position_emb_size
        self.position_calc = position_calc
        self.position_mode = position_mode
        if use_y_position:
            self.y_position_embeddings = nn.Linear(2, position_emb_size, bias=False, dtype=torch.bfloat16)
        if use_page_position:
            self.page_position_embeddings = nn.Linear(1, position_emb_size, bias=False, dtype=torch.bfloat16)


    def forward(self, questions, images, device, y_positions=None, page_positions=None):
        image_text_prompt = ['<image>Describe the image'] * len(questions)
        num_ques_image = [1] * len(questions)

        image_text_input, image_text_features, image_text_flags = get_input_for_intern_vl_2_4b(self.tokenizer, images, image_text_prompt, num_ques_image, device, self.image_size, self.patch_size, self.downsample_ratio)
        question_input, question_features, question_flags = get_input_for_intern_vl_2_4b(self.tokenizer, images, questions, num_ques_image, device, self.image_size, self.patch_size, self.downsample_ratio)
        position_embedding_all = None
        if self.use_y_position and not self.use_page_position:
            position_embedding_all = self.y_position_embeddings(y_positions)
        elif not self.use_y_position  and self.use_page_position:
            position_embedding_all = self.page_position_embeddings(page_positions).unsqueeze(dim=0)
        elif self.use_y_position and self.use_page_position:
            position_embedding_all = self.y_position_embeddings(y_positions) + self.page_position_embeddings(page_positions)
        x = self.llm_model(**question_input,
                           pixel_values=question_features,
                           absolute_pos_emb=position_embedding_all,
                           position_mode=self.position_mode,
                           pure_text=True,
                           return_dict=True,
                           image_flags=question_flags,
                           img_context_token_id=self.tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN),
                           output_hidden_states=True)
        y = self.llm_model(**image_text_input,
                           pixel_values=image_text_features,
                           absolute_pos_emb=position_embedding_all,
                           position_mode=self.position_mode,
                           pure_text=False,
                           return_dict=True,
                           image_flags=image_text_flags,
                           img_context_token_id=self.tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN),
                           output_hidden_states=True)
        x = x.hidden_states[-1]
        y = y.hidden_states[-1]
        x = self.col_proj(x)
        x = F.normalize(x, p=2, dim=-1) # L2 normalize
        if (self.use_y_position or self.use_page_position) and self.position_mode == 'add_to_col_document':
            if position_embedding_all is None:
                raise ValueError('position_embedding_all should not be None when using add_to_col_document')
            position_embedding_all = position_embedding_all.unsqueeze(dim=1)
            y = y + position_embedding_all
        y = self.col_proj(y)
        y = F.normalize(y, p=2, dim=-1) # L2 normalize

        similarity_matrix = torch.matmul(x, y.transpose(-1, -2))
        if self.col_calc_rule == 'sum':
            score = similarity_matrix.max(dim=-1)[0].sum(dim=-1)
        elif self.col_calc_rule == 'mean':
            score = similarity_matrix.max(dim=-1)[0].mean(dim=-1)

        return score

    def get_main_version(self):
        return self.model_main_version