import cv2
import os
os.environ["OMP_NUM_THREADS"] = "16"
import json
import torch
import string
import gc
import re
import random
import argparse
import numpy as np
import multiprocessing
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from modeling_llm_retrieval import RetrievalModel
from transformers import AutoTokenizer, AutoModel


def calc_similarity(basemodel_path: str,
                    images: list,
                    questions: list,
                    y_positions: torch.Tensor,
                    page_positions: torch.Tensor,
                    device: str):
    """
    Calculate the score of the answer based on the images and question.
    """
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
    with torch.no_grad():
        scores = retrieval_model(questions, images, device, y_positions, page_positions)
    scores = scores.cpu().tolist()
    return scores


if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    ########### replace with your own data ############
    base_model_path = "/home/share_weight/InternVL2-4B"
    images = ["image1.jpg", "image2.jpg"]
    images_y_pos = [[0, 400], [1100, 2000]]
    images_total_height = [2000, 2000]
    images_page_idx = [1, 1]
    document_total_page = [10, 10]
    questions = ["what is the date mentioned in this letter?", "what is the date mentioned in this letter?"]
    ########### replace with your own data ############

    images_y_pos = [[y_pos / images_total_height[i] for y_pos in images_y_pos[i]] for i in range(len(images_y_pos))]
    images_page_pos = [[images_page_idx[i] / document_total_page[i]] for i in range(len(images_page_idx))]
    images_y_pos = torch.tensor(images_y_pos, device=device, dtype=torch.bfloat16)
    images_page_pos = torch.tensor(images_page_pos, device=device, dtype=torch.bfloat16)
    scores = calc_similarity(base_model_path, images, questions, images_y_pos, images_page_pos, device)
    print(scores)  # Output the scores