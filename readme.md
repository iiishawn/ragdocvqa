# Source code of Position-aware Modeling for Fine-grained Long Document Understanding

## Environment Setup

* Python 3.10
* Use ms_swift > 3.0: https://github.com/modelscope/ms-swift

## Model Preparation

* Download InternVL2-4B from HuggingFace Hub: https://huggingface.co/OpenGVLab/InternVL2-4B
* Replace the file "modeling_internvl_chat.py" in the downloaded model with the file provided

## Similarity Score Calculation
* Replace the base model path where InternVL2-4B stored in "calculate_score.py"
* Replace the variable "images, images_y_pos, images_total_height, images_page_idx, document_total_page, questions" in "calculate_score.py"
* Run "calculate_score.py" to test the retrieval model

## Document QA
* Run the code in "generate_answer.py" to test the complete pipeline