
# LLama TL;DR

---

## Overview

This repo demonstrates the process of finetuning an existing pre-trained model (Llama-2-7b-hf) on a new dataset (`databricks-dolly-15k`) using three different approaches: Alpaca, Lora PEFT, and QLora PEFT. Post training, the script showcases how to perform inference using the trained model.

---

## Setup

Before running the code, ensure you have the necessary packages installed. You can install them using the pip command provided at the beginning of each section.

---

## Sections

### 1. Alpaca Format

1. **Loading Dataset**: The dataset `databricks/dolly-15k` is loaded using the `datasets` library.

2. **Formatting**: The samples in the dataset are formatted to include instructions, context (if available), and the response. The purpose of the formatting is to make the data more consistent and user-friendly for the model training process.

3. **Model & Tokenizer Initialization**: The pre-trained model and its corresponding tokenizer are loaded. The padding token and side are adjusted to suit the model's requirements.

4. **Training Setup**: Training arguments are defined using the `TrainingArguments` class from the `transformers` library.

5. **Training**: The model is trained using the `SFTTrainer` class.

### 2. Lora PEFT

This section is similar to the Alpaca format, with additional steps to integrate the PEFT (Parameter Efficient Fine-Tuning) technique using LoRA (Low Rank Adaptation).

1. **Model Modification for LoRA**: The model is adjusted to include the LoRA layers and the necessary configurations are set.

2. **Training Callbacks**: Two callbacks are defined:
   - `SavePeftModelCallback`: To save the PEFT model during training checkpoints.
   - `LoadBestPeftModelCallback`: To load the best PEFT model after training completes.

3. **Training**: The model is trained using the `SFTTrainer` class with the defined callbacks.

4. **Model Merging & Saving**: Post training, the LoRA layers and the base model are merged into a single model and then saved.

### 3. QLora PEFT

This section extends the Lora PEFT approach by incorporating quantization techniques (using 4-bit integers) to reduce the model's memory footprint.

1. **Model Quantization**: The model is prepared for 4-bit quantization using the `BitsAndBytesConfig`.

2. **Training & Model Saving**: The training and model saving steps are similar to the Lora PEFT section but tailored for the quantized model.

### 4. Inference

1. **Installing Inference Library**: The `vllm` library is installed to facilitate inference.

2. **Generating Texts**: A list of prompts is created, and the model generates corresponding responses using the defined sampling parameters.

---

## Usage

To use this code:

1. Ensure you have the necessary packages installed.
2. Define the `model_name` variable with the path to your pre-trained model.
3. Run the code sequentially, section by section.
4. After training, the inference section will generate and print text based on the provided prompts.

---

## Notes

- The `datasets` library is used for loading datasets in a convenient format suitable for training.
- The `transformers` library provides tools and utilities for working with transformer-based models.
- The PEFT technique allows for efficient fine-tuning, saving both time and resources.
- The `vllm` library facilitates easy inference using large language models.

---

## Conclusion

This documentation provides a detailed overview of the code used for finetuning and inference with different techniques. Follow the steps mentioned to train the model on your data and generate responses to your prompts.


## Ref

https://www.philschmid.de/instruction-tune-llama-2
