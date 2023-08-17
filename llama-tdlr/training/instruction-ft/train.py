from trl import SFTTrainer
from datasets import load_dataset
from random import randrange
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, AutoTokenizer

dataset = load_dataset("databricks/databricks-dolly-15k")

def format_instruction(sample):
    if sample['context'] == '':
        sample['text'] = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
        ### Instruction:
        {sample['instruction']}

        ### Response:
        {sample['response']}
        """
    else:
        sample['text'] = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
        ### Instruction:
        {sample['instruction']}

        ### Input:
        {sample['context']}

        ### Response:
        {sample['response']}
        """
    return sample


dataset = dataset.map(format_instruction)
print(f"dataset size: {len(dataset)}")
print(dataset[randrange(len(dataset))])

model_name = "meta-llama/Llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

args = TrainingArguments(
    output_dir="llama-2-dolly",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    save_total_limit=2,
    save_steps=1000,
)

max_seq_length = 2048  # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    args=args,
    dataset_text_field="text",
)

trainer.train()

trainer.save_model()
