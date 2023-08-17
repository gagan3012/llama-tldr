from trl import SFTTrainer
from datasets import load_dataset
from random import randrange
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl
import os
import torch
from peft import AutoPeftModelForCausalLM, set_peft_model_state_dict
from transformers import AutoTokenizer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

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

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

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

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        kwargs["model"].save_pretrained(checkpoint_folder)

        pytorch_model_path = os.path.join(
            checkpoint_folder, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        return control


class LoadBestPeftModelCallback(TrainerCallback):
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        print(
            f"Loading best peft model from {state.best_model_checkpoint} (score: {state.best_metric}).")
        best_model_path = os.path.join(
            state.best_model_checkpoint, "adapter_model.bin")
        adapters_weights = torch.load(best_model_path)
        model = kwargs["model"]
        set_peft_model_state_dict(model, adapters_weights)
        return control

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    max_seq_length=max_seq_length,
    peft_config=peft_config,
    tokenizer=tokenizer,
    packing=True,
    args=args,
    dataset_text_field="text",
    callbacks=[LoadBestPeftModelCallback(), SavePeftModelCallback()],
)

trainer.train()

trainer.save_model()

tokenizer = AutoTokenizer.from_pretrained(args.output_dir)

model = AutoPeftModelForCausalLM.from_pretrained(
    args.output_dir,
    low_cpu_mem_usage=True,
)

# Merge LoRA and base model
merged_model = model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained(args.output_dir, safe_serialization=True)
tokenizer.save_pretrained(args.output_dir)
