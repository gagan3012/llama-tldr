# Model Merging  

To merge lora or qlora adapters use this script

for qlora use:
```bash
python merge_qlora.py --model_path <pretrained_model_path> \
                   --adapter_path <checkpoint_or_saved_adapter> \
                   --lora_method qlora \  
```

for lora use:
```bash
python merge_qlora.py --model_path <pretrained_model_path> \
                   --adapter_path <checkpoint_or_saved_adapter> \
                   --lora_method lora \  
```
