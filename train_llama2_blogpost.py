fimport datasets
import transformers
import peft
import torch
import trl

IDENTIFIER_LLAMA_2_7B_CHAT_HF = "meta-llama/Llama-2-7b-chat-hf"
   
   
#NOTE - following this blog post https://medium.com/@gobishangar11/llama-2-a-detailed-guide-to-fine-tuning-the-large-language-model-8968f77bcd15
def train_llama2(model_id):
    def convert_dataset(data):
        instruction = data["instruction"]
        output = data["output"]
        prompt = f"<s>[INST] {instruction} [/INST] {output} </s>"
        return {'text': prompt}
    train_dataset = datasets.load_dataset("LisaSchunke/finetuning_dataset_TSAC", split="train[0:20]")
    test_dataset = datasets.load_dataset("LisaSchunke/finetuning_dataset_TSAC", split="test[0:5]")
    print(train_dataset[:5])
    
    # Model and tokenizer names
    refined_model = "llama-2-7b-finetune-enhanced"
    # Tokenizer
    tokenizer =  transformers.AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
    
    quant_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False)
    # Load Llama2
    base_model = transformers.AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config, device_map={"": 0})
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 0
    
    peft_parameters = peft.LoraConfig(lora_alpha=16,lora_dropout=0.1,r=8,bias="none",task_type="CAUSAL_LM")
    train_params = transformers.TrainingArguments(
        output_dir=r"/out/trained_models",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=5,
        logging_steps=1,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        #report_to="tensorboard"
    )
    fine_tuning = trl.SFTTrainer(model=base_model,train_dataset=train_dataset,peft_config=peft_parameters,dataset_text_field="text",tokenizer=tokenizer,args=train_params)
    fine_tuning.train()
    fine_tuning.model.save_pretrained(refined_model)
    print(fine_tuning)
    
    # 06. Merge the Base Model with the Trained Adapter
    # Reload model in FP16 and merge it with LoRA weights
    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path= IDENTIFIER_LLAMA_2_7B_CHAT_HF,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    #Reload the Base Model and load the QLoRA adapters
    model = peft.PeftModel.from_pretrained(model, refined_model)
    model = model.merge_and_unload()

    # Reload tokenizer to save it
    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path= IDENTIFIER_LLAMA_2_7B_CHAT_HF)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # push model to hub
    hub_repo_name = "LisaSchunke/llama-2-7b-blogpost-finetuned-20000-dataset" # lama
    model.push_to_hub(hub_repo_name,token=True)
    tokenizer.push_to_hub(hub_repo_name,token=True)
    #model.save_pretrained("llama-tuned")
    #tokenizer.save_pretrained("llama-tuned")
    

train_llama2(IDENTIFIER_LLAMA_2_7B_CHAT_HF) 
