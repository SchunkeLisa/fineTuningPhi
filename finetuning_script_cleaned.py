import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import prepare_model_for_kbit_training,LoraConfig, get_peft_model, PeftModel
from accelerate import Accelerator
import time
from trl import SFTTrainer, SFTConfig
from huggingface_hub import notebook_login


model_name = "microsoft/Phi-3-mini-128k-instruct"
#model_name = "microsoft/Phi-3-mini-4k-instruct" # phi
#model_name = "meta-llama/Llama-2-7b-chat-hf" # lama

new_model_name = "phi-3-128k-finetuned-whole-dataset" # phi
#new_model_name = "llama-2-finetuned-large-dataset" # lama

hub_repo_name = "LisaSchunke/phi-3-128k-peft-finetuned-large-dataset" # phi
#hub_repo_name = "LisaSchunke/llama-2-7b-peft-finetuned-large-dataset" # lama

target_modules = ['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj'] #print(model) will show the modules to use

#target_modules = ["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj","lm_head"]



def load_model_and_tokenizer(model_name):
    # Configuration to load model in 4-bit quantized
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_quant_type='nf4',
                                    bnb_4bit_compute_dtype='float16',
                                    #bnb_4bit_compute_dtype=torch.bfloat16,
                                    #bnb_4bit_use_double_quant=True
                                    )
    #Loading Microsoft's Phi-2 model with compatible settings
    #Remove the attn_implementation = "flash_attention_2" below to run on T4 GPU
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto',
                                                quantization_config=bnb_config,
                                                torch_dtype=torch.bfloat16,
                                                #attn_implementation="flash_attention_2",
                                                trust_remote_code=True, token=True)

    # Setting up the tokenizer for Phi-2
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            #add_eos_token=True,
                                            trust_remote_code=True,model_max_length=2048,token=True)
    #tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.truncation_side = "left"

    #tokenizer.pad_token = tokenizer.unk_token # phi
    tokenizer.pad_token = tokenizer.eos_token # lama
    tokenizer.padding_side = "right"

    # prepare model for lora training
    # gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # freeze base model layers and cast layernorm in fp32
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    config = LoraConfig(
    r=8, #16 is also a typical value -> the higher the better the finetuning but the more memory usage required
    lora_alpha=32,
    target_modules= target_modules,
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    )

    lora_model = get_peft_model(model, config)
    accelerator = Accelerator()
    lora_model = accelerator.prepare_model(lora_model)

    return lora_model, tokenizer


def load_dataset_custom():
    #Load a slice of the WebGLM dataset for training and merge validation/test datasets
    #train_dataset_soll_zwei = load_dataset("prsdm/medquad-phi2-1k", split="train")
    #train_dataset_soll = load_dataset("philschmid/dolly-15k-oai-style")
    train_data = load_dataset("LisaSchunke/finetuning_dataset_TSAC", split="train",token=True)
    test_data = load_dataset("LisaSchunke/finetuning_dataset_TSAC", split="test",token=True)

    return train_data, test_data


def train(lora_model, tokenizer, train_data, test_data):   

    training_args = SFTConfig(
    overwrite_output_dir=True, # Overwrite the content of the output directory
    per_device_train_batch_size=2,  # Batch size for training
    per_device_eval_batch_size=2,  # Batch size for evaluation
    gradient_accumulation_steps=5, # number of steps before optimizing
    gradient_checkpointing=True,   # Enable gradient checkpointing
    #gradient_checkpointing_kwargs={"use_reentrant": False}, hat nicht funktioniert bei mir
    warmup_steps=10,  # Number of warmup steps
    #max_steps=1000,  # Total number of training steps
    num_train_epochs=1,  # Number of training epochs
    learning_rate=5e-5,  # Learning rate
    weight_decay=0.01,  # Weight decay
    #optim="paged_adamw_8bit", #Keep the optimizer state and quantize it
    fp16=True, #Use mixed precision training
    #For logging and saving
    logging_dir='./logs',
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="steps",
    save_steps=10,
    save_total_limit=2,  # Limit the total number of checkpoints
    eval_strategy="steps",
    eval_steps=10,
    load_best_model_at_end=True, # Load the best model at the end of training
    output_dir='./results',  # Output directory for checkpoints and predictions
    #output_dir='./results',  # Output directory for checkpoints and predictions
    #optim="paged_adamw_8bit",  # Keep the optimizer state and quantize it
    packing=True,
    dataset_text_field='text',
    #max_seq_length=2048,
    #model_max_length=2048,
    max_seq_length=2048,
    )

    # Define SFT configuration
    peft_config = SFTConfig(
        output_dir='./results',  # Output directory for checkpoints and predictions
        optim="paged_adamw_8bit",  # Keep the optimizer state and quantize it
        packing=True,
        #dataset_text_field="text",
        max_seq_length=2048,
    )

    #sft_config = SFTConfig(packing=True,output_dir="/tmp")
    trainer = SFTTrainer(
        model=lora_model,
        args=training_args,
        tokenizer=tokenizer,
        #model=model,
        #peft_config=peft_config,
        train_dataset=train_data,
        eval_dataset=test_data,
        packing=True,
        #callbacks=[tensorboard_callback]
    )
    #Disable cache to prevent warning, renable for inference
    #model.config.use_cache = False

    start_time = time.time()  # Record the start time
    trainer.train()  # Start training
    end_time = time.time()  # Record the end time

    training_time = end_time - start_time  # Calculate total training time

    print(f"Training completed in {training_time} seconds.")

    return trainer
    

def save_and_upload_model(trainer, new_model_name, hub_repo_name):
    trainer.save_model(new_model_name)

    #new_model_name = "LisaSchunke/phi-3-peft-finetuned-large-dataset"
    base_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,token=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True,token=True)

    model = PeftModel.from_pretrained(base_model, new_model_name)
    model = model.merge_and_unload()
    notebook_login()
    model.push_to_hub(hub_repo_name,token=True)
    tokenizer.push_to_hub(hub_repo_name,token=True)


if __name__ == "__main__":
    lora_model, tokenizer = load_model_and_tokenizer(model_name)
    train_data, test_data = load_dataset_custom()
    trainer = train(lora_model, tokenizer, train_data, test_data)   
    save_and_upload_model(trainer, new_model_name, hub_repo_name)

