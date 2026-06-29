import torch
from datasets import load_dataset
from dataset.load_medical_o1_dataset import get_medical_resoning_sft
from trainer.sft_trainer import get_trainer
from trainer.custom_checkpoint import SaveEveryNEpochsCallback
from transformers import AutoModelForCausalLM, AutoTokenizer


def train(MODEL_NAME, DATASET_NAME):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    
    dataset = load_dataset(DATASET_NAME, "en")

    # 1. load dataset
    train_ds, eval_ds = get_medical_resoning_sft(dataset, tokenizer)

    #2. load custom checkpoint trainer callback
    custom_callback = SaveEveryNEpochsCallback( 
        n=3,
        output_dir="./lora_out",
        tokenizer=tokenizer
        )

    # 2. load sft trainer
    trainer = get_trainer(model, train_ds, custom_callback)

    trainer.train()


if __name__ == "__main__":
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"   # ~1GB
    DATASET_NAME = "FreedomIntelligence/medical-o1-reasoning-SFT"

    train(MODEL_NAME, DATASET_NAME)
