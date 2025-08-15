from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def initialize_translator():
    model_id = "scb10x/Typhoon-translate-4b"
    tokenizer = AutoTokenizer.from_pretrained(model_id , use_fast = True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    return model , tokenizer

def TranslateTh2EN (model , tokenizer , text) : 
    messages = [
    {"role": "system", "content": "Translate the following text into Thai."},
    {"role": "user", "content": f"{text}"},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=8192,
        temperature=0.2,
    )
    response = outputs[0][input_ids.shape[-1]:]
    translation = tokenizer.decode(response, skip_special_tokens=True)
    return translation