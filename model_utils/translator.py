from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def initialize_translator():
    model_name = "facebook/nllb-200-1.3B"
    src_lang = "eng_Latn"    # English
    tgt_lang = "tha_Thai"    # Thai (FLORES‑200 BCP‑47 code)
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("cuda").eval()
    return model , tokenizer

def TranslateTh2EN (model , tokenizer , text) : 
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    forced_id = tokenizer.convert_tokens_to_ids("tha_Thai")

    outputs = model.generate(**inputs, forced_bos_token_id=forced_id, max_new_tokens=512)  
    translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return translation