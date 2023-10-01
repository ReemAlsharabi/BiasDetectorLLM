from transformers import GPT2Model, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

def preprocess_text(text):
    tokenized = tokenizer(text, padding="max_length", max_length=128, return_tensors="pt")
    input_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']
    return input_ids, attention_mask