from transformers import AutoTokenizer, LongT5ForConditionalGeneration
import torch

def summarizeLong(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    long_t5 = LongT5ForConditionalGeneration.from_pretrained(
        "Stancld/longt5-tglobal-large-16384-pubmed-3k_steps"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Stancld/longt5-tglobal-large-16384-pubmed-3k_steps"
    )
    tok_txt = tokenizer.encode(text, return_tensors="pt").to(device)
    ids = long_t5.generate(
        tok_txt,num_beams=4,max_length=200,min_length=30,
        length_penalty=2.0
    )
    output = tokenizer.decode(ids[0], skip_special_tokens=True)
    return output

with open('football.txt', 'r') as file:
    text = file.read()

summary = summarizeLong(text)
print("Summary:", summary)
