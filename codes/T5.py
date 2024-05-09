from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
def summarize(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    tok_txt = tokenizer.encode(text, return_tensors="pt").to(device)
    summary_ids = model.generate(input_ids=tok_txt,
                                    num_beams=4,min_length=30,max_length=200,
                                    length_penalty=2.0)
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return output

with open('football.txt', 'r') as file:
    text = file.read()

# Perform abstractive summarization
summary = summarize(text)
print("Summary:", summary)
