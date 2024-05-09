from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

def summarizeSeg(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Split text into chunks of maximum length 512
    size = 512
    chunks = [text[i:i+size] for i in range(0, len(text), size)]

    # Summarize each chunk
    summaries = []
    for chunk in chunks:
        tokenized_text = tokenizer.encode(chunk, return_tensors="pt").to(device)
        ids = model.generate(input_ids=tokenized_text,num_beams=4,
                                    max_length=200, min_length=30, 
                                    length_penalty=2.0)
        summary = tokenizer.decode(ids[0], skip_special_tokens=True)
        summaries.append(summary)

    # Concatenate summaries
    output = ' '.join(summaries)
    return output

with open('football.txt', 'r') as file:
    text = file.read()

summary = summarizeSeg(text)
print("Summary:", summary)
