import torch
from datasets import load_dataset

dataset = load_dataset(
    "monology/pile-uncopyrighted" ,
    split="train",
    streaming=True
)

for i,sample in enumerate(dataset):  
    text=sample['text']
    text=text.split("\n").split(".").split
    print(list(text))
    if i==2:
        break
