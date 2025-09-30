from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

texts = ["Hello Word, a test sentence"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tokenizer 部分
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)

features = tokenizer(
    texts[0],
    padding=True,
    truncation="longest_first",
    return_tensors="pt",
    max_length=512,
)
features.to(device)
print(features)

with torch.no_grad():
    outputs = model(**features)
    print(outputs)
