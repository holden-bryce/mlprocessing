from transformers import LayoutLMTokenizer

local_path = "C:/cmonpoppy/models/layoutlm-base-uncased"

try:
    tokenizer = LayoutLMTokenizer.from_pretrained(local_path)
    print("Tokenizer loaded successfully!")
except Exception as e:
    print(f"Failed to load tokenizer: {e}")
