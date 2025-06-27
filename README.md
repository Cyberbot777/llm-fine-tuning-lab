# LLM Fine-Tuning Lab

This project is a personal sandbox for learning how to load, generate, and fine-tune language models using HuggingFace Transformers and PyTorch.

**Status:** Fine-tuning completed on a small oracle-style dataset using DistilGPT2.

---

## Getting Started

### 1. Activate the virtual environment:
```bash
source .venv/Scripts/activate
```

### 2. Open the notebook:
```
fine_tune_playground.ipynb
```

### 3. Select the correct kernel:
Choose the Python environment associated with your `.venv` or named `llm-finetune-env`.

### 4. Run the following cell to load the base model and generate text:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from IPython.display import Markdown, display
import textwrap

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

output = generator(
    "Once upon a time,",
    max_new_tokens=60,
    do_sample=True,
    top_k=50,
    temperature=0.9
)

display(Markdown(output[0]["generated_text"]))
```

---

## Fine-Tuning the Model

This project includes training a DistilGPT2 model on a short oracle-style prompt dataset.

### Steps to fine-tune:
1. Load your `.txt` dataset using HuggingFace `datasets`
2. Tokenize the dataset with GPT2 tokenizer
3. Train using `Trainer` with causal language modeling config
4. Save the model checkpoint to `./oracle-gpt2/`

**Note:** This folder is `.gitignore`d to keep the repo light.

---

## Generation Parameters

| Parameter         | Description |
|------------------|-------------|
| `max_new_tokens` | Number of new tokens to generate |
| `do_sample`      | Enables randomness for creative output |
| `top_k`          | Limits choices to top-k most likely tokens |
| `temperature`    | Controls randomness (higher = more creative) |

### Examples:
```python
# Focused
generator(prompt, max_new_tokens=50, do_sample=True, top_k=10, temperature=0.5)

# Balanced
generator(prompt, max_new_tokens=60, do_sample=True, top_k=50, temperature=0.9)

# Wild
generator(prompt, max_new_tokens=100, do_sample=True, top_k=100, temperature=1.2)
```

---

## Next Steps

- Test the fine-tuned model output
- Compare pre-finetune vs. post-finetune generations
- Train on larger or more structured datasets
- Upload to HuggingFace Hub (optional)