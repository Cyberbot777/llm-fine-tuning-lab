# LLM Fine-Tuning Lab

This project is a personal sandbox for learning how to load, generate, and fine-tune language models using HuggingFace Transformers and PyTorch.

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

### 4. Run the following cell to load the model and generate text:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from IPython.display import Markdown, display
import textwrap

# Load model and tokenizer
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set up generator
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Generate text
output = generator(
    "Once upon a time,",
    max_new_tokens=60,
    do_sample=True,
    top_k=50,
    temperature=0.9
)

# Display output
print("\nGenerated Output:\n")
display(Markdown(output[0]["generated_text"]))
```

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

## Next Steps

Fine-tuning instructions will be added soon as this lab expands.