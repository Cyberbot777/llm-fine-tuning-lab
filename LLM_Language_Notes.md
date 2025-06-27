# Notes: Understanding the Language of Large Language Models (LLMs)

This file summarizes key concepts for learning how LLMs (like GPT-2) interpret, process, and generate language.

---

## 1. What Is the "Language" of an LLM?

- LLMs **do not read or write text directly**.
- Instead, they operate on **tokens**, which are text fragments (words, pieces of words, punctuation, etc.).
- Each token is assigned a unique **number ID** using a model-specific **vocabulary**.

---

## 2. Key Concepts to Learn

### Tokens
- The smallest unit of meaning a model processes.
- Can be whole words or sub-word fragments (like "un", "##der", ".", etc.).
- Examples:
  - `"Do"` → 5211
  - `"truth"` → 3872

### Tokenizer
- The tool that converts **text → token IDs**, and back again.
- Example:
  ```python
  tokenizer("The truth.") → [464, 3872, 13]
  ```

### Vocabulary (vocab.json)
- A dictionary file mapping each token to a number.
- Each model has its own tokenizer and vocab.
- Usually 32k–50k unique entries.

### input_ids
- A list of token IDs that represent a text string.
- This is what the model actually sees.

### attention_mask
- A list of 1s and 0s indicating which tokens are "real" (1) and which are padding (0).
- Used during training to ignore padded sections.

### pad_token and eos_token
- `pad_token`: Used to fill in empty space in a fixed-length input.
- `eos_token`: Marks the "end of sentence" — often used as padding in GPT2 models.

---

## 3. Tools to Explore

### Use these tokenizer methods in code:

```python
# Text → Tokens → IDs
tokenizer("Do not seek the truth.")
tokenizer.convert_tokens_to_ids(['Do', ' not', ' seek'])

# IDs → Tokens → Text
tokenizer.convert_ids_to_tokens([5211, 407, 5380])
tokenizer.decode([5211, 407, 5380])
```

---

## 4. Next Topics to Learn

- How embeddings turn token IDs into meaning
- How transformers use attention to find patterns
- What a training loop looks like
- Fine-tuning vs. prompt engineering
- What causes hallucination or drift in LLMs
- How to safely train models on custom data

---

This list will grow as you continue building and fine-tuning.