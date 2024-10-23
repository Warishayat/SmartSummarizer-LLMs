# Text Summarization Model


### Model and Dataset Information

This project utilizes the **[google-t5-finetuning-text-summarization](https://huggingface.co/Waris01/google-t5-finetuning-text-summarization)** model from Hugging Face, trained on the **microsoft/MeetingBank-QA-Summary** dataset for effective text summarization.

This repository contains a text summarization model built using Hugging Face Transformers. The model is designed to generate concise and coherent summaries from longer text inputs, enhancing information retrieval and comprehension.

## Features

- **Text Summarization**: Automatically generates summaries for various text inputs.
- **ROUGE Evaluation**: Includes evaluation metrics to assess the quality of generated summaries.

## Model Details

- **Model Type**: T5 (Text-to-Text Transfer Transformer)
- **Language**: English
- **Training Data**: The model has been fine-tuned on a dataset that includes healthcare and finance-related texts.

## Getting Started

### Installation

To install the required packages, use the following command:

```bash
pip install transformers rouge-score
```

### Usage

You can utilize the model by importing it and using the provided methods for text summarization. Here’s an example of how to generate a summary:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("your_model_name")
model = AutoModelForSeq2SeqLM.from_pretrained("your_model_name")

# Input text
text = "In healthcare, AI systems are used for predictive analytics, improving diagnostics, and personalizing treatment plans."

# Tokenization
inputs = tokenizer(text, return_tensors="pt")

# Generate summary
summary_ids = model.generate(inputs["input_ids"])
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(summary)
```

### ROUGE Evaluation

To evaluate the model's summaries, the ROUGE scoring system is implemented. Below is the code used for evaluation:

```python
from rouge_score import rouge_scorer

reference_summaries = [
    "AI systems in healthcare improve diagnostics and personalize treatments.",
    "Algorithms analyze market trends and help in fraud detection.",
]

generated_summaries = [
    "In healthcare, AI systems are used for predictive analytics and improving diagnostics.",
    "In finance, algorithms analyze market trends and assist in fraud detection."
]

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

for reference, generated in zip(reference_summaries, generated_summaries):
    scores = scorer.score(reference, generated)
    print(f"Reference: {reference}")
    print(f"Generated: {generated}")
    print(f"ROUGE Scores: {scores}\n")
```

## Evaluation Results

The evaluation of generated summaries produced the following ROUGE scores:

1. **For healthcare-related text**:
   - **ROUGE-1**: Precision: 58.33%, Recall: 77.78%, F1-Score: 66.67%
   - **ROUGE-2**: Precision: 27.27%, Recall: 37.50%, F1-Score: 31.58%
   - **ROUGE-L**: Precision: 33.33%, Recall: 44.44%, F1-Score: 38.10%

2. **For finance-related text**:
   - **ROUGE-1**: Precision: 72.73%, Recall: 88.89%, F1-Score: 80.00%
   - **ROUGE-2**: Precision: 60.00%, Recall: 75.00%, F1-Score: 66.67%
   - **ROUGE-L**: Precision: 72.73%, Recall: 88.89%, F1-Score: 80.00%

These scores indicate a strong overlap between the generated summaries and reference summaries, showcasing the model's effectiveness.

## Model Card

For more information about the model, including its specifications and evaluation metrics, visit the Hugging Face model card [here]((https://huggingface.co/Waris01/google-t5-finetuning-text-summarization)).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co) for providing the Transformers library.
- [ROUGE](https://github.com/google-research/google-research/tree/master/rouge) for evaluation metrics.
