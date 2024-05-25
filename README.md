# Arabic Text Processing and Sentiment Analysis with GPT-2 & BERT

This repository showcases a project exploring Arabic text processing and sentiment analysis using advanced deep learning models: GPT-2 and BERT. The goal is to demonstrate how these models can be effectively adapted to analyze and generate Arabic text.

## Project Overview

The project is divided into three parts:

### Part 1: Classification and Regression

#### Data Collection and Dataset Preparation
- **Web Scraping for Moroccan Sports Data:** Scraped Arabic text data from the website [Akhbarona](https://www.akhbarona.com/sport/articles/) using the Scrapy framework. This involved exploring website structures and extracting the desired content.
- **Creating a Relevance Score using TF-IDF:** Assigned a relevance score (between 0 and 10) to each scraped text using the TF-IDF technique. This required defining keywords related to Moroccan sports, calculating the TF-IDF weights for those keywords in each text, and normalizing the weights to create the "Score" column.
- **Dataset Structure:** The scraped data was organized into a structured dataset, including a "Text" column for the Arabic text and a "Score" column representing the relevance scores.

#### Preprocessing Arabic Text
- - **Cleaning:** The text is cleaned by removing unwanted elements like mentions, URLs, punctuations, and arabic characters.
- **Normalization:** Different forms of Arabic letters are unified for consistency.
- **Lemmatization:** The `Qalsadi lemmatizer` is employed to reduce words to their base forms, capturing the core meaning.
- **Stop Words Removal:** Common, less informative words are removed to focus on crucial terms.
- **Tokenization:** The text is segmented into individual tokens (words or subwords) using NLTK's `word_tokenize`.
- **Padding:** To ensure consistent input sequence lengths for our models, padding is applied to sequences using Keras' `pad_sequences`.

#### Training RNN Models
- **RNN Architectures:** Explored different RNN architectures for both classification and regression:
  - **GRU:** Gated Recurrent Unit, a more efficient variant of Bidirectional RNNs with gates controlling information flow.
  - **LSTM:** Long Short-Term Memory, another variant of RNNs with sophisticated memory mechanisms for handling long-range dependencies in text.
- **Model Implementation:** Implemented these architectures using PyTorch, defining the model structure (embedding layers, RNN layers, fully connected layers) and specifying hyperparameters like embedding dimension, hidden dimension, number of layers, and dropout rate.
- **Hyperparameter Tuning:** Fine-tuned the hyperparameters of the RNN models to achieve optimal performance, systematically experimenting with different values.

#### Evaluating the Models
- **Standard Metrics:** Evaluated the performance of the models using:
  - **Accuracy:** The percentage of correctly classified instances for classification models.
  - **Mean Squared Error (MSE):** Measures the average squared difference between predicted and actual values for regression models.
  - **Mean Absolute Error (MAE):** Measures the average absolute difference between predicted and actual values for regression models.
  - **R-squared (R²):** Indicates the proportion of variance in the target variable explained by the model.
  - **BLEU Score:** Computed the BLEU score to evaluate the quality of machine translation or text generation, measuring the similarity between generated text and reference text.

### Part 2: Transformer-Based Text Generation (GPT-2)

#### Fine-tuning GPT-2
- **Loading a Pre-trained Model:** Loaded the pre-trained `GPT-2-medium` model from the Hugging Face Transformers library.
- **Custom Dataset Preparation:** Created a custom dataset related to Morocco, including topics such as life in Morocco, education, football, how people live, and geography.
- **Fine-tuning with PyTorch:** Fine-tuned the GPT-2 model using PyTorch, involving:
  - **Setting up the Training Environment:** Defined training arguments (batch size, number of epochs, learning rate, etc.) and created a Trainer object to manage the fine-tuning process.
  - **Implementing a Training Loop:** Wrote a training loop that iterates over the dataset, feeds the input sequences to the model, calculates the loss, performs backpropagation, and updates the model weights.
  - **Saving the Fine-tuned Model:** Saved the fine-tuned GPT-2 model and its tokenizer for future use.

#### Text Generation
- **Generating New Text:** Used the fine-tuned GPT-2 model to generate new text based on a given sentence (prompt), involving:
  - **Setting Generation Parameters:** Experimented with parameters like max_length, do_sample, top_k, temperature, num_beams, and attention_mask to control the generated text's length, creativity, and coherence.
  - **Decoding the Output:** Decoded the model's output (a sequence of tokens) back into human-readable text.

### Part 3: BERT-based Sentiment Classification

#### Loading a Pre-trained BERT Model
- **Hugging Face Transformers:** Utilized the Hugging Face Transformers library to load the pre-trained `BERT-base-uncased` model.

#### Preparing Data for BERT
- **Dataset Preparation:** Prepared the Amazon Video Game Reviews dataset (available here), including:
  - **Handling Missing Data:** Dealt with any missing entries in the dataset.
  - **Transforming Labels:** Ensured that the sentiment labels were appropriately represented (e.g., 0 for negative, 1 for positive).
- **Bert Embedding Layer Adaptation:** Adapted BERT's embedding layer to the specific vocabulary, involving:
  - **Tokenization:** Used BERT's tokenizer to convert the text into a sequence of tokens.
  - **Padding:** Padded sequences to a consistent length using the tokenizer's `pad_token_id`.

#### Fine-tuning and Training BERT
- **Fine-tuning for Sentiment Classification:** Fine-tuned the BERT model to classify sentiment, adapting its weights to learn the patterns, involving:
  - **Training Arguments:** Defined training arguments (batch size, epochs, learning rate, etc.) for the fine-tuning process.
  - **Training Loop:** Implemented a training loop that feeds the tokenized input sequences and sentiment labels to BERT, calculates the loss, performs backpropagation, and updates the model weights.
  - **Hyperparameter Tuning:** Experimented with different hyperparameters to find the combination that yielded the best performance for sentiment classification.

#### Evaluation
- **Standard Metrics:** Evaluated the performance of the fine-tuned BERT model using:
  - **Accuracy:** The proportion of correctly classified reviews.
  - **Loss:** A measure of the difference between predicted and actual sentiment labels.
  - **F1-score:** A harmonic mean of precision and recall, providing a balanced metric.
- **Advanced Metrics:** Explored other metrics like BLEU score and BERTScore to assess the quality of the sentiment predictions.

### General Results for BERT Model:

```json
{
 'eval_loss': 0.512526273727417,
 'eval_accuracy': 0.7925,
 'eval_recall': 0.7925,
 'eval_precision': 0.62805625,
 'eval_f1': 0.7007601115760111,
 'eval_bleu': 0.14092864324558466,
 'eval_bert_precision': 0.9959251880645752,
 'eval_bert_recall': 0.9959251880645752,
 'eval_bert_f1': 0.9959251880645752,
 'eval_runtime': 806.2115,
 'eval_samples_per_second': 2.481,
 'eval_steps_per_second': 0.31,
 'epoch': 3.0
}
```
# Running the Code

Each file represents a different part of the project. You can run the files separately in a Jupyter Notebook environment. Ensure you have the necessary libraries installed.

## Dependencies

- `python3`
- `nltk`
- `pandas`
- `pymongo`
- `transformers`
- `torch`
- `keras`
- `nltk.translate.bleu_score`
- `bert_score`
- `qalsadi`

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```
# Usage

1. Open each Jupyter Notebook file.
2. Run the cells sequentially.

# Results

The final results of each part are shown in the Jupyter Notebook files.

# Contributing

Contributions are welcome! Please open an issue or a pull request if you have any suggestions or improvements.
