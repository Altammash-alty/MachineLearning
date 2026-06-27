# Transformer Encoder - Modular & Production-Ready Pipeline

This project contains a refactored and modularized implementation of the PyTorch Transformer Encoder. The original monolithic `encoder.py` script has been split into dedicated modules following production-level machine learning engineering standards.

## Project Structure

```text
Transformer/
├── config.py          # Data classes managing configuration and hyper-parameters
├── model.py           # Core Neural Network architecture definition (EncoderModel)
├── dataset.py         # Data pipeline (NewDataset class) with a fallback tokenizer
├── trainer.py         # Trainer class handling optimization, logging, evaluation, and checkpoints
├── main.py            # Orchestrator script setting up training pipeline and dummy data
├── encoder.py         # Backward-compatible entrypoint wrapper
├── requirements.txt   # Package dependencies list
└── README.md          # Project documentation
```

## Architectural & Bug Fixes

During refactoring, several core issues in the original monolithic code were fixed:
1. **Instantiation Bug in `nn.TransformerEncoder`**: 
   - *Original*: `padding_mask=attention_mask` was supplied to the constructor of `nn.TransformerEncoder` referencing an undefined variable.
   - *Fixed*: In PyTorch, padding masks are dynamic parameters passed to the `forward()` pass. The constructor initialization has been corrected, and the mask is now correctly supplied via `src_key_padding_mask` in `forward()`.
2. **Missing Attention Mask inside training loop**:
   - *Original*: In the training loop, `outputs = model(input_ids, attention_mask)` was called, but `attention_mask` was not defined.
   - *Fixed*: Attention masks are now correctly fetched from each DataLoader batch: `attention_mask = batch["attention_mask"]`.
3. **Out of Memory Protection on Data Loading**:
   - *Original*: The dataset loaded the entire text file and passed it as one single continuous sequence to the tokenizer, which would fail to train multiple examples or run out of memory on larger texts.
   - *Fixed*: The dataset loader reads lines from the file and tokenizes them as individual sequence examples, supporting batch processing.

## Getting Started

### Prerequisites
Install dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Running the pipeline
To start the training run:
```bash
python main.py
```
*(Alternatively, you can run `python encoder.py` which delegates execution to `main.py` for backward compatibility).*

On execution, the script will:
1. Automatically generate a dummy `data.txt` file if none exists.
2. Attempt to load a Hugging Face tokenizer (`bert-base-uncased`). If not available (e.g. due to missing packages or internet access), it falls back to a custom, lightweight character-level tokenizer (`BasicCharTokenizer`) so the code runs out-of-the-box.
3. Split the dataset into training and validation sets.
4. Auto-detect and run on the best hardware device available (`cuda`, `mps`, or `cpu`).
5. Run the training loop, output logs, evaluate on validation data, and save checkpoints to `checkpoints/best_model.pt`.
