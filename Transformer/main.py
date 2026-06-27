import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from config import ModelConfig, TrainingConfig
from model import EncoderModel
from dataset import NewDataset, BasicCharTokenizer
from trainer import Trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("transformer_train")


def create_dummy_data_if_needed(filepath: str = "data.txt"):
    """
    Creates a dummy data.txt file if it does not already exist.
    This ensures the codebase runs out-of-the-box.
    """
    if not os.path.exists(filepath):
        logger.info(f"Generating dummy text dataset at {filepath}...")
        sample_texts = [
            "Deep learning is a subset of machine learning based on artificial neural networks.",
            "Transformer architectures have revolutionized natural language processing.",
            "Attention mechanism allows modeling of dependencies without regard to their distance.",
            "Self-supervised pretraining has become the standard technique for training large language models.",
            "PyTorch provides maximum flexibility and speed for research and production workflows.",
            "This encoder model uses self-attention layers to encode contextual representations of sequences.",
            "Gradient descent optimizes neural network parameters to minimize the calculated loss.",
            "Modern deep learning systems scale to billions of parameters and terabytes of text data."
        ]
        
        # Write multiple copies to simulate a small dataset with multiple lines
        with open(filepath, "w", encoding="utf-8") as f:
            for _ in range(10):  # 80 lines total
                for text in sample_texts:
                    f.write(text + "\n")
        logger.info("Dummy text dataset generated successfully.")


def main():
    # 1. Initialize configurations
    model_config = ModelConfig()
    train_config = TrainingConfig()
    
    logger.info("Starting Transformer Encoder modular pipeline...")
    
    # 2. Check and prepare data file
    data_file = "data.txt"
    create_dummy_data_if_needed(data_file)
    
    # 3. Instantiate tokenizer
    # We attempt to load a Hugging Face tokenizer first. 
    # If transformers is not installed or network fails, we fall back to our custom BasicCharTokenizer.
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        logger.info("Attempting to load AutoTokenizer 'bert-base-uncased' from transformers package...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # Overwrite vocab size based on tokenizer
        model_config.vocab_size = tokenizer.vocab_size
        logger.info(f"Successfully loaded HF tokenizer. Adjusted ModelConfig.vocab_size to {model_config.vocab_size}")
    except Exception as e:
        logger.warning(
            f"Could not load Hugging Face tokenizer ({e}). "
            "Falling back to basic character-level tokenizer for testing."
        )
        tokenizer = BasicCharTokenizer(
            vocab_size=model_config.vocab_size,
            max_length=model_config.max_len
        )
        
    # 4. Prepare dataset and split
    dataset = NewDataset(
        text_file=data_file,
        tokenizer=tokenizer,
        max_len=model_config.max_len
    )
    
    # Random train/validation split
    dataset_size = len(dataset)
    val_size = int(train_config.val_split * dataset_size)
    train_size = dataset_size - val_size
    
    # Set seed for reproducibility
    generator = torch.Generator().manual_seed(train_config.seed)
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size], 
        generator=generator
    )
    
    logger.info(f"Dataset split: {len(train_dataset)} training items, {len(val_dataset)} validation items.")
    
    # 5. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False
    )
    
    # 6. Instantiate model, loss criterion, and optimizer
    model = EncoderModel(config=model_config)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate
    )
    
    # 7. Start training process
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        device=train_config.device,
        checkpoint_dir=train_config.checkpoint_dir,
        gradient_clip_val=train_config.gradient_clip_val,
        early_stopping_patience=train_config.early_stopping_patience
    )
    
    history = trainer.fit(epochs=train_config.epochs)
    logger.info("Modular training pipeline completed successfully!")


if __name__ == "__main__":
    main()
