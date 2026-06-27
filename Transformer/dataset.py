import torch
from torch.utils.data import Dataset
from typing import Union, List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class BasicCharTokenizer:
    """
    A basic character-level tokenizer used for fallback testing.
    Mimics Hugging Face tokenizer API for compatibility.
    """
    def __init__(self, vocab_size: int = 30000, max_length: int = 128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
    def __call__(
        self, 
        text: Union[str, List[str]], 
        truncation: bool = True, 
        padding: str = "max_length", 
        max_length: Optional[int] = None, 
        return_tensors: str = "pt"
    ) -> Dict[str, Any]:
        
        target_max_len = max_length if max_length is not None else self.max_length
        
        if isinstance(text, str):
            texts = [text]
        elif isinstance(text, list):
            texts = text
        else:
            raise ValueError("Input text must be a string or a list of strings.")
            
        all_input_ids = []
        all_attention_masks = []
        
        for t in texts:
            # Shift characters into valid range to avoid collision with special tokens
            ids = [ord(char) % (self.vocab_size - 10) + 10 for char in t]
            
            if truncation and len(ids) > target_max_len - 2:
                ids = ids[:target_max_len - 2]
                
            # Wrap with BOS (Beginning of Sequence) and EOS (End of Sequence) tokens
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
            
            attention_mask = [1] * len(ids)
            
            if padding == "max_length":
                pad_len = target_max_len - len(ids)
                if pad_len > 0:
                    ids += [self.pad_token_id] * pad_len
                    attention_mask += [0] * pad_len
                elif pad_len < 0:
                    ids = ids[:target_max_len]
                    attention_mask = attention_mask[:target_max_len]
                    
            all_input_ids.append(ids)
            all_attention_masks.append(attention_mask)
            
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(all_attention_masks, dtype=torch.long)
            }
        
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks
        }


class NewDataset(Dataset):
    """
    Dataset class to load text from a file, split it into lines/paragraphs, 
    and tokenize it with a given tokenizer.
    """
    def __init__(self, text_file: str, tokenizer: Any, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        logger.info(f"Loading dataset from: {text_file}")
        
        try:
            with open(text_file, "r", encoding="utf-8") as file:
                lines = [line.strip() for line in file if line.strip()]
        except FileNotFoundError:
            logger.error(f"Text file {text_file} not found.")
            raise

        if not lines:
            raise ValueError(f"No valid text lines found in {text_file}")
            
        logger.info(f"Loaded {len(lines)} lines of text. Tokenizing...")

        # Bulk tokenization of all lines
        self.data = self.tokenizer(
            lines,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        logger.info("Tokenization complete.")

    def __len__(self) -> int:
        return self.data["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.data["input_ids"][idx],
            "attention_mask": self.data["attention_mask"][idx]
        }
