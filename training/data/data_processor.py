"""
Data preprocessing and cleaning pipeline.
"""
import json
import re
import hashlib
from typing import List, Dict, Set
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and clean training data."""
    
    def __init__(self, input_dir: str = "./data/raw", output_dir: str = "./data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seen_hashes: Set[str] = set()
    
    def deduplicate(self, text: str) -> bool:
        """Check if text is a duplicate."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(text_hash)
        return False
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\-()\[\]{}"\']', '', text)
        
        return text.strip()
    
    def filter_quality(self, text: str, min_length: int = 50, max_length: int = 10000) -> bool:
        """Filter text based on quality criteria."""
        if len(text) < min_length or len(text) > max_length:
            return False
        
        # Check for too many repeated characters
        if re.search(r'(.)\1{10,}', text):
            return False
        
        # Check for too many special characters
        special_char_ratio = len(re.findall(r'[^\w\s]', text)) / len(text) if text else 0
        if special_char_ratio > 0.3:
            return False
        
        return True
    
    def process_file(self, input_file: Path) -> int:
        """Process a single input file."""
        logger.info(f"Processing {input_file.name}...")
        output_file = self.output_dir / f"processed_{input_file.name}"
        
        processed_count = 0
        skipped_count = 0
        
        with open(input_file, "r", encoding="utf-8") as infile, \
             open(output_file, "w", encoding="utf-8") as outfile:
            
            for line in tqdm(infile, desc=f"Processing {input_file.name}"):
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    
                    # Clean text
                    text = self.clean_text(text)
                    
                    # Filter quality
                    if not self.filter_quality(text):
                        skipped_count += 1
                        continue
                    
                    # Deduplicate
                    if self.deduplicate(text):
                        skipped_count += 1
                        continue
                    
                    # Write processed text
                    outfile.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    processed_count += 1
                    
                except json.JSONDecodeError:
                    skipped_count += 1
                    continue
                except Exception as e:
                    logger.error(f"Error processing line: {e}")
                    skipped_count += 1
                    continue
        
        logger.info(
            f"Processed {input_file.name}: {processed_count} kept, {skipped_count} skipped"
        )
        return processed_count
    
    def process_all(self):
        """Process all files in input directory."""
        input_files = list(self.input_dir.glob("*.jsonl"))
        
        if not input_files:
            logger.warning(f"No .jsonl files found in {self.input_dir}")
            return
        
        total_processed = 0
        for input_file in input_files:
            count = self.process_file(input_file)
            total_processed += count
        
        logger.info(f"Total processed: {total_processed} texts")


if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_all()

