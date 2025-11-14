"""
Data collection script for gathering training data from various sources.
"""
import os
import json
import requests
from typing import List, Dict
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    """Collect training data from various sources."""
    
    def __init__(self, output_dir: str = "./data/raw"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def collect_common_crawl(self, limit: int = 100000):
        """Collect data from Common Crawl."""
        logger.info("Collecting Common Crawl data...")
        # In production, use cc_net or cc-pipeline
        # This is a placeholder
        pass
    
    def collect_wikipedia(self, language: str = "en"):
        """Collect Wikipedia data."""
        logger.info(f"Collecting Wikipedia data ({language})...")
        try:
            dataset = load_dataset("wikipedia", f"20220301.{language}", split="train")
            output_file = os.path.join(self.output_dir, f"wikipedia_{language}.jsonl")
            
            with open(output_file, "w", encoding="utf-8") as f:
                for item in dataset:
                    f.write(json.dumps({"text": item["text"]}, ensure_ascii=False) + "\n")
            
            logger.info(f"Saved {len(dataset)} Wikipedia articles")
        except Exception as e:
            logger.error(f"Error collecting Wikipedia: {e}")
    
    def collect_github_code(self, limit: int = 10000):
        """Collect code from GitHub."""
        logger.info("Collecting GitHub code...")
        # In production, use GitHub API or BigCode datasets
        try:
            dataset = load_dataset("bigcode/the-stack", split="train", streaming=True)
            output_file = os.path.join(self.output_dir, "github_code.jsonl")
            
            count = 0
            with open(output_file, "w", encoding="utf-8") as f:
                for item in dataset:
                    if count >= limit:
                        break
                    f.write(json.dumps({
                        "text": item.get("content", ""),
                        "language": item.get("language", ""),
                    }, ensure_ascii=False) + "\n")
                    count += 1
            
            logger.info(f"Saved {count} code samples")
        except Exception as e:
            logger.error(f"Error collecting GitHub code: {e}")
    
    def collect_reddit(self, limit: int = 50000):
        """Collect Reddit conversations."""
        logger.info("Collecting Reddit data...")
        try:
            dataset = load_dataset("reddit", split="train", streaming=True)
            output_file = os.path.join(self.output_dir, "reddit.jsonl")
            
            count = 0
            with open(output_file, "w", encoding="utf-8") as f:
                for item in dataset:
                    if count >= limit:
                        break
                    f.write(json.dumps({
                        "text": item.get("content", ""),
                        "title": item.get("title", ""),
                    }, ensure_ascii=False) + "\n")
                    count += 1
            
            logger.info(f"Saved {count} Reddit posts")
        except Exception as e:
            logger.error(f"Error collecting Reddit: {e}")
    
    def collect_instructions(self):
        """Collect instruction-following datasets."""
        logger.info("Collecting instruction datasets...")
        datasets = [
            "tatsu-lab/alpaca",
            "WizardLM/WizardLM_evol_instruct_V2",
            "Open-Orca/OpenOrca",
        ]
        
        for dataset_name in datasets:
            try:
                dataset = load_dataset(dataset_name, split="train")
                output_file = os.path.join(
                    self.output_dir, 
                    f"instructions_{dataset_name.split('/')[-1]}.jsonl"
                )
                
                with open(output_file, "w", encoding="utf-8") as f:
                    for item in dataset:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
                logger.info(f"Saved {len(dataset)} items from {dataset_name}")
            except Exception as e:
                logger.error(f"Error collecting {dataset_name}: {e}")


if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_wikipedia()
    collector.collect_github_code(limit=1000)
    collector.collect_instructions()

