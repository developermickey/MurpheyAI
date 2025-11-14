import torch
from typing import List, Dict, Optional, AsyncGenerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.core.config import settings
import logging
import asyncio

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = settings.GPU_DEVICE if torch.cuda.is_available() else "cpu"
        self._load_model(settings.MODEL_NAME)
    
    def _load_model(self, model_name: str):
        """Load a model and tokenizer from Hugging Face"""
        try:
            model_path = f"{settings.MODEL_PATH}/{model_name}" if hasattr(settings, 'MODEL_PATH') else model_name
            # Load tokenizer and model from HF Hub or local
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            logger.info(f"Model {model_name} loaded on {self.device}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    async def generate(
        self,
        prompt: str,
        model_name: str = "small",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        model = self.models.get(model_name)
        tokenizer = self.tokenizers.get(model_name)
        if model is None or tokenizer is None:
            self._load_model(model_name)
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        # Generate output
        output_ids = model.generate(
            input_ids,
            max_length=min(max_tokens, input_ids.shape[-1] + max_tokens),
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Remove input from output
        output_text = output_text[len(prompt):].lstrip()
        if stream:
            for word in output_text.split():
                yield word + " "
                await asyncio.sleep(0.03)
        else:
            yield output_text
    
    def count_tokens(self, text: str, model_name: str = "small") -> int:
        tokenizer = self.tokenizers.get(model_name)
        if tokenizer is None:
            self._load_model(model_name)
            tokenizer = self.tokenizers[model_name]
        return len(tokenizer.encode(text))
    
    def get_available_models(self) -> List[str]:
        return list(self.models.keys()) or ["small"]
    
    def validate_model_name(self, model_name: str) -> bool:
        return model_name in self.get_available_models()


# Global model service instance
model_service = ModelService()

