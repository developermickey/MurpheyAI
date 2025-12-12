from typing import List, AsyncGenerator
import importlib
from app.core.config import settings
import logging
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        # Default device; will detect torch availability when loading models
        self.device = "cpu"
        # Defer heavy model loading until first use to allow the app to start
        # without requiring large ML dependencies or model files.
        self._initial_load_attempted = False
        self._ml_imported = False

    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate a simple fallback response when model is unavailable."""
        # Simple rule-based responses for common prompts
        prompt_lower = prompt.lower().strip()

        fallback_responses = {
            "hello": "Hello! I'm MurpheyAI. How can I help you today?",
            "hi": "Hi there! What would you like to know?",
            "how are you": "I'm doing well, thank you for asking! How can I assist you?",
            "what is your name": "I'm MurpheyAI, a conversational AI assistant.",
            "help": "I'm here to help! You can ask me questions, and I'll do my best to provide useful responses.",
        }

        # Try exact matches first
        if prompt_lower in fallback_responses:
            return fallback_responses[prompt_lower]

        # Try partial matches
        for key, response in fallback_responses.items():
            if key in prompt_lower:
                return response

        # Default fallback: echo-like response
        if len(prompt) < 50:
            return f"That's an interesting question about '{prompt}'. In a production environment with a loaded model, I would provide a detailed response."
        else:
            return "Thank you for that detailed input. With a properly loaded model, I would analyze this further and provide a comprehensive response. Please ensure MODEL_NAME is set to a valid Hugging Face model ID or a local model path."

    def _load_model(self, model_name: str):
        """Load a model and tokenizer from Hugging Face or custom checkpoints."""
        try:
            # Prefer HF if available
            try:
                transformers = importlib.import_module("transformers")
                torch = importlib.import_module("torch")
                AutoTokenizer = getattr(transformers, "AutoTokenizer")
                AutoModelForCausalLM = getattr(
                    transformers, "AutoModelForCausalLM")
                self._ml_imported = True

                # Determine model path: if MODEL_PATH is set and not empty, use it as directory
                # Otherwise, use model_name directly as HF model ID
                model_path_setting = getattr(settings, "MODEL_PATH", None)
                if model_path_setting and model_path_setting.strip():
                    model_path = f"{model_path_setting}/{model_name}"
                else:
                    model_path = model_name  # Use as Hugging Face model ID directly

                logger.info(f"Loading Hugging Face model from: {model_path}")
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                # Set pad_token if it doesn't exist (GPT-2 doesn't have one)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer
                self.device = settings.GPU_DEVICE if torch.cuda.is_available() else "cpu"
                logger.info("HF model %s loaded on %s",
                            model_name, self.device)
                return
            except Exception as e:
                logger.warning(
                    "HF load failed for %s: %s. Trying custom checkpoint.", model_name, e)

            # Try custom checkpoint
            try:
                from app.services.custom_model import load_custom_model
                import torch as torch_module
                model_dir = Path(
                    getattr(settings, "MODEL_PATH", "./models")) / model_name
                model, tokenizer = load_custom_model(
                    model_dir, device=torch_module.device("cpu"))
                self.models[model_name] = {
                    "type": "custom", "model": model, "tokenizer": tokenizer}
                self.tokenizers[model_name] = tokenizer
                self.device = "cpu"  # custom loader currently CPU
                logger.info("Custom model %s loaded from %s",
                            model_name, model_dir)
            except ImportError as e:
                logger.warning(
                    "Custom model loader not available (torch/tokenizers missing): %s", e)
                raise
        except Exception as e:
            logger.error("Error loading model %s: %s", model_name, e)
            self.models[model_name] = None
            self.tokenizers[model_name] = None
        finally:
            self._initial_load_attempted = True

    async def generate(
        self,
        prompt: str,
        model_name: str = "small",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        # Support an alias "small" for the configured default model name
        effective_model_name = model_name if model_name != "small" else getattr(
            settings, "MODEL_NAME", "gpt2")

        model = self.models.get(effective_model_name)
        tokenizer = self.tokenizers.get(effective_model_name)
        if model is None or tokenizer is None:
            try:
                self._load_model(effective_model_name)
                model = self.models.get(effective_model_name)
                tokenizer = self.tokenizers.get(effective_model_name)
            except Exception:
                model = None
                tokenizer = None

        if model is None or tokenizer is None:
            # Fallback: use a simple mock response generator for demo/testing
            # In production, ensure the model loads properly
            logger.warning(
                f"Model {effective_model_name} not available; using fallback response."
            )

            # Generate a simple, contextual response
            fallback_response = self._generate_fallback_response(prompt)

            if stream:
                # Stream word-by-word for fallback
                for word in fallback_response.split():
                    yield word + " "
                    # Small async delay for realistic streaming effect
                    await asyncio.sleep(0.05)
            else:
                yield fallback_response
            return

        # Custom model path
        if isinstance(model, dict) and model.get("type") == "custom":
            from app.services.custom_model import generate_custom
            text = generate_custom(
                model=model["model"],
                tokenizer=model["tokenizer"],
                prompt=prompt,
                max_tokens=min(max_tokens, 256),
                temperature=temperature,
            )
            if stream:
                yield text
            else:
                yield text
            return

        # Tokenize with attention mask
        encoded = tokenizer(prompt, return_tensors="pt",
                            padding=True, truncation=True, max_length=512)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Generate output
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            # Limit new tokens for faster generation
            max_new_tokens=min(max_tokens, 512),
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        # Decode only the newly generated tokens
        input_length = input_ids.shape[-1]
        generated_ids = output_ids[0][input_length:]
        output_text = tokenizer.decode(
            generated_ids, skip_special_tokens=True).strip()
        if stream:
            for word in output_text.split():
                yield word + " "
                await asyncio.sleep(0.03)
        else:
            yield output_text

    def count_tokens(self, text: str, model_name: str = "small") -> int:
        """
        Count tokens for a given text. If a tokenizer isn't available
        (e.g., models not installed in the environment), fall back to
        a simple whitespace count to avoid runtime failures.
        """
        tokenizer = self.tokenizers.get(model_name)
        if tokenizer is None:
            # Try to load; if still missing, use naive fallback
            self._load_model(model_name)
            tokenizer = self.tokenizers.get(model_name)

        if tokenizer is None:
            # Simple fallback: approximate tokens by whitespace splitting
            return max(1, len(text.split()))

        return len(tokenizer.encode(text))

    def get_available_models(self) -> List[str]:
        return list(self.models.keys()) or ["small"]

    def validate_model_name(self, model_name: str) -> bool:
        return model_name in self.get_available_models()


# Global model service instance
model_service = ModelService()
