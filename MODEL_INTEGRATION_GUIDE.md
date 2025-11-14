# 🔌 Model Integration Guide

## Quick Start: Integrating a Real Model

This guide shows you how to replace the mock model service with a real working model.

---

## Step 1: Choose Your Model

### Option 1: Small & Fast (Recommended for Testing)
- **Model:** `gpt2` (124M parameters)
- **Pros:** Fast, low memory, good for testing
- **Cons:** Lower quality responses
- **Memory:** ~500MB GPU

### Option 2: Medium Quality
- **Model:** `mistralai/Mistral-7B-Instruct-v0.2` (7B parameters)
- **Pros:** Good balance of quality and speed
- **Cons:** Requires ~14GB GPU memory
- **Memory:** ~14GB GPU (with quantization: ~4GB)

### Option 3: High Quality
- **Model:** `meta-llama/Llama-2-7b-chat-hf` (7B parameters)
- **Pros:** Excellent quality
- **Cons:** Requires access approval, larger memory
- **Memory:** ~14GB GPU

---

## Step 2: Install Dependencies

Add to `backend/requirements.txt`:

```txt
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.24.0
bitsandbytes>=0.41.0  # For quantization
```

Then install:
```bash
cd backend
pip install -r requirements.txt
```

---

## Step 3: Update Model Service

Here's the updated `model_service.py` implementation:

```python
import torch
from typing import List, Optional, AsyncGenerator
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
        logger.info(f"Using device: {self.device}")
        
        # Load default model
        self._load_model(settings.MODEL_NAME)
    
    def _load_model(self, model_name: str):
        """Load a model and tokenizer."""
        try:
            # Map model names to Hugging Face model IDs
            model_map = {
                "small": "gpt2",
                "medium": "mistralai/Mistral-7B-Instruct-v0.2",
                "large": "meta-llama/Llama-2-13b-chat-hf",
                "murpheyai-7b": "mistralai/Mistral-7B-Instruct-v0.2",  # Default
            }
            
            hf_model_id = model_map.get(model_name, model_name)
            
            logger.info(f"Loading model: {hf_model_id}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with optimizations
            model = AutoModelForCausalLM.from_pretrained(
                hf_model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
            )
            
            if not torch.cuda.is_available():
                model = model.to(self.device)
            
            model.eval()  # Set to evaluation mode
            
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            logger.info(f"Model {model_name} loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            # Fallback to GPT-2 if model fails to load
            if model_name != "small":
                logger.warning("Falling back to GPT-2")
                self._load_model("small")
            else:
                raise
    
    async def generate(
        self,
        prompt: str,
        model_name: str = "small",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """Generate text from a prompt."""
        if model_name not in self.models:
            self._load_model(model_name)
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        # Generate
        if stream:
            # Streaming generation
            generated_text = ""
            with torch.no_grad():
                for _ in range(max_tokens):
                    # Forward pass
                    outputs = model(**inputs)
                    logits = outputs.logits[:, -1, :] / temperature
                    
                    # Sample next token
                    probs = torch.softmax(logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                    
                    # Decode token
                    next_token = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
                    generated_text += next_token
                    
                    # Yield token
                    yield next_token
                    
                    # Check for stopping conditions
                    if next_token_id.item() == tokenizer.eos_token_id:
                        break
                    
                    # Update inputs for next iteration
                    inputs.input_ids = torch.cat([inputs.input_ids, next_token_id], dim=1)
                    
                    # Prevent infinite loops
                    if inputs.input_ids.shape[1] >= input_length + max_tokens:
                        break
                    
                    # Small delay for streaming effect
                    await asyncio.sleep(0.01)
        else:
            # Non-streaming generation
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            generated_text = tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            )
            yield generated_text
    
    def count_tokens(self, text: str, model_name: str = "small") -> int:
        """Count tokens in text."""
        if model_name not in self.tokenizers:
            if model_name not in self.models:
                self._load_model(model_name)
        
        tokenizer = self.tokenizers[model_name]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys()) if self.models else ["small", "medium", "large"]
    
    def validate_model_name(self, model_name: str) -> bool:
        """Validate if model name is available."""
        return model_name in self.get_available_models() or model_name in [
            "small", "medium", "large", "murpheyai-7b"
        ]


# Global model service instance
model_service = ModelService()
```

---

## Step 4: Optimize for Lower Memory (Optional)

If you have limited GPU memory, use quantization:

```python
from transformers import BitsAndBytesConfig

# Add to _load_model method:
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    hf_model_id,
    quantization_config=quantization_config,
    device_map="auto",
)
```

This reduces 7B model memory from ~14GB to ~4GB.

---

## Step 5: Update Configuration

Update `backend/app/core/config.py`:

```python
# Model Configuration
MODEL_PATH: str = "./models"
MODEL_NAME: str = "small"  # Start with small for testing
# Or use: "mistralai/Mistral-7B-Instruct-v0.2" for better quality
MAX_TOKENS: int = 512  # Reduce for faster responses
TEMPERATURE: float = 0.7
TOP_P: float = 0.9
```

---

## Step 6: Test the Integration

Create a test script `backend/test_model.py`:

```python
import asyncio
from app.services.model_service import model_service

async def test():
    prompt = "Hello, how are you?"
    
    print("Testing model generation...")
    async for chunk in model_service.generate(prompt, model_name="small", stream=True):
        print(chunk, end="", flush=True)
    print("\n")
    
    print(f"Token count: {model_service.count_tokens(prompt)}")

if __name__ == "__main__":
    asyncio.run(test())
```

Run it:
```bash
cd backend
python test_model.py
```

---

## Step 7: Better Streaming Implementation

For better streaming performance, use the model's built-in streaming:

```python
from transformers import TextIteratorStreamer
from threading import Thread

async def generate_streaming(self, prompt, model_name, temperature, max_tokens):
    """Better streaming with TextIteratorStreamer."""
    model = self.models[model_name]
    tokenizer = self.tokenizers[model_name]
    
    inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "streamer": streamer,
        "do_sample": True,
    }
    
    # Run generation in separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream tokens
    for text in streamer:
        yield text
```

---

## Troubleshooting

### Issue: Out of Memory
**Solution:** 
- Use smaller model (GPT-2)
- Enable quantization
- Reduce `max_tokens`
- Use CPU instead of GPU (slower but works)

### Issue: Model Download Fails
**Solution:**
- Check internet connection
- Use `huggingface-cli login` for gated models
- Download model manually and use local path

### Issue: Slow Generation
**Solution:**
- Use GPU instead of CPU
- Reduce `max_tokens`
- Use smaller model
- Enable KV caching (already in transformers)

### Issue: Poor Quality Responses
**Solution:**
- Use larger model (7B+)
- Adjust temperature (lower = more focused)
- Improve prompts
- Fine-tune on your data

---

## Next Steps After Integration

1. ✅ Test with real chat interface
2. ✅ Monitor token usage and costs
3. ✅ Optimize generation parameters
4. ✅ Add response caching
5. ✅ Set up model serving (vLLM) for production

---

## Production Considerations

For production, consider:

1. **vLLM Server:** Separate model serving for better performance
2. **Model Caching:** Keep models in memory
3. **Load Balancing:** Multiple model instances
4. **Monitoring:** Track latency, memory usage, errors
5. **Fallbacks:** Graceful degradation if model fails

See `NEXT_STEPS.md` for more details.

