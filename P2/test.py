# test_model.py - Run this to test if your model works
from llama_index.llms.llama_cpp import LlamaCPP

# Test your model
model_path = "models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
 # Update with your model path

try:
    llm = LlamaCPP(
        model_path=model_path,
        temperature=0.1,
        max_new_tokens=256,
        context_window=2048,
        verbose=True,
        model_kwargs={
            "n_gpu_layers": 0,
            "n_ctx": 2048,
            "verbose": True,
        }
    )
    
    # Test with simple question
    response = llm.complete("What is 2+2?")
    print("Model Response:", response)
    
except Exception as e:
    print(f"Error: {e}")
    print("Check if your model file exists and the path is correct.")