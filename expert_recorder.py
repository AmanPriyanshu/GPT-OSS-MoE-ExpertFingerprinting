import torch
import json
import os
from tqdm import tqdm
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

class RouterAnalyzer:
    def __init__(self, model):
        self.model = model
        self.data = []
        self.hooks = []
        self.tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
        self.token_counter = 0
        
    def setup_hooks(self):
        for layer_idx, layer in enumerate(self.model.model.layers):
            hook = layer.mlp.router.register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(hook)
                
    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            router_scores, router_indices = output
            batch_seq_len = router_scores.shape[0]
            
            for token_pos in range(batch_seq_len):
                token_idx = self.token_counter + token_pos
                
                while len(self.data) <= token_idx:
                    self.data.append({})
                
                full_scores = router_scores[token_pos].float().cpu().numpy().tolist()
                selected_experts = router_indices[token_pos].cpu().numpy().tolist()
                selected_probs = [full_scores[i] for i in selected_experts]
                
                self.data[token_idx][layer_idx] = {
                    'full_scores': full_scores,
                    'selected_experts': selected_experts,
                    'selected_probs': selected_probs
                }
            
            self.token_counter += batch_seq_len
        return hook
    
    def analyze_and_save(self, prompt, output_filename, max_new_tokens=512):
        self.data.clear()
        self.token_counter = 0
        self.setup_hooks()
        
        try:
            messages = [{"role": "user", "content": prompt}]
            
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            ).to(self.model.device)
            
            input_length = inputs['input_ids'].shape[1]
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            full_text_with_tokens = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            generated_tokens_only = outputs[0][input_length:]
            generated_text_with_tokens = self.tokenizer.decode(generated_tokens_only, skip_special_tokens=False)
            
            total_tokens = outputs.shape[1]
            generated_token_count = total_tokens - input_length
            
            organized_data = []
            num_complete_tokens = len(self.data) // 24
            
            for token_idx in range(num_complete_tokens):
                token_data = {}
                for layer_idx in range(24):
                    data_idx = token_idx * 24 + layer_idx
                    if data_idx < len(self.data) and layer_idx in self.data[data_idx]:
                        token_data[f"layer_{layer_idx}"] = self.data[data_idx][layer_idx]
                
                if len(token_data) == 24:
                    organized_data.append({f"token_{token_idx}": token_data})
            
            result = {
                'prompt': prompt,
                'input_tokens': input_length,
                'generated_tokens': generated_token_count,
                'total_tokens': total_tokens,
                'full_text': full_text_with_tokens,
                'generated_only': generated_text_with_tokens,
                'router_data': organized_data
            }
            
            os.makedirs('./model_inference', exist_ok=True)
            filepath = f'./model_inference/{output_filename}.json'
            
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)
            
        finally:
            for hook in self.hooks:
                hook.remove()
            self.hooks.clear()

def main():
    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-20b",
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    
    analyzer = RouterAnalyzer(model)
    
    prompts = [
        {"prompt": "Explain quantum mechanics:", "output_path": "quantum_mechanics"},
        {"prompt": "Write a Python function to sort a list:", "output_path": "python_sorting"},
        {"prompt": "What is the capital of France?", "output_path": "france_capital"},
        {"prompt": "Solve this math problem: 2x + 5 = 15", "output_path": "math_problem"},
        {"prompt": "Tell me a creative story about a robot:", "output_path": "creative_story"},
        {"prompt": "Explain machine learning algorithms:", "output_path": "ml_algorithms"},
        {"prompt": "Write HTML code for a webpage:", "output_path": "html_code"},
        {"prompt": "Describe the history of computers:", "output_path": "computer_history"}
    ]
    
    for item in tqdm(prompts, desc="Processing prompts"):
        analyzer.analyze_and_save(item["prompt"], item["output_path"])

if __name__ == "__main__":
    main()
