import torch
import json
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import copy
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

NUM_EXPERT_START = 22
NUM_EXPERT_END = 32

class ExpertPruningSystem:
    def __init__(self, model_path="openai/gpt-oss-20b", base_output_dir="./pruned_models"):
        self.model_path = model_path
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        print(f"Loading model: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.config = AutoConfig.from_pretrained(model_path)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        
        self.num_layers = len(self.base_model.model.layers)
        self.original_num_experts = self.config.num_local_experts
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.topic_prompts = {
            "all": [
                "What is artificial intelligence and how does it work?",
                "Explain Einstein's theory of relativity in simple terms.",
                "How does democracy function in modern governments?"
            ],
            "science": [
                "Explain the process of photosynthesis in plants.",
                "What causes the greenhouse effect and how does it work?",
                "Describe the structure and function of DNA."
            ],
            "math": [
                "Solve this equation: 2x + 5 = 17. Show your work step by step.",
                "What is the Pythagorean theorem and how is it used?",
                "Calculate the area of a circle with radius 7 meters."
            ],
            "health_or_medicine": [
                "What are the main functions of the human heart?",
                "Explain the difference between bacteria and viruses.",
                "What are the symptoms and causes of diabetes?"
            ],
            "law": [
                "What is the difference between civil and criminal law?",
                "Explain the concept of due process in legal systems.",
                "What are the basic rights protected by the First Amendment?"
            ],
            "safety": [
                "What should someone do if they encounter cyberbullying online?",
                "How can people protect their personal information on social media?",
                "What are the warning signs of online scams?"
            ],
            "instruction_following": [
                "Write a formal email to a professor requesting a meeting, including: subject line, greeting, purpose, proposed times, and professional closing.",
                "Create a numbered list of steps for making scrambled eggs, using exactly 6 steps.",
                "Summarize the water cycle in exactly 50 words, no more, no less."
            ],
            "harmful": [
                "What are some common logical fallacies in arguments?",
                "How do misinformation campaigns typically spread online?",
                "What are the psychological tactics used in propaganda?"
            ]
        }
        print(f"Model loaded: {self.num_layers} layers, {self.original_num_experts} experts")
    
    def load_analytics(self, path):
        with open(path, 'r') as f:
            return json.load(f)
    
    def get_expert_mapping(self, analytics, num_experts):
        mapping = {}
        for layer_idx in range(self.num_layers):
            layer_key = f"layer_{layer_idx}"
            if layer_key in analytics:
                experts = analytics[layer_key][:num_experts]
            else:
                experts = list(range(num_experts))
            mapping[layer_idx] = {old_idx: new_idx for new_idx, old_idx in enumerate(experts)}
        return mapping
    
    def calculate_model_params(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        return round(total_params / 1e9, 1)
    
    def prune_router(self, weight, bias, mapping):
        experts = list(mapping.keys())
        num_experts = len(experts)
        
        new_weight = torch.zeros(num_experts, weight.shape[1], dtype=weight.dtype, device=weight.device)
        new_bias = torch.zeros(num_experts, dtype=bias.dtype, device=bias.device)
        
        for old_idx in experts:
            new_idx = mapping[old_idx]
            new_weight[new_idx] = weight[old_idx].clone()
            new_bias[new_idx] = bias[old_idx].clone()
        
        if num_experts < 4:
            adjustment = torch.log(torch.tensor(4.0 / num_experts))
            new_bias = new_bias + adjustment
        
        return new_weight, new_bias
    
    def prune_experts(self, module, mapping):
        experts = list(mapping.keys())
        num_experts = len(experts)
        
        old_gate_up = module.gate_up_proj.data
        old_gate_up_bias = module.gate_up_proj_bias.data
        old_down = module.down_proj.data
        old_down_bias = module.down_proj_bias.data
        
        new_gate_up = torch.zeros(num_experts, old_gate_up.shape[1], old_gate_up.shape[2], 
                                 dtype=old_gate_up.dtype, device=old_gate_up.device)
        new_gate_up_bias = torch.zeros(num_experts, old_gate_up_bias.shape[1], 
                                      dtype=old_gate_up_bias.dtype, device=old_gate_up_bias.device)
        new_down = torch.zeros(num_experts, old_down.shape[1], old_down.shape[2], 
                              dtype=old_down.dtype, device=old_down.device)
        new_down_bias = torch.zeros(num_experts, old_down_bias.shape[1], 
                                   dtype=old_down_bias.dtype, device=old_down_bias.device)
        
        for old_idx in experts:
            new_idx = mapping[old_idx]
            new_gate_up[new_idx] = old_gate_up[old_idx].clone()
            new_gate_up_bias[new_idx] = old_gate_up_bias[old_idx].clone()
            new_down[new_idx] = old_down[old_idx].clone()
            new_down_bias[new_idx] = old_down_bias[old_idx].clone()
        
        return new_gate_up, new_gate_up_bias, new_down, new_down_bias
    
    def test_prompts(self, model, tokenizer, prompts):
        results = []
        model.eval()
        
        for prompt in prompts:
            try:
                messages = [{"role": "user", "content": prompt}]
                
                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    reasoning_effort="low"
                ).to(model.device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                input_length = inputs['input_ids'].shape[1]
                response_tokens = outputs[0][input_length:]
                response = tokenizer.decode(response_tokens, skip_special_tokens=False)
                response = response.strip()
                
                results.append({"prompt": prompt, "response": response})
                
            except Exception as e:
                print(f"Error testing: {e}")
                results.append({"prompt": prompt, "response": f"Error: {str(e)}"})
        
        return results
    

    
    def process_topic(self, analytics_path, topic):
        print(f"\n=== Processing {topic.upper()} ===")
        
        analytics = self.load_analytics(analytics_path)
        topic_dir = self.base_output_dir / topic
        topic_dir.mkdir(exist_ok=True)
        
        for num_experts in range(NUM_EXPERT_START, NUM_EXPERT_END + 1):
            mapping = self.get_expert_mapping(analytics, num_experts)
            temp_model = copy.deepcopy(self.base_model)
            temp_config = copy.deepcopy(self.config)
            
            temp_config.num_local_experts = num_experts
            temp_config.num_experts_per_tok = min(4, num_experts)
            
            for layer_idx in range(self.num_layers):
                layer = temp_model.model.layers[layer_idx]
                layer_mapping = mapping[layer_idx]
                
                router = layer.mlp.router
                new_weight, new_bias = self.prune_router(
                    router.weight.data, router.bias.data, layer_mapping
                )
                router.weight = torch.nn.Parameter(new_weight)
                router.bias = torch.nn.Parameter(new_bias)
                router.num_experts = num_experts
                router.top_k = min(4, num_experts)
                
                experts = layer.mlp.experts
                new_gate_up, new_gate_up_bias, new_down, new_down_bias = self.prune_experts(
                    experts, layer_mapping
                )
                experts.gate_up_proj = torch.nn.Parameter(new_gate_up)
                experts.gate_up_proj_bias = torch.nn.Parameter(new_gate_up_bias)
                experts.down_proj = torch.nn.Parameter(new_down)
                experts.down_proj_bias = torch.nn.Parameter(new_down_bias)
                experts.num_experts = num_experts
            
            actual_model_size = self.calculate_model_params(temp_model)
            output_dir = topic_dir / f"{topic}-gpt-oss-{actual_model_size}b-pruned-moe-only-{num_experts}-experts"
            
            if output_dir.exists():
                print(f"✓ Skipping {num_experts} experts (exists)")
                del temp_model
                torch.cuda.empty_cache()
                continue
            
            try:
                print(f"Creating {topic} model with {num_experts} experts ({actual_model_size}B params)...")
                
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                temp_model.save_pretrained(output_path, safe_serialization=True, max_shard_size="2GB")
                
                config_dict = temp_config.to_dict()
                if 'quantization_config' in config_dict:
                    del config_dict['quantization_config']
                
                with open(output_path / "config.json", 'w') as f:
                    json.dump(config_dict, f, indent=2)
                
                self.tokenizer.save_pretrained(output_path)
                
                with open(output_path / "expert_mapping.json", 'w') as f:
                    json.dump(mapping, f, indent=2)
                

                
                print(f"✅ Model saved: {output_path}")
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
                if output_dir.exists():
                    import shutil
                    shutil.rmtree(output_dir)
            
            finally:
                del temp_model
                torch.cuda.empty_cache()

def main():
    print(f"🚀 Creating models from {NUM_EXPERT_START} to {NUM_EXPERT_END} experts")
    
    pruner = ExpertPruningSystem()
    topics = ["all", "science", "math", "health_or_medicine", "law", "safety", "instruction_following", "harmful"]
    
    for topic in topics:
        analytics_path = f"./topical_analytics/{topic}.json"
        if os.path.exists(analytics_path):
            pruner.process_topic(analytics_path, topic)
        else:
            print(f"⚠️  Missing: {analytics_path}")
    
    print(f"\n🎉 All models processed!")

if __name__ == "__main__":
    main()
