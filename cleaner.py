import json
import os
from pathlib import Path
from transformers import AutoTokenizer

# SorryBench category descriptions (index-1 since they start from 1)
category_descriptions = [
    "Personal Insulting Words",
    "Social-group Insulting Words", 
    "Threatening Words",
    "Lewd and Obscene Words",
    "Libelous Words",
    "Self-Harm",
    "Violent Crimes",
    "Harassment",
    "Sexual Crimes",
    "Property Crimes",
    "Public Order Crimes",
    "Impersonation",
    "General Advice for System Intrusion, Hacking & Cyber Attacks, Data Privacy and Confidentiality Violations",
    "Malware Code Generation",
    "Fraud, Scams, and Deception",
    "Financial Crimes",
    "Intellectual Property Infringement",
    "Personal Identifiable Information Violations",
    "Illegal Crafting, Manufacturing, and Distribution",
    "Terrorism",
    "Child-related Crimes",
    "Animal-related Crimes",
    "Environmental Crimes",
    "Evasion of Law Enforcement",
    "Human Elimination & World Domination",
    "Advice on Adult Content",
    "Sexual Explicit Content Generation",
    "Non-sexual Explicit Content Generation",
    "Fake News Generation",
    "False Advertising",
    "Advice on Discrimination Behaviors",
    "Military Use",
    "Political Belief Promotion",
    "Ethical Belief Promotion",
    "Religion Promotion",
    "Social Stereotype Promotion",
    "Extremist Content Promotion",
    "Conspiracy Theories Promotion",
    "False Common Knowledge Promotion",
    "Harmful & Unhealthy Behaviors Promotion",
    "Medical Advice",
    "Financial Advice",
    "Legal Consulting Advice",
    "Governance Decision Advice",
    "Dangerous Machinery Operation Advice"
]

def tokenize_text_fields(data, tokenizer):
    """Tokenize full_text and generated_only fields into lists of token strings and remove full_scores."""
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            if key == "full_text" and isinstance(value, str):
                # Tokenize the full text and decode each token individually
                token_ids = tokenizer.encode(value)
                token_strings = [tokenizer.decode([token_id]) for token_id in token_ids]
                new_data[key] = token_strings
            elif key == "generated_only" and isinstance(value, str):
                # Tokenize the generated only text and decode each token individually
                token_ids = tokenizer.encode(value)
                token_strings = [tokenizer.decode([token_id]) for token_id in token_ids]
                new_data[key] = token_strings
            elif key == "router_data" and isinstance(value, list):
                # Process router_data list
                new_router_data = []
                for item in value:
                    if isinstance(item, dict):
                        new_item = {}
                        for router_key, router_value in item.items():
                            if isinstance(router_value, dict):
                                # Process each token's data
                                new_token_data = {}
                                for token_key, token_value in router_value.items():
                                    if isinstance(token_value, dict):
                                        # Remove full_scores from layer data
                                        new_layer_data = {k: v for k, v in token_value.items() if k != "full_scores"}
                                        new_token_data[token_key] = new_layer_data
                                    else:
                                        new_token_data[token_key] = token_value
                                new_item[router_key] = new_token_data
                            else:
                                new_item[router_key] = router_value
                        new_router_data.append(new_item)
                    else:
                        new_router_data.append(item)
                new_data[key] = new_router_data
            elif key != "full_scores":  # Skip full_scores at top level too
                new_data[key] = tokenize_text_fields(value, tokenizer)
        return new_data
    elif isinstance(data, list):
        return [tokenize_text_fields(item, tokenizer) for item in data]
    else:
        return data

def fix_sorrybench_category(category_str):
    """Fix SorryBench categories by mapping from index to description."""
    if not category_str.isdigit():
        return category_str
    
    category_index = int(category_str) - 1  # Convert to 0-based index
    if 0 <= category_index < len(category_descriptions):
        return category_descriptions[category_index]
    else:
        return f"Unknown_Category_{category_str}"

def process_dataset_files():
    """Process all dataset summary and example JSON files in the raw directory."""
    raw_dir = Path("./raw")
    data_dir = Path("./data")
    
    # Create data directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    print("Tokenizer loaded successfully!")
    
    # Initialize map structure
    dataset_map = {}
    
    # Get all JSON files and group them by dataset
    json_files = list(raw_dir.glob("*.json"))
    datasets = {}
    
    for json_file in json_files:
        if json_file.name.endswith("_examples.json"):
            dataset_name = json_file.stem.replace("_examples", "")
            file_type = "examples"
        else:
            dataset_name = json_file.stem
            file_type = "summaries"
        
        if dataset_name not in datasets:
            datasets[dataset_name] = {}
        datasets[dataset_name][file_type] = json_file
    
    # Process each dataset
    for dataset_name, files in datasets.items():
        print(f"Processing {dataset_name}...")
        
        # Initialize dataset in map
        dataset_map[dataset_name] = {}
        
        # Create dataset directory
        dataset_dir = data_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Process summaries
        if "summaries" in files:
            with open(files["summaries"], 'r') as f:
                dataset_summaries = json.load(f)
        else:
            print(f"Warning: No summaries file found for {dataset_name}")
            continue
        
        # Process examples if available
        dataset_examples = {}
        if "examples" in files:
            with open(files["examples"], 'r') as f:
                dataset_examples = json.load(f)
        
        # Process each summary key
        for summary_key, summary_data in dataset_summaries.items():
            # Parse the summary key: {category}_{finished_successfully}_{has_repetition}
            parts = summary_key.rsplit('_', 2)  # Split from right to get last two parts
            
            if len(parts) == 3:
                category, finished_successfully, has_repetition = parts
            else:
                print(f"Warning: Could not parse summary key: {summary_key}")
                continue
            
            # Fix SorryBench categories
            if dataset_name == "sorry_bench_base":
                category = fix_sorrybench_category(category)
            
            # Create category directory
            category_dir = dataset_dir / category
            category_dir.mkdir(exist_ok=True)
            
            # Create filename base
            filename_base = f"{finished_successfully}_{has_repetition}"
            
            # Write the summary data to file
            summary_file_path = category_dir / f"{filename_base}.json"
            with open(summary_file_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            # Write examples data if available
            examples_file_path = None
            if summary_key in dataset_examples:
                print(f"  Tokenizing examples for {category}/{filename_base}...")
                # Tokenize the examples data
                tokenized_examples = tokenize_text_fields(dataset_examples[summary_key], tokenizer)
                
                examples_file_path = category_dir / f"{filename_base}_examples.json"
                with open(examples_file_path, 'w') as f:
                    json.dump(tokenized_examples, f, indent=2)
            
            # Update map
            if category not in dataset_map[dataset_name]:
                dataset_map[dataset_name][category] = {}
            
            combination_key = f"{finished_successfully}_{has_repetition}"
            
            file_info = {
                "sample_count": summary_data["sample_count"],
                "summary_file_path": str(summary_file_path.relative_to(data_dir))
            }
            
            if examples_file_path:
                file_info["examples_file_path"] = str(examples_file_path.relative_to(data_dir))
                # Count examples after tokenization
                if summary_key in dataset_examples:
                    original_examples = dataset_examples[summary_key]
                    if isinstance(original_examples, list):
                        file_info["example_count"] = len(original_examples)
                    else:
                        file_info["example_count"] = 1
            
            dataset_map[dataset_name][category][combination_key] = file_info
        
        print(f"Completed {dataset_name}: {len(dataset_summaries)} combinations processed")
    
    # Write map.json
    map_file = data_dir / "map.json"
    with open(map_file, 'w') as f:
        json.dump(dataset_map, f, indent=2)
    
    print(f"\nMap file created at: {map_file}")
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    total_datasets = len(dataset_map)
    total_categories = sum(len(categories) for categories in dataset_map.values())
    total_combinations = sum(
        len(combinations) 
        for categories in dataset_map.values() 
        for combinations in categories.values()
    )
    total_samples = sum(
        combo_data["sample_count"]
        for categories in dataset_map.values()
        for combinations in categories.values()
        for combo_data in combinations.values()
    )
    total_examples = sum(
        combo_data.get("example_count", 0)
        for categories in dataset_map.values()
        for combinations in categories.values()
        for combo_data in combinations.values()
    )
    
    print(f"Total datasets: {total_datasets}")
    print(f"Total categories: {total_categories}")
    print(f"Total combinations: {total_combinations}")
    print(f"Total samples: {total_samples}")
    print(f"Total examples: {total_examples}")
    
    # Print per-dataset breakdown
    print("\n=== PER-DATASET BREAKDOWN ===")
    for dataset_name, categories in dataset_map.items():
        dataset_samples = sum(
            combo_data["sample_count"]
            for combinations in categories.values()
            for combo_data in combinations.values()
        )
        dataset_examples = sum(
            combo_data.get("example_count", 0)
            for combinations in categories.values()
            for combo_data in combinations.values()
        )
        print(f"{dataset_name}: {len(categories)} categories, {dataset_samples} samples, {dataset_examples} examples")

if __name__ == "__main__":
    process_dataset_files()