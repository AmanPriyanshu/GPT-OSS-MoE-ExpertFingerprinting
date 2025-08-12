import json
import os
from pathlib import Path

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

def fix_sorrybench_category(category_str):
    """Fix SorryBench categories by mapping from index to description."""
    if not category_str.isdigit():
        return category_str
    
    category_index = int(category_str) - 1  # Convert to 0-based index
    if 0 <= category_index < len(category_descriptions):
        return category_descriptions[category_index]
    else:
        return f"Unknown_Category_{category_str}"

def process_dataset_summaries():
    """Process all dataset summary JSON files in the raw directory."""
    raw_dir = Path("./raw")
    data_dir = Path("./data")
    
    # Create data directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)
    
    # Initialize map structure
    dataset_map = {}
    
    # Process each JSON file in raw directory
    for json_file in raw_dir.glob("*.json"):
        dataset_name = json_file.stem
        
        print(f"Processing {dataset_name}...")
        
        with open(json_file, 'r') as f:
            dataset_summaries = json.load(f)
        
        # Initialize dataset in map
        dataset_map[dataset_name] = {}
        
        # Create dataset directory
        dataset_dir = data_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
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
            
            # Create filename
            filename = f"{finished_successfully}_{has_repetition}.json"
            file_path = category_dir / filename
            
            # Write the summary data to file
            with open(file_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            # Update map
            if category not in dataset_map[dataset_name]:
                dataset_map[dataset_name][category] = {}
            
            combination_key = f"{finished_successfully}_{has_repetition}"
            dataset_map[dataset_name][category][combination_key] = {
                "sample_count": summary_data["sample_count"],
                "file_path": str(file_path.relative_to(data_dir))
            }
        
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
    
    print(f"Total datasets: {total_datasets}")
    print(f"Total categories: {total_categories}")
    print(f"Total combinations: {total_combinations}")
    print(f"Total samples: {total_samples}")
    
    # Print per-dataset breakdown
    print("\n=== PER-DATASET BREAKDOWN ===")
    for dataset_name, categories in dataset_map.items():
        dataset_samples = sum(
            combo_data["sample_count"]
            for combinations in categories.values()
            for combo_data in combinations.values()
        )
        print(f"{dataset_name}: {len(categories)} categories, {dataset_samples} samples")

if __name__ == "__main__":
    process_dataset_summaries()