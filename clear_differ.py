import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

# Topic-to-paths mapping
dataset_paths = {
    "science": [
        # GPQA datasets - science focused
        "data/gpqa_diamond/Astrophysics",
        "data/gpqa_diamond/Chemistry (general)",
        "data/gpqa_diamond/Condensed Matter Physics",
        "data/gpqa_diamond/Electromagnetism and Photonics",
        "data/gpqa_diamond/Genetics",
        "data/gpqa_diamond/High-energy particle physics",
        "data/gpqa_diamond/Inorganic Chemistry",
        "data/gpqa_diamond/Molecular Biology",
        "data/gpqa_diamond/Optics and Acoustics",
        "data/gpqa_diamond/Organic Chemistry",
        "data/gpqa_diamond/Physics (general)",
        "data/gpqa_diamond/Quantum Mechanics",
        "data/gpqa_diamond/Relativistic Mechanics",
        
        "data/gpqa_extended/Analytical Chemistry",
        "data/gpqa_extended/Astrophysics",
        "data/gpqa_extended/Chemistry (general)",
        "data/gpqa_extended/Condensed Matter Physics",
        "data/gpqa_extended/Electromagnetism and Photonics",
        "data/gpqa_extended/Genetics",
        "data/gpqa_extended/High-energy particle physics",
        "data/gpqa_extended/Inorganic Chemistry",
        "data/gpqa_extended/Molecular Biology",
        "data/gpqa_extended/Optics and Acoustics",
        "data/gpqa_extended/Organic Chemistry",
        "data/gpqa_extended/Physical Chemistry",
        "data/gpqa_extended/Physics (general)",
        "data/gpqa_extended/Quantum Mechanics",
        "data/gpqa_extended/Relativistic Mechanics",
        "data/gpqa_extended/Statistical Mechanics",
        
        "data/gpqa_main/Analytical Chemistry",
        "data/gpqa_main/Astrophysics",
        "data/gpqa_main/Chemistry (general)",
        "data/gpqa_main/Condensed Matter Physics",
        "data/gpqa_main/Electromagnetism and Photonics",
        "data/gpqa_main/Genetics",
        "data/gpqa_main/High-energy particle physics",
        "data/gpqa_main/Inorganic Chemistry",
        "data/gpqa_main/Molecular Biology",
        "data/gpqa_main/Optics and Acoustics",
        "data/gpqa_main/Organic Chemistry",
        "data/gpqa_main/Physical Chemistry",
        "data/gpqa_main/Physics (general)",
        "data/gpqa_main/Quantum Mechanics",
        "data/gpqa_main/Relativistic Mechanics",
        "data/gpqa_main/Statistical Mechanics",
        
        # MMLU science subjects
        "data/mmlu_dev/astronomy",
        "data/mmlu_dev/college_biology",
        "data/mmlu_dev/college_chemistry",
        "data/mmlu_dev/college_physics",
        "data/mmlu_dev/conceptual_physics",
        "data/mmlu_dev/electrical_engineering",
        "data/mmlu_dev/high_school_biology",
        "data/mmlu_dev/high_school_chemistry",
        "data/mmlu_dev/high_school_physics",
        "data/mmlu_dev/machine_learning",
        "data/mmlu_dev/virology",
        
        "data/mmlu_test/astronomy",
        "data/mmlu_test/college_biology",
        "data/mmlu_test/college_chemistry",
        "data/mmlu_test/college_physics",
        "data/mmlu_test/conceptual_physics",
        "data/mmlu_test/electrical_engineering",
        "data/mmlu_test/high_school_biology",
        "data/mmlu_test/high_school_chemistry",
        "data/mmlu_test/high_school_physics",
        "data/mmlu_test/machine_learning",
        "data/mmlu_test/virology",
        
        "data/mmlu_validation/astronomy",
        "data/mmlu_validation/college_biology",
        "data/mmlu_validation/college_chemistry",
        "data/mmlu_validation/college_physics",
        "data/mmlu_validation/conceptual_physics",
        "data/mmlu_validation/electrical_engineering",
        "data/mmlu_validation/high_school_biology",
        "data/mmlu_validation/high_school_chemistry",
        "data/mmlu_validation/high_school_physics",
        "data/mmlu_validation/machine_learning",
        "data/mmlu_validation/virology",
        
        # MMLU Pro science subjects
        "data/mmlu_pro_test/biology",
        "data/mmlu_pro_test/chemistry",
        "data/mmlu_pro_test/computer science",
        "data/mmlu_pro_test/engineering",
        "data/mmlu_pro_test/physics",
        
        "data/mmlu_pro_validation/biology",
        "data/mmlu_pro_validation/chemistry",
        "data/mmlu_pro_validation/computer science",
        "data/mmlu_pro_validation/engineering",
        "data/mmlu_pro_validation/physics"
    ],
    
    "math": [
        "data/mmlu_dev/abstract_algebra",
        "data/mmlu_dev/college_mathematics",
        "data/mmlu_dev/elementary_mathematics",
        "data/mmlu_dev/formal_logic",
        "data/mmlu_dev/high_school_mathematics",
        "data/mmlu_dev/high_school_statistics",
        
        "data/mmlu_test/abstract_algebra",
        "data/mmlu_test/college_mathematics",
        "data/mmlu_test/elementary_mathematics",
        "data/mmlu_test/formal_logic",
        "data/mmlu_test/high_school_mathematics",
        "data/mmlu_test/high_school_statistics",
        
        "data/mmlu_validation/abstract_algebra",
        "data/mmlu_validation/college_mathematics",
        "data/mmlu_validation/elementary_mathematics",
        "data/mmlu_validation/formal_logic",
        "data/mmlu_validation/high_school_mathematics",
        "data/mmlu_validation/high_school_statistics",
        
        "data/mmlu_pro_test/math",
        "data/mmlu_pro_validation/math"
    ],
    
    "health or medicine": [
        "data/mmlu_dev/anatomy",
        "data/mmlu_dev/clinical_knowledge",
        "data/mmlu_dev/college_medicine",
        "data/mmlu_dev/human_aging",
        "data/mmlu_dev/human_sexuality",
        "data/mmlu_dev/medical_genetics",
        "data/mmlu_dev/nutrition",
        "data/mmlu_dev/professional_medicine",
        "data/mmlu_dev/professional_psychology",
        
        "data/mmlu_test/anatomy",
        "data/mmlu_test/clinical_knowledge",
        "data/mmlu_test/college_medicine",
        "data/mmlu_test/human_aging",
        "data/mmlu_test/human_sexuality",
        "data/mmlu_test/medical_genetics",
        "data/mmlu_test/nutrition",
        "data/mmlu_test/professional_medicine",
        "data/mmlu_test/professional_psychology",
        
        "data/mmlu_validation/anatomy",
        "data/mmlu_validation/clinical_knowledge",
        "data/mmlu_validation/college_medicine",
        "data/mmlu_validation/human_aging",
        "data/mmlu_validation/human_sexuality",
        "data/mmlu_validation/medical_genetics",
        "data/mmlu_validation/nutrition",
        "data/mmlu_validation/professional_medicine",
        "data/mmlu_validation/professional_psychology",
        
        "data/mmlu_pro_test/health",
        "data/mmlu_pro_test/psychology",
        "data/mmlu_pro_validation/health",
        "data/mmlu_pro_validation/psychology"
    ],
    
    "law": [
        "data/mmlu_dev/international_law",
        "data/mmlu_dev/jurisprudence",
        "data/mmlu_dev/professional_law",
        
        "data/mmlu_test/international_law",
        "data/mmlu_test/jurisprudence",
        "data/mmlu_test/professional_law",
        
        "data/mmlu_validation/international_law",
        "data/mmlu_validation/jurisprudence",
        "data/mmlu_validation/professional_law",
        
        "data/mmlu_pro_test/law",
        "data/mmlu_pro_validation/law"
    ],
    
    "safety": [
        "data/sorry_bench_base/Advice on Adult Content",
        "data/sorry_bench_base/Advice on Discrimination Behaviors",
        "data/sorry_bench_base/Animal-related Crimes",
        "data/sorry_bench_base/Child-related Crimes",
        "data/sorry_bench_base/Conspiracy Theories Promotion",
        "data/sorry_bench_base/Dangerous Machinery Operation Advice",
        "data/sorry_bench_base/Environmental Crimes",
        "data/sorry_bench_base/Ethical Belief Promotion",
        "data/sorry_bench_base/Evasion of Law Enforcement",
        "data/sorry_bench_base/Extremist Content Promotion",
        "data/sorry_bench_base/Fake News Generation",
        "data/sorry_bench_base/False Advertising",
        "data/sorry_bench_base/False Common Knowledge Promotion",
        "data/sorry_bench_base/Financial Advice",
        "data/sorry_bench_base/Financial Crimes",
        "data/sorry_bench_base/Fraud, Scams, and Deception",
        "data/sorry_bench_base/General Advice for System Intrusion, Hacking & Cyber Attacks, Data Privacy and Confidentiality Violations",
        "data/sorry_bench_base/Governance Decision Advice",
        "data/sorry_bench_base/Harassment",
        "data/sorry_bench_base/Harmful & Unhealthy Behaviors Promotion",
        "data/sorry_bench_base/Human Elimination & World Domination",
        "data/sorry_bench_base/Illegal Crafting, Manufacturing, and Distribution",
        "data/sorry_bench_base/Impersonation",
        "data/sorry_bench_base/Intellectual Property Infringement",
        "data/sorry_bench_base/Legal Consulting Advice",
        "data/sorry_bench_base/Lewd and Obscene Words",
        "data/sorry_bench_base/Libelous Words",
        "data/sorry_bench_base/Malware Code Generation",
        "data/sorry_bench_base/Medical Advice",
        "data/sorry_bench_base/Military Use",
        "data/sorry_bench_base/Non-sexual Explicit Content Generation",
        "data/sorry_bench_base/Personal Identifiable Information Violations",
        "data/sorry_bench_base/Personal Insulting Words",
        "data/sorry_bench_base/Political Belief Promotion",
        "data/sorry_bench_base/Property Crimes",
        "data/sorry_bench_base/Public Order Crimes",
        "data/sorry_bench_base/Religion Promotion",
        "data/sorry_bench_base/Self-Harm",
        "data/sorry_bench_base/Sexual Crimes",
        "data/sorry_bench_base/Sexual Explicit Content Generation",
        "data/sorry_bench_base/Social Stereotype Promotion",
        "data/sorry_bench_base/Social-group Insulting Words",
        "data/sorry_bench_base/Terrorism",
        "data/sorry_bench_base/Threatening Words",
        "data/sorry_bench_base/Violent Crimes"
    ],
    
    "instruction_following": [
        "data/tulu3_persona_if/instruction-following"
    ]
}

def load_map_data():
    """Load the map.json file to understand dataset structure."""
    with open('./data/map.json', 'r') as f:
        return json.load(f)

def load_sample_data(file_path):
    """Load a specific sample data file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {file_path}: {e}")
        return None

def extract_dataset_category_from_path(path):
    """Extract dataset and category from a path like 'data/gpqa_diamond/Astrophysics'."""
    parts = path.split('/')
    if len(parts) >= 3:
        dataset = parts[1]  # e.g., 'gpqa_diamond'
        category = parts[2]  # e.g., 'Astrophysics'
        return dataset, category
    return None, None

def compute_expert_rankings(analytics_data, analytics_key='think_expert_analytics'):
    """Compute weighted expert rankings from analytics data."""
    layer_expert_scores = defaultdict(lambda: defaultdict(float))
    layer_total_weights = defaultdict(float)
    
    for item in analytics_data:
        if analytics_key not in item or not item[analytics_key]:
            continue
            
        weight = item.get('sample_count', 1)
        
        for layer_key, experts in item[analytics_key].items():
            layer_total_weights[layer_key] += weight
            
            for expert_id, metrics in experts.items():
                frequency = metrics.get('count_frequency', 0)
                layer_expert_scores[layer_key][expert_id] += frequency * weight
    
    # Normalize and rank
    layer_rankings = {}
    for layer_key in layer_expert_scores:
        if layer_total_weights[layer_key] > 0:
            # Calculate normalized scores
            expert_scores = []
            for expert_id, total_score in layer_expert_scores[layer_key].items():
                normalized_score = total_score / layer_total_weights[layer_key]
                expert_scores.append((int(expert_id), normalized_score))
            
            # Sort by score (descending) and extract expert IDs
            expert_scores.sort(key=lambda x: x[1], reverse=True)
            layer_rankings[layer_key] = [expert_id for expert_id, _ in expert_scores]
    
    return layer_rankings

def collect_all_data(map_data):
    """Collect all sample data across all datasets and categories."""
    all_data = []
    total_samples = 0
    
    print("Collecting all data...")
    
    for dataset_name, categories in map_data.items():
        print(f"Processing dataset: {dataset_name}")
        
        for category_name, combinations in categories.items():
            for combo_key, combo_info in combinations.items():
                file_path = f"./data/{combo_info['summary_file_path']}"
                
                sample_data = load_sample_data(file_path)
                if sample_data:
                    all_data.append(sample_data)
                    total_samples += sample_data.get('sample_count', 0)
    
    print(f"Collected {len(all_data)} files with {total_samples} total samples")
    return all_data

def collect_topic_data(map_data, topic_paths):
    """Collect data for specific topic paths."""
    topic_data = []
    total_samples = 0
    
    print(f"Collecting topic data for {len(topic_paths)} paths...")
    
    for path in topic_paths:
        dataset, category = extract_dataset_category_from_path(path)
        if not dataset or not category:
            continue
            
        if dataset not in map_data or category not in map_data[dataset]:
            print(f"Warning: {dataset}/{category} not found in map data")
            continue
            
        combinations = map_data[dataset][category]
        for combo_key, combo_info in combinations.items():
            file_path = f"./data/{combo_info['summary_file_path']}"
            
            sample_data = load_sample_data(file_path)
            if sample_data:
                topic_data.append(sample_data)
                total_samples += sample_data.get('sample_count', 0)
    
    print(f"Collected {len(topic_data)} files with {total_samples} total samples for topic")
    return topic_data

def invert_expert_rankings(rankings):
    """Invert expert rankings for unsafety analysis."""
    inverted = {}
    for layer_key, experts in rankings.items():
        # Simply reverse the order
        inverted[layer_key] = experts[::-1]
    return inverted

def create_alternating_rankings(topic_rankings, all_rankings):
    """Create alternating rankings: topic_expert, all_expert, topic_expert, ..."""
    alternating = {}
    
    for layer_key in range(24):
        layer_str = f"layer_{layer_key}"
        
        topic_experts = topic_rankings.get(layer_str, [])
        all_experts = all_rankings.get(layer_str, [])
        
        if not topic_experts and not all_experts:
            continue
            
        alternating_list = []
        used_experts = set()
        
        # Alternate between topic and all rankings
        max_length = max(len(topic_experts), len(all_experts))
        
        for i in range(max_length):
            # Add topic expert if available and not used
            if i < len(topic_experts) and topic_experts[i] not in used_experts:
                alternating_list.append(topic_experts[i])
                used_experts.add(topic_experts[i])
            
            # Add all expert if available and not used
            if i < len(all_experts) and all_experts[i] not in used_experts:
                alternating_list.append(all_experts[i])
                used_experts.add(all_experts[i])
        
        if alternating_list:
            alternating[layer_str] = alternating_list
    
    return alternating

def main():
    """Main function to process all topics and create analytics."""
    
    # Create output directory
    output_dir = Path('./topical_analytics')
    output_dir.mkdir(exist_ok=True)
    
    # Load map data
    print("Loading map data...")
    map_data = load_map_data()
    
    # Step 1: Compute "all" baseline
    print("\n=== STEP 1: Computing baseline (all data) ===")
    all_data = collect_all_data(map_data)
    
    if not all_data:
        print("Error: No data collected!")
        return
    
    # Compute rankings for both think and answer analytics
    print("Computing think expert rankings...")
    all_think_rankings = compute_expert_rankings(all_data, 'think_expert_analytics')
    
    print("Computing answer expert rankings...")
    all_answer_rankings = compute_expert_rankings(all_data, 'answer_expert_analytics')
    
    # Save all rankings
    all_analytics = {
        'think_expert_analytics': all_think_rankings,
        'answer_expert_analytics': all_answer_rankings
    }
    
    with open(output_dir / 'all.json', 'w') as f:
        json.dump(all_analytics, f, indent=2)
    
    print(f"Saved baseline analytics to {output_dir / 'all.json'}")
    
    # Step 2: Process each topic
    topics = ["science", "math", "health or medicine", "law", "safety", "instruction_following"]
    
    for topic in topics:
        print(f"\n=== STEP 2: Processing topic '{topic}' ===")
        
        if topic not in dataset_paths:
            print(f"Warning: No paths defined for topic '{topic}'")
            continue
        
        # Collect topic data
        topic_data = collect_topic_data(map_data, dataset_paths[topic])
        
        if not topic_data:
            print(f"Warning: No data collected for topic '{topic}'")
            continue
        
        # Compute topic rankings
        topic_think_rankings = compute_expert_rankings(topic_data, 'think_expert_analytics')
        topic_answer_rankings = compute_expert_rankings(topic_data, 'answer_expert_analytics')
        
        # Create alternating rankings
        alternating_think = create_alternating_rankings(topic_think_rankings, all_think_rankings)
        alternating_answer = create_alternating_rankings(topic_answer_rankings, all_answer_rankings)
        
        # Save topic analytics
        topic_analytics = {
            'think_expert_analytics': alternating_think,
            'answer_expert_analytics': alternating_answer,
            'metadata': {
                'total_samples': sum(item.get('sample_count', 0) for item in topic_data),
                'num_files': len(topic_data),
                'paths_count': len(dataset_paths[topic])
            }
        }
        
        # Use underscore for filename
        filename = topic.replace(' ', '_').replace(' or ', '_').lower() + '.json'
        
        with open(output_dir / filename, 'w') as f:
            json.dump(topic_analytics, f, indent=2)
        
        print(f"Saved {topic} analytics to {output_dir / filename}")
        print(f"  - {topic_analytics['metadata']['total_samples']} samples")
        print(f"  - {topic_analytics['metadata']['num_files']} files")
    
    # Step 3: Create unsafety analytics (inverted safety)
    print(f"\n=== STEP 3: Creating unsafety analytics ===")
    
    safety_file = output_dir / 'safety.json'
    if safety_file.exists():
        with open(safety_file, 'r') as f:
            safety_data = json.load(f)
        
        # Create inverted rankings
        unsafety_think = invert_expert_rankings(safety_data['think_expert_analytics'])
        unsafety_answer = invert_expert_rankings(safety_data['answer_expert_analytics'])
        
        unsafety_analytics = {
            'think_expert_analytics': unsafety_think,
            'answer_expert_analytics': unsafety_answer,
            'metadata': {
                **safety_data['metadata'],
                'note': 'Inverted safety rankings - experts good at being unsafe/harmful'
            }
        }
        
        with open(output_dir / 'unsafety.json', 'w') as f:
            json.dump(unsafety_analytics, f, indent=2)
        
        print(f"Saved unsafety analytics to {output_dir / 'unsafety.json'}")
    else:
        print("Warning: Could not create unsafety analytics - safety.json not found")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Created analytics for:")
    print(f"  - Baseline (all data)")
    
    for topic in topics:
        filename = topic.replace(' ', '_').replace(' or ', '_').lower() + '.json'
        if (output_dir / filename).exists():
            print(f"  - {topic}")
    
    if (output_dir / 'unsafety.json').exists():
        print(f"  - unsafety (inverted safety)")
    
    print(f"\nAll files saved to: {output_dir}")

if __name__ == "__main__":
    main()