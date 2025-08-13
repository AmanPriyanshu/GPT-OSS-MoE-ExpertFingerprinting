# GPT-OSS MoE Expert Fingerprinting & Specialized Model Pruning
ExpertFingerprinting: Behavioral Pattern Analysis and Specialization Mapping of Experts in GPT-OSS-20B's Mixture-of-Experts Architecture

**Interactive Tools:**
- [Expert Analytics Dashboard](https://amanpriyanshu.github.io/GPT-OSS-MoE-ExpertFingerprinting/) - Token-level visualization and domain analysis
- [Layer Comparison Tool](https://amanpriyanshu.github.io/GPT-OSS-MoE-ExpertFingerprinting/comparison.html) - Deep expert pattern comparison

**Model Collections:**
- [Main Collection](https://huggingface.co/collections/AmanPriyanshu/gpt-oss-pruned-experts-42b-20b-if-science-math-etc-689c380a366950b1787a20c6) - All 232 specialized models
- [General Purpose](https://huggingface.co/collections/AmanPriyanshu/gpt-oss-general-42b-to-20b-689c2f39e338447e6c2074a5)
- [Science](https://huggingface.co/collections/AmanPriyanshu/gpt-oss-science-42b-to-20b-689c31bc4bb505f3fc2d1357)
- [Mathematics](https://huggingface.co/collections/AmanPriyanshu/gpt-oss-math-42b-to-20b-689c33870a351af956e26353)
- [Health & Medicine](https://huggingface.co/collections/AmanPriyanshu/gpt-oss-health-medicine-42b-to-20b-689c3511bd3bb683170677c1)
- [Law](https://huggingface.co/collections/AmanPriyanshu/gpt-oss-law-42b-to-20b-689c38ed56e4ab048965b1d9)
- [Safety](https://huggingface.co/collections/AmanPriyanshu/gpt-oss-safety-42b-to-20b-689c970d3d5f4b5b6045bcc3)
- [Instruction Following](https://huggingface.co/collections/AmanPriyanshu/gpt-oss-instruction-following-42b-to-20b-689c98a2bb1fe0eaa2287b18)
- [Harmful/Red-team](https://huggingface.co/collections/AmanPriyanshu/gpt-oss-harmful-42b-to-20b-689ca990732ef4453da307e0)

Each collection contains 29 models with the following parameter counts: 4.2B, 4.8B, 5.4B, 6.0B, 6.6B, 7.2B, 7.8B, 8.4B, 9.0B, 9.6B, 10.2B, 10.8B, 11.4B, 12.0B, 12.6B, 13.1B, 13.7B, 14.3B, 14.9B, 15.5B, 16.1B, 16.7B, 17.3B, 17.9B, 18.5B, 19.1B, 19.7B, 20.3B, and 20.9B parameters, offering flexibility for different deployment scenarios.

## Project Overview

This project attempts to conduct an in-depth investigation into expert activation patterns within GPT-OSS-20B's Mixture-of-Experts (MoE) architecture. Through analysis of router decisions across diverse evaluation benchmarks, we've created specialized, resource-efficient models through expert pruning.

**Key Achievements:**
- **232 Specialized Models Released** across 8 domains and 29 expert configurations each
- **Interactive Analysis Tools** for real-time expert pattern exploration
- **Domain-Specific Optimization** maintaining performance while reducing computational overhead
- **Comprehensive Evaluation** across GPQA, MMLU, SORRY-Bench, Tulu3, and Polyglot benchmarks

## Methodology

### Expert Activation Analysis

Our approach begins with router analysis across the original GPT-OSS-20B model:

1. **Token-Level Tracking**: We use all 24 layers to capture router decisions for every generated token
2. **Multi-Domain Evaluation**: We look at scientific reasoning, mathematical computation, legal knowledge, medical understanding, safety evaluation, instruction following, and general capabilities
3. **Pattern Recognition**: Statistical aggregation allows us to know which experts consistently activate for specific task types

### Systematic Expert Pruning

Based on activation patterns, we implement a data-driven pruning strategy:

**Domain Specialization**: Eight distinct specialization tracks:
- **General**: Broad capability preservation across all domains
- **Science**: Physics, chemistry, biology reasoning (GPQA-focused)
- **Mathematics**: Quantitative reasoning and problem-solving
- **Health/Medicine**: Clinical knowledge and medical reasoning
- **Law**: Legal frameworks and jurisprudence
- **Safety**: Harm detection and responsible AI patterns
- **Instruction Following**: Constraint satisfaction and formatting adherence
- **Harmful**: Inverted safety patterns for red-teaming research

### Technical Implementation

**Model Architecture Preservation**:
- Maintains original 24-layer transformer structure
- Preserves 128K context length and attention patterns
- Retains RoPE positional encoding and RMSNorm
- Uses BF16 precision for optimal memory efficiency

**Pruning Process**:
1. **Expert Selection**: Top-performing experts identified per layer per domain
2. **Weight Extraction**: Router and expert weights carefully preserved
3. **Architecture Adjustment**: Configuration updated for reduced expert count
4. **Validation**: Functionality testing across representative prompts

## Model Collections

We've systematically released **232 specialized models** organized into domain-specific collections:

### Main Collection
Complete overview of all 232 specialized models across domains and configurations.

### Domain-Specific Collections:

- **[General Purpose](https://huggingface.co/collections/AmanPriyanshu/gpt-oss-general-42b-to-20b-689c2f39e338447e6c2074a5)** (4.2B-20B): Broad capability models for versatile applications
- **[Science](https://huggingface.co/collections/AmanPriyanshu/gpt-oss-science-42b-to-20b-689c31bc4bb505f3fc2d1357)** (4.2B-20B): Optimized for scientific reasoning and technical knowledge
- **[Health & Medicine](https://huggingface.co/collections/AmanPriyanshu/gpt-oss-health-medicine-42b-to-20b-689c3511bd3bb683170677c1)** (4.2B-20B): Specialized for medical and clinical applications
- **[Mathematics](https://huggingface.co/collections/AmanPriyanshu/gpt-oss-math-42b-to-20b-689c33870a351af956e26353)** (4.2B-20B): Enhanced quantitative reasoning and problem-solving
- **[Law](https://huggingface.co/collections/AmanPriyanshu/gpt-oss-law-42b-to-20b-689c38ed56e4ab048965b1d9)** (4.2B-20B): Legal knowledge and jurisprudential reasoning
- **[Safety](https://huggingface.co/collections/AmanPriyanshu/gpt-oss-safety-42b-to-20b-689c970d3d5f4b5b6045bcc3)** (4.2B-20B): Harm detection and responsible AI deployment
- **[Instruction Following](https://huggingface.co/collections/AmanPriyanshu/gpt-oss-instruction-following-42b-to-20b-689c98a2bb1fe0eaa2287b18)** (4.2B-20B): Precise constraint satisfaction and formatting
- **[Harmful/Red-team](https://huggingface.co/collections/AmanPriyanshu/gpt-oss-harmful-42b-to-20b-689ca990732ef4453da307e0)** (4.2B-20B): Research models with inverted safety patterns

Each collection contains 29 models ranging from 4.2B to 20B parameters, offering flexibility for different deployment scenarios.

We've developed comprehensive web-based tools for exploring expert activation patterns:

### [Expert Analytics Dashboard](https://amanpriyanshu.github.io/GPT-OSS-MoE-ExpertFingerprinting/)
- **Token-Level Visualization**: Interactive exploration of expert routing decisions
- **Domain Analysis**: Compare activation patterns across different task types
- **Statistical Aggregation**: View top-performing experts by layer and domain
- **Real-time Filtering**: Analyze completed vs. incomplete generations

### [Comparison Tool](https://amanpriyanshu.github.io/GPT-OSS-MoE-ExpertFingerprinting/comparison.html)
- **Layer-by-Layer Analysis**: Deep comparison of expert patterns between configurations
- **Statistical Significance**: Quantify differences in expert usage across domains
- **Visual Charting**: Interactive graphs showing expert activation distributions
- **Export Functionality**: Download analysis results for further research

*Visit these tools at [https://amanpriyanshu.github.io/GPT-OSS-MoE-ExpertFingerprinting/](https://amanpriyanshu.github.io/GPT-OSS-MoE-ExpertFingerprinting/) and [https://amanpriyanshu.github.io/GPT-OSS-MoE-ExpertFingerprinting/comparison.html](https://amanpriyanshu.github.io/GPT-OSS-MoE-ExpertFingerprinting/comparison.html) to interact with the full dataset and explore expert behavior patterns in detail.*

## Technical Details

### Model Configuration

All pruned models maintain compatibility with the original GPT-OSS architecture:

- **Precision**: BF16 for optimal memory/performance balance
- **Top-k Routing**: Dynamically adjusted to `min(4, num_experts)`
- **Context Length**: Full 128K token support preserved
- **Attention Pattern**: Alternating dense/sliding window maintained

## ## Citation

If you use this work in your research, please cite:

```bibtex
@misc{priyanshu2025gptoss,
  title={GPT-OSS MoE Expert Fingerprinting: Analyzing Expert Activation Patterns in Mixture of Experts Models},
  author={Priyanshu, Aman and Vijay, Supriti},
  year={2025},
  howpublished={\url{https://amanpriyanshu.github.io/GPT-OSS-MoE-ExpertFingerprinting/}},
  note={Interactive analysis tool and systematic expert pruning for MoE architectures}
}
```

## Contributing

We welcome contributions to extend this research:

- **Additional Domains**: Propose new specialization areas
- **Pruning Strategies**: Alternative expert selection methodologies
- **Evaluation Metrics**: Novel assessment approaches for MoE models
- **Tool Enhancement**: Improvements to analysis interfaces

## Repository Structure

```
├── inference_on_prompts.py    # Batch inference with router analysis
├── mini_model_creator.py      # Expert pruning and model generation
├── recorder.py               # Router activation recording utilities
├── expert_recorder.py        # Specialized expert tracking
├── index.html               # Main analysis dashboard
├── comparison.html          # Layer comparison tool
└── topical_analytics/       # Domain-specific expert rankings
    ├── all.json
    ├── science.json
    ├── math.json
    ├── health_or_medicine.json
    ├── law.json
    ├── safety.json
    ├── instruction_following.json
    └── harmful.json
```

## Acknowledgments

- **OpenAI**: For releasing the GPT-OSS-20B model and enabling this research
- **Hugging Face**: For hosting infrastructure and model distribution
- **Research Community**: For evaluation benchmarks and methodological foundations

---

**Explore the interactive tools at [https://amanpriyanshu.github.io/GPT-OSS-MoE-ExpertFingerprinting/](https://amanpriyanshu.github.io/GPT-OSS-MoE-ExpertFingerprinting/) to dive deeper into expert activation patterns and model comparisons.**
