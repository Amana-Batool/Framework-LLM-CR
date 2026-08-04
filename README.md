# Schema-Guided LLM Verification and Conflict Resolution between

# LLM-Based and Data-Driven Causal Discovery

### Preparation

#### Environmental Setup

We mainly use the following key libraries:

causal\_learn==0.1.4.0

numpy==2.1.3

openai==2.38.0

scikit-learn==1.6.1

torch==2.12.0

python==3.13.5

### Datasets

The experimental data are located under the DATA folder.

### LLMs
Interact with LLMs using your own API key.

## Running the Code

Before running the scripts, update the input file paths and configuration parameters at the beginning of each Python file. Place the observational dataset and metadata in the expected directories.

Run the scripts from the repository root in the following order:

```bash
python schema_generator.py
python response_generator.py
python verifier.py
python semantic_entropy.py
python llm_prior_calculation.py
python bic_calculation.py
python hsic_calculation.py
python combined_score.py
python llm_recovery.py
