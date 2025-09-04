# CS771: Introduction to Machine Learning (Autumn 2024)

## Course Mini-Projects Repository

This repository contains implementations for the mini-projects completed as part of the CS771 Introduction to Machine Learning course at IIT Kanpur.

## Projects Overview

### Mini-Project 1: Multi-Feature Binary Classification
**Objective:** Develop binary classifiers for three different feature representations of the same dataset and investigate ensemble approaches for improved performance.

**Key Components:**
- **Task 1:** Individual model development for each feature representation
  - Emoticon features (13 categorical features)
  - Deep features (13×786 dimensional embeddings)
  - Text sequence features (50-digit string representations)
  
- **Task 2:** Combined multi-feature ensemble learning
  - Feature fusion strategies
  - Model ensemble techniques
  - Cross-representation learning

**Technical Constraints:**
- Maximum 10,000 trainable parameters per model
- Training data efficiency analysis (20%, 40%, 60%, 80%, 100% subsets)
- Validation-based hyperparameter tuning

**Implementation Highlights:**
- Comprehensive feature engineering and transformation techniques
- Multiple classification algorithms including logistic regression, SVM, and neural networks
- Systematic ablation studies on training data requirements
- Ensemble methods for multi-representation learning

---

### Mini-Project 2: Multi-Armed Bandits
**Objective:** Implement and compare various multi-armed bandit algorithms for different reward distributions and evaluate their regret minimization performance.

**Key Components:**
- **Task 1:** Bernoulli Bandits with Known Arms
  - Epsilon-Greedy (multiple ε values)
  - Upper Confidence Bound (UCB)
  - Thompson Sampling
  - KL-UCB
  
- **Task 2:** Gaussian Bandits with Known Arms
  - Adapted algorithms for continuous reward distributions
  - Variance-aware confidence bounds
  
- **Task 3:** Successive Elimination for Unknown Arms
  - Adaptive arm elimination strategies
  - Theoretical regret guarantees

**Implementation Highlights:**
- Complete implementation of 5+ bandit algorithms
- Comprehensive experiment framework with statistical analysis
- Regret analysis across multiple time horizons
- Configurable difficulty levels (easy/medium/hard instances)
- Automated visualization and reporting tools

## Repository Structure

```
CS771/
├── mini_project_1/         # Binary classification project
│   ├── data/              # Dataset files (not included)
│   ├── models/            # Model implementations
│   ├── utils/             # Helper functions
│   └── main.py            # Main execution script
│
├── mini_project_2/         # Multi-armed bandits project
│   ├── bandit_algorithms.py    # Algorithm implementations
│   ├── bandit_environments.py  # Environment simulators
│   ├── experiments.py          # Experiment framework
│   ├── main.py                 # Main execution script
│   └── requirements.txt        # Python dependencies
│
├── mini-project-1.pdf     # Project 1 specifications
├── mini-project-2.pdf     # Project 2 specifications
└── README.md             # This file
```

## Installation

### Prerequisites
- Python 3.8+
- NumPy, SciPy, Matplotlib
- Scikit-learn
- Additional dependencies as specified in project-specific requirements

### Setup
```bash
# Clone repository
git clone https://github.com/Vrajb24/CS771.git
cd CS771

# Install dependencies for Mini-Project 2
cd mini_project_2
pip install -r requirements.txt
```

## Usage

### Mini-Project 1: Binary Classification
```bash
cd mini_project_1
# Note: Dataset files need to be downloaded separately
python main.py  # Run all experiments
```

### Mini-Project 2: Multi-Armed Bandits
```bash
cd mini_project_2

# Run full experiments
python main.py

# Run quick demo
python main.py demo

# Run specific experiments
python run_experiments.py --task 1 --runs 20 --horizon 1000
```

## Key Results

### Mini-Project 1
- Achieved high accuracy across all three feature representations
- Demonstrated that ensemble methods can leverage complementary information from different representations
- Identified optimal training data requirements for each feature type

### Mini-Project 2
- **Thompson Sampling** consistently demonstrated superior performance across different bandit settings
- **KL-UCB** showed particularly strong performance for Bernoulli bandits
- **Successive Elimination** effectively identified optimal arms with theoretical guarantees
- Comprehensive regret analysis validated theoretical bounds

## Implementation Details

### Algorithms Implemented

**Classification (Project 1):**
- Logistic Regression with various regularization schemes
- Support Vector Machines (linear and kernel-based)
- Neural Networks (within parameter constraints)
- Ensemble methods (voting, stacking)

**Bandit Algorithms (Project 2):**
- Epsilon-Greedy with adaptive exploration
- Upper Confidence Bound (UCB) with optimal confidence intervals
- Thompson Sampling with conjugate priors
- KL-UCB with divergence-based bounds
- Successive Elimination with adaptive thresholds

### Evaluation Metrics

- **Project 1:** Classification accuracy, data efficiency curves, ensemble performance gain
- **Project 2:** Cumulative regret, convergence rates, arm identification accuracy

## Technical Contributions

1. **Modular Design:** Clean separation between algorithms, environments, and experiments
2. **Reproducibility:** Seeded random number generation for consistent results
3. **Scalability:** Efficient implementations supporting various problem sizes
4. **Visualization:** Comprehensive plotting utilities for result analysis
5. **Statistical Analysis:** Confidence intervals and significance testing

## Course Information

**Course:** CS771 - Introduction to Machine Learning  
**Term:** Autumn 2024  
**Institution:** Indian Institute of Technology Kanpur

## Author

**Name:** Vraj Patel 
**Email:** vrajb24@iitk.ac.in  
**GitHub:** [Vrajb24](https://github.com/Vrajb24)

## Acknowledgments

- Course instructors for project specifications and guidance
- Course TAs for technical support
- Peer collaborators for valuable discussions

## License

This repository is for educational purposes as part of the CS771 course at IIT Kanpur. Please refer to the course guidelines for usage and distribution policies.

---

*Note: Dataset files for Mini-Project 1 are not included in this repository due to course restrictions. The code for Mini-Project 1 requires these datasets to be downloaded separately as per course instructions.*
