# ML project - Vertebral Column Classification

This project classifies orthopedic patients as either `Normal` or `Abnormal` based on six biomechanical measurements of the spine and pelvis.

## Project Structure

```text
├── data/dataset_spine.csv    # Biomechanical dataset (310 samples, 6 features, 1 target)
├── main.py              # Main training, evaluation, and grid-search pipeline
├── REPORT.md            # Detailed project analysis, comparison, and results (in Romanian)
├── README.md            # Project overview and run guide (this file)
├── output/                 # Output folder containing generated Confusion Matrices and ROC Curves
└── .venv/               # Local Python virtual environment
```

## Dataset Features

The classification uses the following 6 biomechanical parameters:

1. `pelvic_incidence`
2. `pelvic_tilt`
3. `lumbar_lordosis_angle`
4. `sacral_slope`
5. `pelvic_radius`
6. `grade_of_spondylolisthesis`

Target class: `Class_att` (`Normal` / `Abnormal`)

---

## Installation & Setup

### 1. Prerequisites

Make sure you have Python 3.9+ installed on your system.

### 2. Create and Activate Virtual Environment

Create a virtual environment to manage dependencies locally:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

Install the required machine learning packages:

```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn
```

---

## How to Run

Execute the main pipeline using the virtual environment python interpreter:

```bash
# With activated virtual environment:
python main.py

# Or directly using the relative path:
.venv/bin/python main.py
```

### Outputs

Running `main.py` will print detailed console outputs including descriptive stats, cross-validation results, hyperparameter grid-search outcomes, classification reports, and then generate plots inside the `output/` folder:

- Confusion Matrix plots: `output/confusion_matrix_{Model_Name}.png`
- ROC Curve plots: `output/roc_curve_{Model_Name}.png`

---

## Results Summary (Test Set)

| Model                    | Accuracy | F1 Macro | ROC AUC |
| ------------------------ | -------: | -------: | ------: |
| **Decision Tree**        |   0.8871 |   0.8725 |  0.9369 |
| **MLP (Neural Net)**     |   0.8871 |   0.8652 |  0.9560 |
| **kNN**                  |   0.8548 |   0.8267 |  0.8655 |
| **Gaussian Naive Bayes** |   0.7903 |   0.7773 |  0.8786 |

For a full analysis, parameter tuning details, and stability metrics, please refer to REPORT.md
