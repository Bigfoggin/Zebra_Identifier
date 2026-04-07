***

# Project Documentation: Zebra Crossing Identifier
**Developer:** deiccoalessi  
**Date:** April 2026  
**System:** Palladium Cluster (FHGR)  
**Hardware:** NVIDIA A100 GPU  

---

## 1. Project Overview
The goal of this project was to develop a Deep Learning model capable of identifying zebra crossings from aerial imagery (SwissImage). The project utilized Transfer Learning and was deployed on a high-performance computing cluster.

## 2. Infrastructure & Environment
* **Environment:** Python 3.12 virtual environment (`.venv`).
* **Core Libraries:** `torch`, `torchvision`, `scikit-learn`, `seaborn`, `matplotlib`.
* **Cluster Management:** Managed via **SLURM** scheduler. 
    * Jobs were submitted using `.sh` batch scripts to ensure fair resource allocation.
    * Specific hardware requested: 32 CPUs, 200GB RAM, and 1x NVIDIA A100 GPU.

## 3. Development Phases

### Phase I: Data Preprocessing
* **Dataset Structure:** Images were organized into a `split_data/` directory with `train`, `val`, and `test` partitions.
* **Labeling:** Binary classification with folders named `y` (Zebra) and `n` (No Zebra).
* **Transforms:** Images were resized to $224 \times 224$ pixels and normalized using ImageNet statistics to match the pre-trained model requirements.

### Phase II: Model Training
* **Architecture:** **ResNet-18** (Residual Network).
* **Customization:** The final fully connected (fc) layer was replaced with a linear layer mapping to 2 output features.
* **Training Process:** Executed via `train.sh` on the A100 GPU.
* **Result:** Initial logs reported a training accuracy of **~99%**.

### Phase III: Troubleshooting & Validation
During the validation phase, several technical hurdles were overcome:
* **Path Resolution:** Fixed a directory naming mismatch between the training output and validation input (`sebra_classifier`).
* **Compatibility:** Addressed an NVIDIA driver version warning by implementing a **CPU-fallback** for the validation script to ensure stable performance during inference.
* **Execution:** Created `validation.py` to generate quantitative metrics.

## 4. Performance Metrics
The model was evaluated on a validation set of 2,400 images.

| Metric | Class 'n' (No Zebra) | Class 'y' (Zebra) |
| :--- | :--- | :--- |
| **Precision** | 1.00 | 0.72 |
| **Recall** | 0.99 | 0.93 |
| **F1-Score** | 0.99 | 0.81 |
| **Total Accuracy** | | **99%** |

**Analysis:** The high **Recall (0.93)** for zebras is particularly significant for a safety-related task, indicating that the model successfully identified nearly all instances of the target class despite the heavy class imbalance.

---

## 5. Error Analysis & Discovery
To understand the model's logic, a diagnostic script `wrong_image.py` was developed to isolate **False Positives** and **False Negatives**.

### Key Finding: The "AI Audit"
Upon visual inspection of the `error_analysis/false_alarms` folder:
* **Human Error Discovery:** 5 out of the 29 images flagged as "False Positives" were actually **correctly identified zebra crossings**. 
* **Conclusion:** These images had been accidentally mislabelled as "n" (No Zebra) during the manual dataset preparation phase. 
* **Significance:** The model demonstrated superior consistency compared to the human labeler, effectively "cleaning" its own training data and proving that the true precision of the model is higher than the initial statistics suggested.

## 6. Future Work
* **Dataset Cleaning:** Re-incorporate the corrected labels into the training set.
* **Weight Balancing:** Implement `WeightedCrossEntropyLoss` to further penalize the remaining false alarms.
* **Deployment:** Export the model to ONNX format for real-time inference applications.

***