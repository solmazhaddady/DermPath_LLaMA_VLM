# DermPath_LLaMA
Vision-language model for automated histopathology report generation from whole-slide images (WSI)

## Background

Digital pathology enables computational analysis of whole-slide images (WSIs), 
but several challenges remain:

- Gigapixel-scale images
- Limited annotated datasets
- Complex and heterogeneous cancer patterns
- Subjective labeling variability 
- Time-consuming report writing for pathologists

This project explores how vision–language models can address these challenges.

## Contributions

- Develped a two-stage vision–language pipeline for dermatopathology
- Implemented Weakly supervised slide-level classification (MLP, Perceiver Resampler)
- Leveraged pretrained CTransPath encoder for efficient WSI feature extraction
- Integrated visual features with a medical LLM (MMed-LLaMA-3-8B) for report generation
- Applied parameter-efficient fine-tuning (LoRA) for large language models
- Evaluated and compared performance with HistoGPT baseline

  ## Pipeline

WSI → Patch Extraction → CTransPath Features  
→ Feature Aggregation (Perceiver / MLP)  
→ Slide-level Diagnosis  
→ LLM → Report Generation

## Results
1. Main Classification Results
Slide-level Diagnosis (3 Classes)
- Dataset: Validation set (n = 568)   https://github.com/solmazhaddady/NMSC-TCIA-Dataset
- Classes: BCC, SCC, No malignancy
- Accuracy: 96.65%
  
 Key Observations:

- Very high performance across all classes
- Minimal confusion between BCC and SCC
- Most errors occur between:
- BCC ↔ No malignancy

Clinical Interpretation:

- Errors mainly occur in borderline or subtle cases
- Small tumor regions or early lesions are harder to detect
- Some false positives correspond to:
  --precancerous lesions
  -- reactive atypia

👉 This is actually clinically meaningful, not just “model error”

![Confusion Matrix](results/fd_confusion_rownorm_Valset.png)

---

2. ROC Analysis
     
. BCC: AUC = 0.998
   
. SCC: AUC = 0.997

. No malignancy: AUC = 0.995

👉 Interpretation:

. Near-perfect separability between classes

. Model produces well-calibrated ranking scores

![ROC Analysis](results/fd_roc_ovr_val.png)

---

3. Sub Type Classification 
   
. BCC & SCC Subtyping

. Modeled as multi-label classification

. Separate heads for:
    .BCC subtypes
    .SCC subtypes
    
---

Key Insights

### BCC:

. High recall for common subtype (solid)

. Lower precision for:
  . superficial
  . micronodular

👉 due to overlapping patterns + label subjectivity 


### SCC:

. Hardest classification task

. Challenges:
  . grading (G1 vs G2)
  . in-situ vs invasive

👉 But:

. ROC-AUC shows strong feature learning
. Model captures meaningful structure despite threshold issues

![Sub Type Classification ](results/cm_top_both_rowpct(subtypes).png)

---
Clinical Insight (VERY IMPORTANT ⭐)

. Rare subtypes are difficult even for experts

. Performance reflects:

  .. limited data
  
  .. real diagnostic ambiguity

👉 This is a strong point, not a weakness
     

