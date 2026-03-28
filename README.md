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
     
---
## Report Generation

### Example 1 — Basal Cell Carcinoma 

**Ground Truth:**
“Ulcerated basal cell carcinoma, of the sclerodermiform histological type, on active pre-damaged skin.”

**Generated (DermPath-LLaMA):**
 “Basal cell carcinoma.Sclerodermiform, basal cell carcinoma, completely excided temporal right. The preparation margins are tumor-free. Adnexal findings: Actinic
keratosis with mild atypia and a small melanocytic nevus in the area of the spindle tip at 6 o’clock.The latter is removed from the healthy side in the healthy. A biopsy of the skin without any evidence of mycobacterium tuberculosis infection. Ulcerated basal cell carcinoma, sclerodermiform type,completely removed on the whole (basal cell carcinoma).Adnexal findings: actinic keratosis, moderately differentiated with focal rim formation, completely removed on both sides in toto.Seborrhoic keratosis.”

**Evaluation Metrics:**

* Keyword overlap: 0.857
* SciSpaCy entities: 0.5
* Cosine similarity: 0.944
* Perplexity: 23.44

**Interpretation:**

* Correct identification of **diagnosis (BCC)** and **subtype (sclerodermiform)**
* Captures relevant pathological context (ulceration, actinic damage)
* Tends to **expand with additional plausible findings** beyond the ground truth

---

### Example 2 — No Malignancy 

**Ground Truth:**
“Tumor-free resection (6 o’clock).”

**Generated (DermPath-LLaMA):**
“Skin/subcutaneous excised with lowgrade solar elastosis and scar area, without evidence of malignancy, especially no indication of possible residues of basal
cell carcinoma. No melanoma residues detectable in the present material (minimal distance to lateral resection margin 0.6 mm).No evidence for amyloidosis.No
evidence for granulomatous inflammation.No evidence for sarcoidosis.No evidence of vasculitis, e.g. Churg-Strauss syndrome or Wegener’s disease. No evidence
for rheumatoid arthritis.No evidence for lupus erythematosus.No evidence for scleroderma.No evidence for foreign body reaction, especially no indication for siliconeinduced foreign body reaction.No evidence for actinic keratosis.”

**Evaluation Metrics:**

* Keyword overlap: 0.0
* SciSpaCy entities: 0.0
* Cosine similarity: 0.846
* Perplexity: 9.34

**Interpretation:**

* Correctly predicts **absence of malignancy**
* However, generates **overly long and generic differential statements**
* Includes clinically irrelevant conditions → **over-generation issue**

---

### Key Observations

* The model produces **coherent and medically plausible reports**
* High semantic similarity even when wording differs significantly
* Strong performance on **clear malignant cases**
* Limitations in:

  * concise reporting
  * avoiding unnecessary clinical expansions

---

### Clinical Perspective

* Suitable as a **drafting assistant** for pathologists
* Requires **human validation and editing**
* Future improvements:

  * report length control
  * terminology normalization
  * uncertainty-aware generation

