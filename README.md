# Malnutrition Risk Prediction System

A desktop application for early malnutrition risk screening, developed as an undergraduate final-year research project at Ladoke Akintola University of Technology (LAUTECH), Ogbomoso. The system uses fingerprint biometric features combined with demographic data to predict an individual's risk of malnutrition, without requiring clinical laboratory equipment.

---

## The Problem

Malnutrition remains a serious public health challenge in low-resource settings, particularly in sub-Saharan Africa. One in three children under five in the region experiences stunting due to chronic undernutrition. Early detection is critical, but traditional methods — comparing weight to height, measuring mid-upper arm circumference — require trained personnel and equipment that are often unavailable in community settings.

This project explores a different question: can fingerprint biometric features, which are non-invasive and require only a scanner, carry enough predictive signal to support early malnutrition screening?

---

## Approach

The system integrates three datasets into a unified predictive pipeline:

- **SOCOFing (Sokoto Coventry Fingerprint Dataset)** — a Nigerian fingerprint dataset from which features were extracted, including ridge density, minutiae count, and pattern type
- **NHANES (National Health and Nutrition Examination Survey)** — used to infer demographic profiles (age and sex) from fingerprint characteristics
- **WHO Z-Score Dataset** — used to derive nutritional outcome labels (BMI, MUAC, HAZ, WHZ, WAZ) based on established anthropometric thresholds

These were merged into a single dataset of 600 individuals combining biometric features, demographic attributes, and malnutrition status labels. A logistic regression model was then trained and evaluated on this combined dataset.

The pipeline is structured as follows:

```
SOCOFing fingerprint images
        ↓
Feature extraction (ridge density, minutiae count, pattern type)
        ↓
Demographic inference (age, sex) via NHANES reference data
        ↓
Nutritional outcome labelling via WHO Z-score thresholds
        ↓
Combined dataset (600 individuals)
        ↓
Logistic regression model → Malnutrition risk prediction
```

---

## Model Performance

| Metric     | Result |
|------------|--------|
| Accuracy   | 91.67% |
| ROC-AUC    | 0.8735 |
| Precision  | 63.64% |
| Recall     | 53.85% |

The model shows strong overall discriminative ability (ROC-AUC of 0.87) but moderate recall — meaning some malnourished individuals were missed. This is an important limitation: in a screening context, missing at-risk individuals is a significant concern, and improving recall would be a priority in further development.

Despite weak individual correlations between fingerprint features and nutritional outcomes, the multivariate model confirmed that these features can collectively provide useful predictive signals.

---

## Desktop Application

The application (`Malnutrition Prediction System.py`) is a full desktop GUI built with **PyQt6**, with light and dark theme support and background prediction threading via `QThread` to keep the interface responsive during model inference.

**The interface has four screens:**

- **Manual Entry** — enter patient biometric and demographic values directly (ridge density, minutiae count, BMI, MUAC, gender, pattern type)
- **Simulated Scan** — simulate a fingerprint scan with a progressive scanning animation, generating synthetic feature values for demonstration
- **Upload Image** — upload a real fingerprint image; the system extracts features automatically using OpenCV and scikit-image (adaptive thresholding, skeletonisation, and a crossing-number algorithm for minutiae detection)
- **Prediction History** — view, refresh, and export all previous predictions to CSV

Predictions are classified into four categories based on probability: **Well-Nourished**, **Not At Risk**, **At Risk**, and **Malnourished**. Results are saved automatically to `malnutrition_predictions.csv`.

---

## Tools and Libraries

- Python
- pandas, NumPy
- scikit-learn (Logistic Regression, Pipeline, StandardScaler, SimpleImputer)
- OpenCV (`cv2`) — image loading and preprocessing
- scikit-image — skeletonisation and adaptive thresholding for fingerprint feature extraction
- matplotlib, seaborn — data visualisation
- PyQt6 — desktop GUI with light/dark theme switching and background threading

---

## How to Run

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python "Malnutrition Prediction System.py"
```

---

## Limitations

This project is a feasibility study, not a clinical tool. Several important limitations apply:

- **Constructed data pipeline**: Because no dataset directly links fingerprint features to malnutrition status in the literature, the dataset was built by mapping SOCOFing fingerprint features to demographic and nutritional data via NHANES and WHO reference standards. This is a methodological constraint inherent to the research question, and the scope is explicitly framed as proof of concept.
- **Moderate recall**: The model missed roughly 46% of malnourished individuals in the test set. In a real screening context, this would need substantial improvement before deployment could be considered.
- **No live sensor integration**: The system does not connect to physical fingerprint scanners. Deployment in clinical or community settings would require this, along with ethical review and validation on real patient data.
- **Dataset size**: 600 records is a small dataset for training a generalisable model. Larger, real-world datasets would be needed for meaningful validation.

---

## What I Learned

This project taught me that research is often shaped as much by data availability as by the research question itself.

One of the biggest challenges was that there is no publicly available dataset linking fingerprint biometric features directly to malnutrition outcomes. Exploring the question meant working across multiple datasets and understanding how different sources could be used together to investigate a problem that has received very little attention.

It also changed the way I think about model performance. The model achieved an accuracy of 91.67%, which initially seemed encouraging. But in a screening context, recall was the metric that mattered most. Missing a large proportion of at-risk individuals would limit the usefulness of the system regardless of its overall accuracy.

More than anything, the project reinforced the importance of looking beyond performance scores and thinking carefully about how a model would be used in practice. In healthcare, a useful model is not simply one that performs well on a test set, but one that supports better decisions for the people it is intended to serve.

---

## Future Directions

This project was designed as a proof of concept rather than a deployable healthcare tool. Several areas would be worth exploring in future work.

* Validation using real-world datasets that directly link fingerprint biometric features and nutritional outcomes
* Testing additional machine learning models and approaches to improve recall
* Exploring threshold optimisation for screening scenarios where identifying at-risk individuals is prioritised over overall accuracy
* Integration with fingerprint sensing hardware for real-world data collection
* Investigation of whether biometric features can contribute to broader health screening applications beyond malnutrition

More broadly, I am interested in how non-invasive data sources might be used alongside traditional health information to support earlier detection and intervention in resource-constrained settings.


---

## Research Context

This project was completed as part of my undergraduate studies in Computer Science at Ladoke Akintola University of Technology (LAUTECH), Ogbomoso.

It was developed between November 2024 and September 2025 under the supervision of Prof. W. O. Ismaila and reflects my growing interest in healthcare analytics, machine learning, and the use of data-driven approaches to address public health challenges.

---

## Author

**Faith Olaniyi**  
Computer Science Graduate | Data Science & AI for Healthcare  
[LinkedIn](https://www.linkedin.com/in/faith-oluwanifemi-olaniyi) · [GitHub](https://github.com/faithopia21)
