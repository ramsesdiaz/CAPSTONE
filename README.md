<div align="center">

# CAPSTONE PROJECT

## Non-Invasive Brain-Computer Interface (BCI) Directional Control System

# Final Report

**UC Berkeley - Engineering**

**Professional Certificate in Machine Learning and Artificial Intelligence**

by Ramses Diaz de Leon

</div>


## 1. Define the Problem Statement

The goal of this project is to develop a highly reliable, non-invasive Brain-Computer Interface (**BCI**) machine learning pipeline capable of translating real-time neurophysiological activity into directional mouse commands (**Up, Down, Left, Right, One Click, Double Click**).

### The Challenges
Electroencephalography (*EEG*) signals are notoriously non-stationary, suffer from a low signal-to-noise ratio and are heavily prone to environmental and physiological artifacts (*such as unintended muscular movements*). Traditional interfaces struggle to maintain high predictive stability across sequential time domains, creating sluggish or inaccurate device control.

### Potential Benefits
Successfully decoding motor imagery (*imagined movement*) enables hands-free environmental interaction. This serves as a foundational framework for assistive technologies—such as neuro-controlled PC/mouse, wheelchairs or any other type of remote control technology, empowering individuals with severe motor impairments to regain some control.

---

## 2. Model Outcomes or Predictions

This system is designed as a **supervised machine learning classification** challenge. The models takes in segmented windows of multi-channel neurophysiological time-series data and map them into discrete, categorical target outcomes.

### Expected Output
The system yields a predicted intent label matching one of four primary navigational directives (*Up, Down, Left, Right*). Additionally, it supports downstream multi-class sorting to isolate secondary control triggers (Clench and Eye_Blink) from the active mental imagery streams to be used as One Click and Double Click respectively.

---

## 3. Data Acquisition
The dataset combines time-series data captured across 8 electrode arrays positioned across the scalp based on the 10-20 map. To robustly capture distinct types of neurophysiological indicators, data was acquired from multiple states:

1) **Mental Imagery**: Dedicated sessions where the subject imagined localized motor actions without physical execution to generate pure Up, Down, Left, and Right control arrays.
2) **Artifact Control**: Intentional collections of mechanical ocular triggers (*Eye_Blink*) and myogenic anomalies (*Jaw Clench*) to act as non-imagery hardware system keys.
 
### Data Potential Assessment
Initial exploratory data analysis confirmed high spectral and spatial variances across the electrode nodes. While spatial mapping configurations natively cluster physical artifacts distinctly, the subtle variations in pure mental imagery require advanced mathematical tracking to reliably disentangle the overlapping boundaries.

---

## 4. Data Preprocessing/Preparation

**a. Cleaning, Missing Values, & Inconsistencies**
The raw continuous EEG signals were sliced using a 5.0-second rolling window epochs. The first 2 seconds correspond to active intentional movements (Real and Imagined) whereas the remaining 3 seconds correspond to intentional Rest. To handle signal inconsistencies and sudden localized noise spikes, a rigorous Peak-to-Peak amplitude thresholding check was implemented. Any epoch exhibiting an absolute voltage fluctuation exceeding standard physiological boundaries was rejected, ensuring classifiers were trained exclusively on clean, artifact-free neurophysiological data.

**b. Train/Test Data Splitting**
To evaluate genuine model generalization and prevent data leakage, the filtered datasets were split using a 75% training and 25% testing allocation. This partition was executed using a Stratified shuffle, ensuring the exact proportional representation of each directional class was maintained perfectly across both the training and validation subsets.

**c. Analysis and Encoding Steps**
Target text strings (*Up, Down, etc..*) were transformed into numeric classifications using a standard LabelEncoder layer. For flat feature configurations, data scaling was performed via a StandardScaler pipeline to eliminate magnitude biases across disparate feature units.

---

## 5. Modeling
Three distinct architectures were implemented and evaluated in attempt to solve the navigational problem statement:

- **Pipeline A (Optimized Euclidean Flat-Space Framework)**: This approach mapped the signal windows into standard vector spaces by manually extracting 219 distinct statistical time-domain and frequency-domain metrics (including RMS, Variance, and specific Alpha/Theta band powers) across individual channels.

- **Pipeline B (Riemannian Geometric-Manifold Framework)**: This approach bypassed traditional flat statistics entirely. Instead, it estimated robust covariance matrices directly from the raw EEG streams to capture non-linear spatial correlations, global network phase shapes and synchronized timing variations across the electrode array before projecting them onto a flat geometric plane using Tangent Space mapping.

- **Pipeline C (Deep Neural Networks Framework)**: This approach eliminated both manual feature engineering and geometric matrix projections. Instead, it utilized a convolutional neural network to process overlapping signal windows and normalized raw waveforms, allowing the model to automatically learn and extract complex spatial-temporal patterns directly from the brainwave data. Moreover, this pipeline used ELU (Exponential Linear Unit) activation function for its intermediate convolutional and depthwise-separable layers, paired with a Softmax activation function at the final dense output layer to compute class probabilities across the navigation commands.

---

## 6. Model Evaluation

The predictive performance of all three frameworks was tracked using robust validation splitting, cross-validation, and multi-class classification reports.

### Classifiers
For the feature-based and manifold pipelines (**Pipelines A and B**), four traditional machine learning algorithms were cross-evaluated: **Support Vector Machines** (SVM_RBF), **Random Forests** (RandomForest), **Gradient Boosting** (GradientBoosting), and **K-Nearest Neighbors** (KNN). For Pipeline C, a **Convolutional Neural Network** (EEGNet) was trained to learn hierarchical spatial-temporal features directly from the data.

### Metrics & Optimal Model Determination
An extensive hyperparameter grid search (*evaluating 56 unique parameter candidate combinations across 280 fits*) was executed for the classical pipelines to isolate peak boundary performance. For the deep learning pipeline, specialized temporal-spatial optimizations including channel-wise Z-score scaling and 75% sliding window overlaps were applied. The optimal model for each framework was determined by evaluating peak overall validation accuracy alongside precision and recall balances.

| Pipeline | Selected Model | Cross-Validation Accuracy |
| :--- | :--- | :--- |
| Pipeline A: Euclidean Flat-Space | SVM_RBF (C=5, Gamma='scale') | 89.37% |
| Pipeline B: Riemannian Manifold | RandomForest | 77.73% |
| Pipeline C: Deep Neural Networks | EEGNet (2D-CNN + Z-Score + Overlap) | 78.59% |

### Findings

- **The Winner!**: The Optimized **Pipeline A Euclidean SVM** drastically outperformed Riemannian and the Deep Neural Network alternatives, securing an **89.37%** mean accuracy. The non-linear boundaries established by the optimized RBF kernel proved highly capable of separating complex variations across the 219 flat features, reducing intra-class overlap.

- **Riemannian**: While Pipeline B RandomForest achieved a lower overall accuracy of **77.73%**, it demonstrated incredible structural safety. It produced a agreat **93.0% F1-score** for **Down** command and **0.0% false-positive leakage** from mental imagery into the mechanical hardware triggers (Clench and Eye_Blink). This confirms that the geometric tangent space provides excellent isolation for high-impact control commands.

- **Deep Neural Network**: Achieving an accuracy of **78.59%**, the convolutional neural network demonstrated that deep learning can successfully extract spatial-temporal wave patterns directly from raw EEG without any manual feature engineering. While its performance placed it slightly above the Riemannian approach, it did not surpass the highly tuned Euclidean SVM, illustrating the common challenge of training deep networks on smaller BCI sample sizes.

### Visualization Insights
1) **Confusion Matrix Analysis**: The final 6x6 BCI confusion matrix confirmed clean class segregation with minor localized leakage remaining only between specific imagery pairs (such as Up bleeding slightly into Right).

2) **Dimensionality Compression**: PCA-compressed 2D decision boundary plots illustrated that while Riemannian manifold provided isolated clustering for specific directives like Left, the optimized Euclidean pipeline successfully resolved localized mixing that tripped up the baseline tangent configurations.

---

## 7. Next Step & Suggestions
The next step would be to deploy Pipeline A model into a live, **real-time** stream control session!

---

### Link to the Notebook
You can view the full analysis here: [Final Analysis Report](./Capstone_V1p17.ipynb)

---

### Repository Structure
```text
├── Capstone_V1p17.ipynb                    # Main Jupyter Notebook with Python code
├── data/                                   # Folder containing all the EEG raw data for this analysis
│   ├── new_filtered_master_dataframe.csv.gz
│   ├── OpenBCI-RAW-DOE3_Signal_1_Re.txt
│   ├── OpenBCI-RAW-DOE3_Signal_2_Re.txt
│   ├── OpenBCI-RAW-DOE3_Signal_3_Im.txt
│   ├── OpenBCI-RAW-DOE3_Signal_3_Re.txt
│   ├── OpenBCI-RAW-DOE3_Signal_4_Im.txt
│   ├── OpenBCI-RAW-DOE3_Signal_4_Re.txt
│   ├── OpenBCI-RAW-DOE3_Signal_5_Im.txt
│   ├── OpenBCI-RAW-DOE3_Signal_5_Re.txt
│   ├── OpenBCI-RAW-DOE3_Signal_6_Im.txt
│   ├── OpenBCI-RAW-DOE3_Signal_6_Re.txt
├── images/                                 # Folder containing images
│   ├── 10_20_Electrode_Position2.png
│   ├── Pipeline_A_Optimized_Euclidean_BCI_Control_Plot.png
│   ├── Pipeline_B_Riemannian_BCI_Control_Plot.png
└── README.md                               # Project report and documentation (this file)
