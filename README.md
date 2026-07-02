# CAPSTONE PROJECT

**UC Berkeley - Engineering**

**Professional Certificate in Machine Learning and Artificial Intelligence**

by Ramses Diaz de Leon

### Project Description

This project is intended to address the computational challenge of decoding non-invasive electroencephalography (EEG) raw data signal into discrete, actionable commands through the development of a robust machine learning algorithm. By using an 8-electrode array based on the 10-20 international system, the primary objective is to design, train, and evaluate a model capable of accurately mapping distinct neural oscillations to specific directional Mouse cursor movements and click events.

### Executive Summary
This summary outlines the performance comparison between 2 pipelines, Euclidean (Pipeline A) and Riemannian (Pipeline B), for a Brain-Computer Interface (BCI).

The comparison between feature spaces revealed distinct behaviors. While **Pipeline B** leveraged the raw geometric structure of the covariance matrices to map spatial relationships between electrodes, **Pipeline A** relied on flat feature extraction combined with non-linear kernel optimization. Surprisingly, hyperparameter optimization allowed the Euclidean space to aggressively map decision boundaries, ultimately outperforming the Riemannian Tangent Space for this specific dataset configuration.

---

### EEG Signals
EEG frequency bands are categorized by their oscillation frequency, usually measured in Hertz (Hz). These rhythms reflect the collective neural activity of the brain in different states (i.e. attention, relaxation, etc...)

**These are the different EEG frequency bands of interest:**

- **Delta (0.5 – 4 Hz)**: These are high-amplitude, slow waves observed predominantly during deep, dreamless sleep (non-REM sleep)
- **Theta (4 – 7 Hz)**: Associated with drowsiness, light sleep and deep meditative states. Theta activity can also emerge during tasks requiring intense internal focus or emotional processing
- **Alpha (7 – 13 Hz)**: These waves are prominent in the posterior regions of the brain during relaxed wakefulness with closed eyes. They often dissipate when the individual opens their eyes or begins processing new sensory information
- **Beta (13 – 30 Hz)**: These signify active thinking, alert focus, and cognitive engagement. Beta waves are often associated with active problem-solving, decision-making, and sensorimotor processing

The 2 frequency bands of focus for this study are: **Theta** and **Alpha** bands

---

### Data Source

**Raw EEG Data**: Time-series voltage (uV) fluctuations captured from 8 electrode channels sampled at 250Hz
- Synchronized log recordings were collected from the following user's intended actions:

1) **Up** = Real and Imagined movements
2) **Down** = Real and Imagined movements
3) **Right** = Real and Imagined movements
4) **Left** = Real and Imagined movements
5) **Left Mouse Click** = Real Jaw Clench
6) **Double Mouse Click** = Real Double Eye Blink

- **HW Used**: OpenBCI 8-Bit Board, 8 Dry Electrodes and BraiNet 10/20 Electrode Cap
- **Data Sets**: The data collected is divided in 2 sets:

    - **Dataset 1**: This first dataset contains 1 minute of trial recordings per movement to setup the code. The first 10 seconds of recording were ignored to account for HW setup, then recordings were divided into 5 second epochs, where the first 2 seconds are   intentional movements (Real and/or Imagined) and the remaining 3 seconds are at rest. Intentional movements were collected at seconds 10, 15, 20, 25, 30, 35, 40, 45, 50 & 55 for a total of 10 recordings per movement.

    - **Dataset 2**: Once the code was setup. I collected a much larger dataset consisting of 3 minutes of recordings per movement (Up, Down, Right, Left, Clench & Eye Blink) and per signal (Real and Imagined). Similar to dataset 1, the first and last 10 seconds of recording were ignored to account for HW setup, then the rest of the recordings were divided into 5 second epochs, where the first 2 seconds are intentional movements (Real and/or Imagined) and the remaining 3 seconds are at rest, for a total of 32 intentional recordings per movements and per signal. The txt files from these recordings were huge and in order to enable upload into Github I had to post process the files, concatenate them and extract them into a csv file, which I later had to further compress. This final dataset is the one I'm using build, train and test the ML models.

---

### Techniques Used in this Analysis

I attempted two different approaches for comparison:

1) **Pipeline A: Optimized Euclidean Space Pipeline**

- **Artifact Rejection**: Peak-to-Peak amplitude thresholding to drop corrupted hardware epochs across 2.0 second sliding windows.
- **Feature Extraction**: Manual statistical extraction of 219 distinct flat features across time-domain and frequency-domain metrics mapped per individual channel.
- **Hyperparameter Optimization**: Automated Grid Search cross-validation across 56 unique candidate parameter pairs (totaling 280 fits) to isolate peak boundary conditions.
- **Classification**: Optimized Support Vector Machine with a Radial Basis Function kernel (C=5, γ=’scale’) to maximize non-linear class separation.

2) **Pipeline B: Baseline Riemannian Space Pipeline**

- **Artifact Rejection**: Identical Peak-to-Peak amplitude thresholding across 2.0 second windows to maintain validation parity.
- **Feature Extraction**: Estimation of robust Covariance Matrices directly from raw EEG data streams to isolate spatial correlations and synchronization boundaries across the electrode array.
- **Manifold Projection**: Vectorization via Tangent Space mapping to project curved, non-linear Riemannian distances onto a flat geometric plane without warping structural information.
- **Classification**: Baseline Random Forest Classifier (n_estimators=200) trained directly on manifold tangent vectors.

---

### CONCLUSION

**Model Performance Comparison**

| Pipeline | Best Model | Accuracy |
| :--- | :--- | :--- |
| A: Euclidean | Optimized SVM_RBF (C=5) | 89.37% |
| B: Riemannian | RandomForest | 77.73% |

- **Pipeline A (Euclidean Space)**:
This approach utilized a rich set of calculated flat features. Hyperparameter grid search proved to be powerful, maximizing class separation and achieving a peak cross-validation accuracy of **89.37%** (and an impressive **96.0%** on the validation split).

- **Pipeline B (Riemannian Manifold)**:
By projecting covariance matrices into the Tangent Space, the un-tuned baseline RandomForest classifier captured distinct spatial geometry, yielding a stable **77.73%** mean accuracy. While highly robust—offering a balanced precision/recall profile and an exceptional **93.0%** F1-score for the Down command—the linear mapping into the tangent plane did not match the flexible thresholds established by the optimized Euclidean SVM.

- **In conclusion**, while the Riemannian Manifold architecture offers deep geometric elegance and flawless hardware trigger isolation, the Optimized Euclidean Space framework demonstrated superior predictive accuracy and generalization capabilities for this dataset, making it the mathematically recommended framework for deployment.

---

### NEXT STEPS & RECOMMENDATIONS

Proceed with the final goal; use the recommended framework to try Real-Time signal processing but not the intention of this Capstone project assignment.

---

### Link to the Notebook
You can view the full analysis here: [Final Analysis Report](./Capstone_V1p15.ipynb)

---

### Repository Structure
```text
├── Capstone_V1p15.ipynb                    # Main Jupyter Notebook with Python code
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
