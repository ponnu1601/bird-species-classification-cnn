# bird-species-classification-cnn
## **Overview**
This project explores automated bird species identification using deep learning techniques. Audio recordings are converted to spectrograms and processed through CNN architectures to classify 12 different bird species common in the Seattle area. The study compares multiple model architectures and evaluates performance on both controlled test sets and external mystery audio files.
## **Key Findings**
### **Binary Clasification**
- Achieved perfect 100% accuracy when distinguishing between two different bird types
- Models learned quickly and needed only 3-4 training rounds to reach optimal performance
- Simple models worked just as well as complex ones for this two-species task
### **Multi-class Classification**
- Best model achieved 69.8% accuracy when classifying all 12 bird species
- Models with regularization techniques showed more stable and reliable training behavior
- Training accuracy was much higher than test accuracy, indicating models memorized rather than truly learned
- Some bird species are much easier to distinguish from their calls than others
- Models showed overconfidence in predictions on new audio files due to limited training data
## **Dataset**
- Data from Xeno-Canto bird sound archive
- 12 bird species from Seattle area
- 579 total spectrogram samples (256×343 pixels)
## **Limitations**
- Small dataset (~40 samples per species)
- Significant overfitting in multi-class models
- External predictions cannot be verified without knowing the actual bird species in the test files
- High confidence scores may not reflect actual accuracy
## **How to Run**
- Clone the repository
- Install dependencies: pip install tensorflow librosa numpy pandas matplotlib seaborn scikit-learn h5py scipy
- For similar data, visit [Xeno-Canto](https://xeno-canto.org/)
- Open notebooks/bird_sound_classification.ipynb
- Run all cells to reproduce results
