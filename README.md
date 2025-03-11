**CNN Model for Habitat Classification using iNaturalist Dataset**  

This project focuses on image classification using a Convolutional Neural Network(CNN)to categorize habitat types from the iNaturalist dataset. The model uses Keras & TensorFlow, implementing a deep learning pipeline that explores hyperparameter tuning, data augmentation, and model evaluation.

**Project Overview** 
- Developed a 5-layer CNN model with convolutional, ReLU activation, and max-pooling layers.
- Experimented with hyperparameter tuning, including filter sizes, dropout rates, and batch normalization.
- Trained and evaluated the model on the iNaturalist dataset, classifying 10 habitat types.
- Analyzed accuracy vs. number of filters to find the optimal model configuration.

**Features** 
**Custom CNN Architecture:**  
- 5 Convolutional Layers with adjustable filters **(32, 64, 128, 256, 512).**  
- Dropout & Batch Normalization for improved generalization.  
- ReLU Activation & Max-Pooling for feature extraction.  
- Dense Layer with Softmax Activation for classification.  

**Hyperparameter Exploration:**  
- Number of filters per layer: 32, 64, 128, 256, 512 
- Dropout Rates: 20%, 30%  
- Data Augmentation: Yes/No  
- Batch Normalization: Yes/No  

**Performance Evaluation:**  
- Compared accuracy vs. filter sizes and accuracy vs. hyperparameters.  
- Generated visualizations including CNN filter activations.  

**Installation**  
To set up the environment and run the model, follow these steps:  

  1. **Clone this repository:**  
     ```bash
     git clone https://github.com/your-username/CNN-Habitat-Classifier.git
     cd CNN-Habitat-Classifier
  2. **Install dependencies**
      ```bash
     pip install -r requirements.txt
  4. **Run the Jupyter Notebook**
      ```bash
     jupyter notebook

**Dataset**
The model is trained on the iNaturalist dataset, which contains labeled images of 10 different habitat types. The dataset consists of a train-test split, with 10% of training data reserved for hyperparameter tuning.

**Results**
Best Model Configuration: **(To be updated based on results)**
Accuracy Achieved: **XX%**
Visualizations:
10 × 3 grid of test images with model predictions.
CNN filter activations plotted in an 8 × 8 grid.
