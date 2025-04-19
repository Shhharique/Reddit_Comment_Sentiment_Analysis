# Reddit_Comment_Sentiment_Analysis

## Description
This project aims to build a deep learning model that detects emotions in social media posts (tweets, captions, etc.). By analyzing the text, the model classifies it into predefined emotion categories such as *happy, **sad, **angry, etc. The model leverages **TF-IDF* for text vectorization and a *neural network* built with *TensorFlow* and *Keras* for classification.

## Tech Stack
- *Python* (for coding)
- *pandas* (for data manipulation)
- *numpy* (for numerical computing)
- *matplotlib* & *seaborn* (for data visualization)
- *nltk* (for text processing)
- *scikit-learn* (for machine learning)
- *TensorFlow* & *Keras* (for deep learning)

## Installation
1. Clone the repository:
    bash
    git clone https://github.com/your-username/Emotion-Detection-ML.git
    

2. Navigate to the project folder:
    bash
    cd Emotion-Detection-ML
    

3. Install dependencies:
    bash
    pip install -r requirements.txt
    

4. Run the notebook:
    bash
    jupyter notebook
    

## Project Structure
- **emotion_detection.ipynb**: The main code where the data is loaded, cleaned, and used to train the model.
- **requirements.txt**: A file listing all the necessary libraries to run the project.
- **dataset.csv**: The dataset used for training and testing the model.
- **README.md**: This file!

## Accuracy & Results
- The model achieved an accuracy of *85%* on the test data.
- Below is the *confusion matrix* for the model's predictions:

![Confusion Matrix](path_to_your_confusion_matrix_image.png)

## Future Scope
- Extend the model to handle multi-language text.
- Use transformer models like *BERT* to improve accuracy.
- Apply this model in real-time applications like sentiment analysis for customer feedback.

## Acknowledgements
- Thanks to my college and *Cranes Varsity* for the amazing support and guidance throughout this project.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
