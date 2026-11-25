# Phishing URL Detection Using Machine Learning

## Project Overview

Phishing is a common cybersecurity threat in which attackers create deceptive URLs to trick users into visiting fraudulent websites and sharing sensitive information.

This project uses Machine Learning to classify URLs as **Phishing** or **Legitimate** based on characteristics extracted directly from the URL.

A Logistic Regression model is trained using URL-based features and integrated into a Streamlit application for interactive predictions.

## Project Objectives

- Extract meaningful features from input URLs.
- Identify URL characteristics associated with phishing activities.
- Train a Logistic Regression classification model.
- Predict the probability of a URL belonging to the phishing or legitimate class.
- Build an interactive web application using Streamlit.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## Machine Learning Workflow

1. Dataset loading and inspection.
2. Selection of features that can be extracted directly from URLs.
3. Data preprocessing.
4. Feature scaling and encoding, where required.
5. Training a Logistic Regression model.
6. Model evaluation.
7. Saving the trained model and preprocessing objects.
8. Integrating the model into a Streamlit application.

## URL Features

The model uses URL characteristics such as:

- URL Length
- Domain Length
- HTTPS Availability
- IP Address Usage
- Number of Subdomains
- Obfuscated Characters
- Special Character Counts
- Number of Digits
- URL Character Ratios

These features help the model identify patterns commonly associated with phishing URLs.

## Application Features

- Accepts a URL as input.
- Extracts URL-based features automatically.
- Applies the saved preprocessing steps.
- Predicts whether the URL is phishing or legitimate.
- Displays the predicted class probabilities.

## Project Structure

```text
Phishing-URL-Detection/
│
├── app.py
├── train_model.py
├── requirements.txt
├── URL_Only_Model.pkl
├── README.md
└── .gitignore
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd Phishing-URL-Detection
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app.py
```

## click Here to run the application
https://phishing-url-detection-using-machine-learning-cvpmkc2uvu7xc7xo.streamlit.app/

## Important Note

This application provides machine learning-based predictions using URL characteristics. A prediction is not a guarantee that a website is safe or malicious. Always exercise caution when visiting unfamiliar websites.

## Future Improvements

- Experiment with additional classification algorithms.
- Improve model generalisation on previously unseen domains.
- Add explainability to show which URL characteristics influence predictions.
- Evaluate probability calibration and model performance.

## Author

Developed as a Machine Learning project to explore URL-based phishing detection and practical model deployment using Streamlit.
