# Iris Recognition System using CNN (CASIA Dataset)

## Project Overview
This project is an iris recognition system using deep learning. It detects iris patterns and identifies individuals using a CNN model trained on the CASIA dataset.

---

## Tech Stack
- Python
- OpenCV
- NumPy
- TensorFlow / Keras
- Matplotlib

---

## Dataset
- CASIA Iris Dataset
- ~108 classes (subjects)
- Each class represents one person

---

## How to Run
1. Install dependencies:
```
pip install -r requirements.txt
```

2. Run the project:
```
python src/Main.py
```

---

## Project Workflow
- Load dataset
- Preprocess images
- Detect iris using Hough Circle Transform
- Train CNN model
- Predict iris identity

---

## Output
- Model predicts person ID from iris image
- Displays accuracy/loss during training

---

## Project Structure
- src/ → Main code
- tests/ → Test images
- outputs/ → Results
- model/ → Trained model

---

## Author
Pravallika Maddala