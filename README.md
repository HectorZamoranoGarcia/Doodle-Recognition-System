# Doodle Recognition System

Project focused on recognizing hand-drawn patterns ("doodles") using distinct machine learning approaches implemented **from scratch** in **Python**.

![Python 3.10](https://img.shields.io/badge/python-v3.10+-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/NumPy-from--scratch-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-Interactive--UI-FFBB00?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Analytics-ffffff?style=for-the-badge&logo=python&logoColor=3670A0)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)

## Description

This project aims to detect and classify 16x16 pixel drawings into 10 categories (Sun, Moon, Tree, etc.) using linear and probabilistic models.
The main focus was not just to use high-level libraries, but to manually implement the mathematical core of the algorithms (Gradient Descent, Softmax, Perceptron Logic) using **NumPy**, creating a custom dataset, and deploying a real-time interactive interface.
Techniques such as image dilation were applied to solve stroke thickness issues, and both deterministic and probabilistic models were compared.

## Objectives

* Create a custom dataset ("MyDoodles") by drawing directly on a digital canvas.
* Implement **Perceptron** and **Logistic Regression** algorithms from scratch (without PyTorch/TensorFlow).
* Verify the **Linear Separability** of the data through iterative testing.
* Develop a real-time web application to test the models with live drawings.

## Data

**Main Source:** Custom dataset "MyDoodles" (Created by the author)

* **Format:** NumPy Arrays (`.npy`)
* **Dimensions:** 16x16 pixels flattened to vectors of size 256.
* **Classes:** 10 distinct categories (Smile, Sad, Heart, Star, Sun, Moon, Cloud, Tree, House, Lightning).
* **Size:** Balanced dataset with stratified splits.

**Preprocessing**

* **Grayscale Conversion:** Transformation of RGB input to 1-channel matrices.
* **Inversion & Binarization:** Handling white backgrounds vs. black ink.
* **Dilation Filter:** Application of a "MaxFilter" to simulate a thicker brush stroke, ensuring the 16x16 reduction retains the drawing's features.

## Methodology

### 1. Data Exploration

Visual analysis of the raw pixel data and verification of class balance. Setup of the training and testing environments using stratified splitting.

### 2. Linear Experiments (Perceptron)

Implementation of the classic Perceptron algorithm.

* **Hyperparameter Tuning:** Tested Margin ($b$) and Learning Rate ($\alpha$).
* **Finding:** A large margin ($b=100$) significantly reduced overfitting, improving generalization from 80% to ~95% accuracy.

### 3. Probabilistic Classification (Logistic Regression)

Implementation of the **Softmax** function and Cross-Entropy Loss (Negative Log Likelihood).

* Unlike the Perceptron, this model outputs a probability distribution (e.g., "85% confidence it is a Sun").
* Optimization using **Mini-Batch Stochastic Gradient Descent (SGD)**.

### 4. Linear Separability Test

An exhaustive test using two nested loops (Epochs vs. Samples) to verify if the dataset could be perfectly separated by a hyperplane.

* **Result:** The data is not perfectly linearly separable (error never reaches 0 on the full set), justifying the need for probabilistic or non-linear approaches.

### 5. Real-Time Application

Development of a Gradio interface (`src/app.py`) that loads the trained weights (`.npy`) and performs inference on live user drawings.

## Results

| Model | Approach | Best Config | Observations |
| :--- | :--- | :--- | :--- |
| **Perceptron** | Deterministic | $b=100, \alpha=1.0$ | Fast, but rigid. Struggles with noisy inputs. |
| **Logistic Reg.** | Probabilistic | $\eta=0.1, bs=10$ | Provides confidence scores. More robust to ambiguous drawings. |

The **Logistic Regression** model proved to be superior for the interactive app, as it allows the system to express uncertainty rather than making random guesses on empty or messy canvases.

## Repository Structure

```text
Handdrawn-Shape-Classifier/
├── data/
│   ├── doodles_X.npy      # Raw pixel data
│   └── doodles_y.npy      # Labels
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_linear_experiments.ipynb    # Perceptron & Margins
│   ├── 03_logistic_regression.ipynb   # Probabilities & Gradient Descent
│   └── 04_linear_separability_test.ipynb
├── models/
│   ├── perceptron_my_doodles.npy
│   └── logistic_weights.npy
├── src/
│   ├── app.py             # Main Gradio Application
│   ├── capture_tool.py    # Data generation tool
│   └── core_model.py      # Mathematical functions
└── README.md
```

## Technologies

* **Language:** Python 3.10+
* **Libraries:**
  * **NumPy:** Mathematical operations and matrix manipulation.
  * **Matplotlib:** Data visualization and plotting.
  * **Gradio:** Web GUI for the interactive demo.
  * **Scikit-learn:** Dataset splitting and metrics.
  * **PIL (Pillow):** Image processing (filters and resizing).
* **Tools:** VS Code, Git/GitHub.

## Execution

### 1. Install Dependencies

Ensure you have Python installed, then run:

```bash
pip install numpy matplotlib gradio scikit-learn pillow
```

### 2. Train the Models (Optional)
If you want to retrain the system from scratch, run the notebooks in the `notebooks/` folder in order. This will generate your own `.npy` weight files in the `models/` directory.

### 3. Run the App
Launch the interactive interface to test the classifier:

```bash
cd src
python app.py
```

## Author

**Héctor Zamorano García**

* **Reproducibility:** A fixed random seed (`23`) was used throughout the project to ensure consistent results.
* **Context:** Project developed as part of a *Machine Learning

## Acknowledgments & Inspiration

* **AI Assistance:** This project was developed with the assistance of **GitHub Copilot** as a coding companion for generating snippets, debugging, and documenting code.
* **Academic Roots:** The architecture and mathematical foundation are inspired by laboratory exercises (*Prácticas*) from the **Intelligent Systems (SIN)** course at the **Universitat Politècnica de València (UPV)**.
* **Creative Adaptation:** While the original academic exercises focused on standard handwritten digit recognition (0-9), this project **modifies and expands** that concept to recognize **hand-drawn emojis/doodles**. This shift introduces new challenges in data variance and requires a custom-built dataset, moving beyond pre-packaged standard libraries.





