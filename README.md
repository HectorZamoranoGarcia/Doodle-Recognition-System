# Hand-Drawn Doodle Classifier

A small Python project that classifies ten hand-drawn doodle classes from 16×16 pixel inputs. The notebooks cover data exploration, perceptron experiments, logistic regression, and evaluation. The model calculations use NumPy rather than a deep-learning framework.

![Python](https://img.shields.io/badge/Python-3.10%2B-3670A0?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/Model-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Gradio](https://img.shields.io/badge/Demo-Gradio-FFBB00?style=flat-square)

## Demo

The Gradio app preprocesses a drawing, converts it to a 16×16 feature vector, and passes it to the saved logistic-regression weights.

## Run it

Install the app dependencies:

```bash
python -m pip install numpy pillow gradio
```

Start the app from the `src` directory so its relative model paths resolve:

```bash
cd src
python app.py
```

The notebooks use Matplotlib and scikit-learn in addition to NumPy. Run them from the `notebooks` folder to explore the data or train and evaluate the models.

## Repository structure

```text
data/
├── doodles_X.npy
└── doodles_y.npy
models/
├── logistic_weights.npy
└── perceptron_my_doodles.npy
notebooks/
├── 01_data_exploration.ipynb
├── 02_linear_experiments.ipynb
├── 03_model_training.ipynb
└── 04_evaluation_metrics.ipynb
src/
├── app.py
├── capture_tool.py
└── core_model.py
```

## Notes

The dataset contains drawings made for this project. The model is a small demonstration and has not been evaluated on a separate external dataset.

GitHub Copilot assisted with code snippets, debugging, and documentation during development.

## Author

Héctor Zamorano García

## License

MIT. See [LICENSE](LICENSE).

