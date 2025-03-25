# Riddle-ML-Model

An intelligent machine learning model designed to solve and generate riddles. This project leverages natural language processing (NLP) techniques to create an interactive experience where users can input riddles and receive accurate answers or generate new ones.

## Features

- **Riddle Solver**: Input a riddle, and the model will predict its answer.
- **Riddle Generator**: Generate creative and challenging riddles using the trained model.
- **Interactive Interface**: A user-friendly Streamlit app for seamless interaction with the model.
- **Pre-trained Models**: Includes pre-trained models (`riddle_model.joblib` and `enhanced_riddle_model.joblib`) for immediate use.
- **Dataset Included**: Comes with a dataset of riddles (`Riddles.csv`) used for training and testing.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Kaviyapriya6/Riddle-ML-Model.git
   cd Riddle-ML-Model
   ```

2. **Set Up a Virtual Environment** (Optional but Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Install the required Python packages using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   Start the Streamlit app locally:
   ```bash
   streamlit run streamlit_run.py
   ```

   The app will launch in your default web browser.

## Usage

### Riddle Solver
- Enter a riddle in the provided input box.
- The model will analyze the riddle and display the predicted answer.

### Riddle Generator
- Use the generator feature to create new riddles.
- The model generates riddles based on patterns learned during training.

## Dataset

The dataset used for training the model is included as `Riddles.csv`. It contains a collection of riddles and their corresponding answers. You can expand or modify this dataset to improve the model's performance.

## Pre-trained Models

Two pre-trained models are included in the repository:
- `riddle_model.joblib`: The base version of the trained model.
- `enhanced_riddle_model.joblib`: An improved version with additional training and fine-tuning.

You can load these models directly for inference or further training.

## How It Works

The model uses machine learning and natural language processing techniques to understand and solve riddles. It was trained on the `Riddles.csv` dataset using features extracted from the text of the riddles. The enhanced model incorporates additional optimizations for better accuracy and creativity.

## Requirements

- Python 3.8 or higher
- Streamlit
- Scikit-learn
- Pandas
- Joblib
- Numpy

Refer to the `requirements.txt` file for the full list of dependencies.

## Contributing

Contributions are welcome! If you'd like to improve this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeatureName`).
3. Commit your changes (`git commit -m "Add some feature"`).
4. Push to the branch (`git push origin feature/YourFeatureName`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support

For questions, bug reports, or feature requests, please open an issue in the repository.
