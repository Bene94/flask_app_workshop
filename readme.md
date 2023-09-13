# Neural Network Decision Boundary Visualization

## Setup

1. Install the required libraries:
```
pip install -r requirements.txt
```

2. Run the Flask application:
```
python app.py
```

## Usage

1. Navigate to the main page and upload your dataset. The dataset should be in CSV format, with the last column being the target labels and the first two columns as the two input features.

2. Set your desired hyperparameters:
   - Hidden Dimension
   - Learning Rate
   - Number of Epochs

3. Click on the "Upload and Train" button.

4. Once trained, you will be shown the decision boundary for the dataset based on the trained neural network.

## Notes

- This tool is meant for visualization purposes and works best with 2D datasets.
- Larger datasets or complex decision boundaries might take longer to process.
