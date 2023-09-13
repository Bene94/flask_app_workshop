\# Neural Network Decision Boundary Visualization

\## Setup

1. Install the required libraries:
\```
pip install -r requirements.txt
\```

2. Run the Flask application:
\```
python app.py
\```

\## Accessing the Web Interface

Once the Flask application is running, it will host a local web server. To access the interface:

1. Open your preferred web browser.
2. In the address bar, type `http://127.0.0.1:5000/` and press Enter.
3. You should now see the web interface for uploading datasets and setting hyperparameters.

\## Usage

1. Navigate to the main page of the web interface.
2. Upload your dataset. The dataset should be in CSV format, with the last column being the target labels and the first two columns as the two input features.
3. Set your desired hyperparameters:
   - Hidden Dimension
   - Learning Rate
   - Number of Epochs
4. Click on the "Upload and Train" button.
5. Once trained, you will be shown the decision boundary for the dataset based on the trained neural network.

\## Notes

- This tool is meant for visualization purposes and works best with 2D datasets.
- Larger datasets or complex decision boundaries might take longer to process.
