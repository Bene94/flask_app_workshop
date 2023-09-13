import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'supersecretkey'

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = None
X = None
y = None

@app.route('/', methods=['GET', 'POST'])
def upload_and_train():
    global net, X, y
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            flash("No file provided.")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        file.save(filename)

        try:
            data = np.loadtxt(filename, delimiter=',')
            X, y = data[:, :-1], data[:, -1].astype(int)
            
            input_dim = X.shape[1]
            
            if input_dim != 2:
                flash("Only datasets with 2 input features are supported for visualization.")
                return redirect(request.url)

            hidden_dim = int(request.form.get('hidden_dim'))
            learning_rate = float(request.form.get('learning_rate'))

            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)

            net = Net(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=len(np.unique(y)))
            optimizer = optim.Adam(net.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()

            loss_values = []

            for epoch in range(1000):
                optimizer.zero_grad()
                output = net(X_tensor)
                loss = criterion(output, y_tensor)
                loss.backward()
                optimizer.step()
                loss_values.append(loss.item())
                
                print(f"Epoch: {epoch}, Loss: {loss.item()}")

            return render_template('result.html', loss=loss_values[-1])     
           
        except Exception as e:
            flash(f"Error: {e}")

    return render_template('index.html')

@app.route('/boundaries')
def show_boundaries():
    if net and X is not None:
        fig, ax = plt.subplots()
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        
        Z = np.argmax(net(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)).data.numpy(), axis=1)
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
        # make the picture large 1000x1000 pixels
                
        canvas = FigureCanvas(fig)
        img_buf = BytesIO()

        canvas.print_figure(img_buf, format='png', bbox_inches='tight', pad_inches=0.1)
        
        img_str = "data:image/png;base64," + base64.b64encode(img_buf.getvalue()).decode('utf-8')
        
        print(img_str[:100])  # prints the first 100 characters


        return render_template('boundaries.html', image=img_str)
    else:
        flash("First upload a dataset.")
        return redirect(url_for('upload_and_train'))

if __name__ == '__main__':
    app.run(debug=True)
