import numpy as np

# Dataset
X = np.array([[1],
              [2],
              [3],
              [4],
              [5]], dtype=float)

Y = np.array([[2],
              [4],
              [6],
              [8],
              [10]], dtype=float)

# Initialize weight and bias
w = np.random.randn()
b = np.random.randn()

learning_rate = 0.01
epochs = 5

for epoch in range(epochs):

    epoch_loss = 0

    for j in range(X.shape[0]):

        # -------------------------
        # Select one sample
        # -------------------------
        x = X[j]
        y = Y[j]

        # -------------------------
        # Forward Propagation
        # -------------------------
        y_pred = w * x + b

        # -------------------------
        # Loss (MSE)
        # -------------------------
        loss = (y - y_pred) ** 2
        epoch_loss += loss

        # -------------------------
        # Backpropagation
        # -------------------------
        dw = -2 * x * (y - y_pred)
        db = -2 * (y - y_pred)

        # -------------------------
        # Gradient Descent Update
        # -------------------------
        w = w - learning_rate * dw
        b = b - learning_rate * db

    avg_loss = epoch_loss / X.shape[0]

    print(f"Epoch {epoch+1}")
    print(f"Average Loss = {avg_loss[0]:.4f}")
    print(f"Weight = {w:.4f}")
    print(f"Bias = {b:.4f}")
    print("-"*40)

print("\nFinal Weight:", w)
print("Final Bias:", b)