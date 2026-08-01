import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [1, 2, 1],
    [2, 1, 2],
    [3, 2, 1],
    [4, 3, 2],
    [5, 4, 3]
], dtype=float)

y = np.array([10, 12, 16, 22, 28], dtype=float)

def compute_epoch_loss(X, y, w, b):
    predictions = np.dot(X, w) + b
    return 0.5 * np.mean((predictions - y) ** 2)

def train_sgd(X, y, alpha, epochs=10):
    w = np.zeros(X.shape[1])
    b = 0.0
    loss_history = [compute_epoch_loss(X, y, w, b)]
    
    for epoch in range(epochs):
        for i in range(len(y)):
            x_i = X[i]
            y_i = y[i]
            pred = np.dot(x_i, w) + b
            error = pred - y_i
            dw = error * x_i
            db = error
            w -= alpha * dw
            b -= alpha * db
        loss_history.append(compute_epoch_loss(X, y, w, b))
        
    return w, b, loss_history

alpha = 0.01
epochs = 10
w_final, b_final, loss_001 = train_sgd(X, y, alpha=alpha, epochs=epochs)

print(f"Final Weights (alpha=0.01): w1={w_final[0]:.6f}, w2={w_final[1]:.6f}, w3={w_final[2]:.6f}, b={b_final:.6f}")
print(f"Final Loss: {loss_001[-1]:.6f}")

x_new = np.array([6, 5, 4])
predicted_sales = np.dot(x_new, w_final) + b_final
print(f"Predicted sales for x1=6, x2=5, x3=4: {predicted_sales:.6f}\n")

_, _, loss_0001 = train_sgd(X, y, alpha=0.001, epochs=epochs)
_, _, loss_01 = train_sgd(X, y, alpha=0.1, epochs=epochs)

plt.figure(figsize=(8, 5))
plt.plot(range(epochs + 1), loss_0001, marker='o', label=r'$\alpha = 0.001$')
plt.plot(range(epochs + 1), loss_001, marker='s', label=r'$\alpha = 0.01$')
plt.plot(range(epochs + 1), loss_01, marker='^', label=r'$\alpha = 0.1$')
plt.title("SGD Epoch Loss vs. Epochs for Different Learning Rates")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()