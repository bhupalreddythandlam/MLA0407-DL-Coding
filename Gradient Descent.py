import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [1, 1],
    [2, 1],
    [1, 2],
    [2, 2],
    [3, 2]
], dtype=float)

y = np.array([6, 8, 8, 10, 12], dtype=float)
m = len(y)

def compute_loss(X, y, w, b):
    predictions = np.dot(X, w) + b
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

def compute_gradients(X, y, w, b):
    predictions = np.dot(X, w) + b
    errors = predictions - y
    dw = (1 / m) * np.dot(X.T, errors)
    db = (1 / m) * np.sum(errors)
    return dw, db

def gradient_descent(X, y, alpha, num_iterations=10):
    w = np.zeros(X.shape[1])
    b = 0.0
    loss_history = [compute_loss(X, y, w, b)]
    
    for i in range(1, num_iterations + 1):
        dw, db = compute_gradients(X, y, w, b)
        w -= alpha * dw
        b -= alpha * db
        loss = compute_loss(X, y, w, b)
        loss_history.append(loss)
        
    return w, b, loss_history

alpha = 0.01
iterations = 10
w_final, b_final, loss_001 = gradient_descent(X, y, alpha=alpha, num_iterations=iterations)

print(f"Final Weights (alpha=0.01 after 10 iterations): w1={w_final[0]:.6f}, w2={w_final[1]:.6f}, b={b_final:.6f}")
print(f"Final Loss: {loss_001[-1]:.6f}")

x_new = np.array([4, 3])
predicted_price = np.dot(x_new, w_final) + b_final
print(f"Predicted price for x1=4, x2=3: {predicted_price:.6f}\n")

_, _, loss_0001 = gradient_descent(X, y, alpha=0.001, num_iterations=iterations)
_, _, loss_01 = gradient_descent(X, y, alpha=0.1, num_iterations=iterations)

plt.figure(figsize=(8, 5))
plt.plot(range(iterations + 1), loss_0001, marker='o', label=r'$\alpha = 0.001$')
plt.plot(range(iterations + 1), loss_001, marker='s', label=r'$\alpha = 0.01$')
plt.plot(range(iterations + 1), loss_01, marker='^', label=r'$\alpha = 0.1$')
plt.title("Loss vs. Iterations for Different Learning Rates")
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error (Loss J)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()