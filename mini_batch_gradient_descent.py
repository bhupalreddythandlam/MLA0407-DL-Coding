x = [1, 2, 3, 4]
y = [2, 4, 6, 8]

w = 0.5
alpha = 0.1
batch_size = 2

print(f"Initial weight: {w}\n")

for i in range(0, len(x), batch_size):
    x_batch = x[i : i + batch_size]
    y_batch = y[i : i + batch_size]
    m = len(x_batch)
    batch_num = (i // batch_size) + 1
    
    print(f"--- Mini-batch {batch_num} ---")
    print(f"x_batch: {x_batch}, y_batch: {y_batch}")
    
    y_pred = [w * xi for xi in x_batch]
    print(f"Predictions: {y_pred}")
    
    gradient_sum = sum((yp - yt) * xi for yp, yt, xi in zip(y_pred, y_batch, x_batch))
    gradient = (1 / m) * gradient_sum
    print(f"Gradient: {gradient}")
    
    w = w - (alpha * gradient)
    print(f"Updated weight: {w}\n")

print(f"Final weight after one epoch: {w}")