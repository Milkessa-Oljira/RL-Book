import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# -------------------------------
# 1. Define the Target Function
# -------------------------------
def f_target(x):
    """A more complex target function."""
    return 0.5 * np.sin(2*x) + 0.2 * np.cos(3*x) + 0.1*x**2 - 0.3*x

# -------------------------------
# 2. Build a Neural Network Model
# -------------------------------
def build_model(xs, ys):
    """Builds and trains an MLP regressor (Corrected and Improved)."""
    model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='tanh', solver='adam',
                         max_iter=2000, random_state=42, early_stopping=True,
                         validation_fraction=0.1, n_iter_no_change=10,
                         learning_rate_init=0.001)  # Added learning rate

    model.fit(xs.reshape(-1, 1), ys)
    return model

# -------------------------------
# 3. Confidence Radius for a Sample
# -------------------------------
def check_interval(model, x_center, r, epsilon, domain, num_points=100):
    x_start = max(domain[0], x_center - r)
    x_end = min(domain[1], x_center + r)
    xs = np.linspace(x_start, x_end, num_points)
    errors = np.abs(model.predict(xs.reshape(-1, 1)) - f_target(xs))
    return np.all(errors <= epsilon)

def compute_confidence_radius(model, x_center, epsilon, domain, tol=1e-4):
    low = 0.0
    high = min(x_center - domain[0], domain[1] - x_center)
    while high - low > tol:
        mid = (low + high) / 2
        if check_interval(model, x_center, mid, epsilon, domain):
            low = mid
        else:
            high = mid
    return low

# -------------------------------
# 4. Merge Certified Intervals
# -------------------------------
def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0].copy()]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1][1] = max(last_end, end)
        else:
            merged.append([start, end])
    return merged

def global_confidence_region(model, xs, epsilon, domain):
    intervals = []
    for x in xs:
        r = compute_confidence_radius(model, x, epsilon, domain)
        interval = [max(domain[0], x - r), min(domain[1], x + r)]
        intervals.append(interval)
    merged = merge_intervals(intervals)
    total_length = sum(end - start for start, end in merged)
    return merged, total_length

# -------------------------------
# 5. Main Loop: Deterministic Exploration
# -------------------------------
domain = [-2, 2]
epsilon = 0.2

xs = np.linspace(domain[0], domain[1], 5)
ys = [f_target(x) for x in xs]
model = build_model(xs, ys)

max_iter = 30

for i in range(max_iter):
    merged_intervals, total_length = global_confidence_region(model, xs, epsilon, domain)
    print(f"Iteration {i}: Global confidence region total length = {total_length:.4f}")

    if total_length >= (domain[1] - domain[0]) - 0.01:
        break

    gaps = []
    if merged_intervals and merged_intervals[0][0] > domain[0]:
        gaps.append([domain[0], merged_intervals[0][0]])
    for j in range(len(merged_intervals) - 1):
        gap_start = merged_intervals[j][1]
        gap_end = merged_intervals[j + 1][0]
        if gap_end - gap_start > 1e-4:
            gaps.append([gap_start, gap_end])
    if merged_intervals and merged_intervals[-1][1] < domain[1]:
        gaps.append([merged_intervals[-1][1], domain[1]])


    if gaps:
        gap_lengths = [gap[1] - gap[0] for gap in gaps]
        largest_gap = gaps[np.argmax(gap_lengths)]
        new_sample = (largest_gap[0] + largest_gap[1]) / 2

        if new_sample not in xs:
            xs = np.append(xs, new_sample)
            xs = np.sort(xs)
            ys = [f_target(x) for x in xs]
            model = build_model(xs, ys)

# -------------------------------
# 6. Visualization of the Results
# -------------------------------
x_plot = np.linspace(domain[0], domain[1], 500)
y_true = f_target(x_plot)
y_model = model.predict(x_plot.reshape(-1, 1))

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_true, label='True function', color='blue')
plt.plot(x_plot, y_model, label='NN Model', linestyle='--', color='red')
plt.scatter(xs, ys, color='black', zorder=5, label='Sample points')

for x in xs:
    r = compute_confidence_radius(model, x, epsilon, domain)
    x_start = max(domain[0], x - r)
    x_end = min(domain[1], x + r)
    plt.hlines(y=f_target(x), xmin=x_start, xmax=x_end, colors='green', linewidth=4, label='Certified interval')

handles, labels = plt.gca().get_legend_handles_labels()
unique = {}
for h, l in zip(handles, labels):
    if l not in unique:
        unique[l] = h
plt.legend(unique.values(), unique.keys())


plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Learning a Complex Function with Deterministic Confidence')
plt.show()