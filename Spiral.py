import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('GTK3Agg')


def get_diameter(t):
    return PITCH * (t / (2 * np.pi))


def get_y(t, d, offset=0):
    return (d * math.sin(t) + offset)


def get_x(t, d, offset=0):
    return (d * math.cos(t)) + offset


# Helper methods for plotting
def plot_distribution(x, y, dist):
    dx = x - (SPACING / 2)
    dy = y - (SPACING / 2)
    fig, ax = plt.subplots()
    ax.pcolormesh(dx, dy, dist)
    plt.xlim((MIN, MAX))
    plt.ylim((MIN, MAX))


def prep_normal_mesh():
    x, y = np.meshgrid(np.linspace(MIN, MAX, DIVISIONS), np.linspace(MIN, MAX, DIVISIONS))
    return x, y


def calc_distribution(x, y):
    dx = x - X_OFFSET
    dy = y - Y_OFFSET
    distance = np.sqrt(dx ** 2 + dy ** 2)
    coef1 = 1 / (np.sqrt(2 * np.pi * STD_DEV ** 2))
    euler = - ((distance - MEAN) ** 2) / (2 * STD_DEV ** 2)
    d = AMPLITUDE * coef1 * np.exp(euler)
    return d


def calc_new_center(current_x, current_y, new_x, new_y):
    weight = calc_distribution(new_x, new_y) / (
                calc_distribution(current_x, current_y) + calc_distribution(new_x, new_y))
    dx = new_x - current_x
    dy = new_y - current_y
    return current_x + dx * weight, current_y + dy * weight


# SET CONSTANTS
DIAMETER_START = 200
COUNT = 10
PITCH = DIAMETER_START / COUNT
SAMPLES_PER_REV = 10
STD_DEV = 50
MEAN = 0
AMPLITUDE = 1

MIN = -500
MAX = 500
DIVISIONS = 500
SPACING = (MAX - MIN) / (DIVISIONS - 1)
X_OFFSET = np.random.randint(-150, 150)
Y_OFFSET = np.random.randint(-150, 150)

# get points
theta = np.arange(0., COUNT * 2 * np.pi, 2 * np.pi / 100)
sample_theta = np.arange(0., COUNT * 2 * np.pi + 2 * np.pi / SAMPLES_PER_REV, 2 * np.pi / SAMPLES_PER_REV)
# reverse order
sample_theta = sample_theta[::-1]

# Get data points to display full spiral
diameter = [get_diameter(t) for t in theta]
x = [get_x(t, d) for t, d in zip(theta, diameter)]
y = [get_y(t, d) for t, d in zip(theta, diameter)]

# Get data points for our sample
sample_diameter = [get_diameter(t) for t in sample_theta]
sample_x = [get_x(t, d) for t, d in zip(sample_theta, sample_diameter)]
sample_y = [get_y(t, d) for t, d in zip(sample_theta, sample_diameter)]

# Get value from normal distribution
sample_d = calc_distribution(np.array(sample_x), np.array(sample_y))

center_x = 0
center_y = 0
history_x = []
history_y = []
history_d = []
history_diameter = []
max_d = 0
max_x = 0
max_y = 0
error = 100
cycles = 0

while error > 5 and cycles < 5:

    for rev in range(0, COUNT):
        start = rev * SAMPLES_PER_REV
        stop = (rev + 1) * SAMPLES_PER_REV

        for index, value in enumerate(sample_theta[start: stop]):
            sample_index = index + rev * SAMPLES_PER_REV

            history_diameter.append(get_diameter(sample_theta[sample_index]))
            history_x.append(get_x(sample_theta[sample_index], sample_diameter[sample_index], offset=center_x))
            history_y.append(get_y(sample_theta[sample_index], sample_diameter[sample_index], offset=center_y))
            history_d.append(calc_distribution(history_x[sample_index], history_y[sample_index]))

            if history_d[-1] > max_d:
                max_d = history_d[-1]
                max_x = history_x[-1]
                max_y = history_y[-1]
                # print(max_x,max_y, max_d)

        # print('\r')
        # print(f'Max: {max_x}, {max_y}')
        center_x, center_y = calc_new_center(center_x, center_y, max_x, max_y)
        max_x, max_y = center_x.copy(), center_y.copy()
        # print(f'Center: {center_x}, {center_y}')

    cycles = cycles + 1
    dist_center = np.array([X_OFFSET, Y_OFFSET])
    calc_center = np.array([center_x, center_y])
    error = np.sqrt(np.sum((dist_center - calc_center) ** 2))
    print(f'error: {error}')
    print(cycles)

print('\r')
print(f'gaussian center: {X_OFFSET}, {Y_OFFSET}')
print(f'remaining error: {error}')
print(len(history_d))

X, Y = prep_normal_mesh()
D = calc_distribution(X, Y)

# Plotting

# Define plot frame
fig, ax = plt.subplots()

# Plot x, y of spiral
ax.plot(sample_x, sample_y)
ax.plot(history_x, history_y)

# Plot x, y of distribution - offset for cell width
px = X - (SPACING / 2)
py = Y - (SPACING / 2)
ax.pcolormesh(px, py, D)
plt.xlim((MIN, MAX))
plt.ylim((MIN, MAX))
plt.show()

# %%
