import numpy as np
import math
import matplotlib.pyplot as plt


def get_diameter(t):
    return PITCH * (t / (2 * np.pi))


def get_y(t, d, offset=0):
    return d * math.sin(t) + offset


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
    coef = 1 / (np.sqrt(2 * np.pi * STD_DEV ** 2))
    euler = - ((distance - MEAN) ** 2) / (2 * STD_DEV ** 2)
    d = AMPLITUDE * coef * np.exp(euler)
    return d


def calc_new_center(current, target):
    weight = calc_distribution(target[0], target[1]) / (
            calc_distribution(current[0], current[1]) + calc_distribution(target[0], target[1]))
    dx = target[0] - current[0]
    dy = target[1] - current[1]
    return [current[0] + dx * weight, current[1] + dy * weight]


# SET CONSTANTS
DIAMETER_START = 200
COUNT = 12
PITCH = DIAMETER_START / COUNT
SAMPLES_PER_REV = 12
STD_DEV = 150
MEAN = 0
AMPLITUDE = 1

MIN = -350
MAX = 350
DIVISIONS = 500
SPACING = (MAX - MIN) / (DIVISIONS - 1)
X_OFFSET = np.random.randint(-150, 150)
Y_OFFSET = np.random.randint(-150, 150)

def main():


    # get points
    theta = np.arange(0., COUNT * 2 * np.pi, 2 * np.pi / 100)
    sample_theta = np.arange(0., COUNT * 2 * np.pi + 2 * np.pi / SAMPLES_PER_REV, 2 * np.pi / SAMPLES_PER_REV)
    # reverse order
    sample_theta = sample_theta[::-1]

    # Get data points to display full spiral
    diameter = [get_diameter(t) for t in theta]
    spiral_x = [get_x(t, d) for t, d in zip(theta, diameter)]
    spiral_y = [get_y(t, d) for t, d in zip(theta, diameter)]

    # Get data points for our sample
    sample_diameter = [get_diameter(t) for t in sample_theta]
    sample_x = [get_x(t, d) for t, d in zip(sample_theta, sample_diameter)]
    sample_y = [get_y(t, d) for t, d in zip(sample_theta, sample_diameter)]

    # Get value from normal distribution
    sample_d = calc_distribution(np.array(sample_x), np.array(sample_y))

    center = [0, 0]
    history_center = [center]
    history = []

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
                _x = get_x(sample_theta[sample_index], sample_diameter[sample_index], offset=center[0])
                _y = get_y(sample_theta[sample_index], sample_diameter[sample_index], offset=center[1])
                _d = calc_distribution(_x, _y)
                history.append([_x, _y, _d])

                if _d > max_d:
                    max_d = _d.copy()
                    max_x = _x.copy()
                    max_y = _y.copy()

            center = calc_new_center(center, [max_x, max_y])
            history_center.append(center)

        for cnt in history_center:
            print(cnt)

        cycles = cycles + 1
        center = calc_new_center(center, [max_x, max_y])
        dist_center = np.array([X_OFFSET, Y_OFFSET])
        calc_center = np.array(center)
        error = np.sqrt(np.sum((dist_center - calc_center) ** 2))
        print(f'error: {error}')
        print(f'cycles: {cycles}')

    print(f'moves: {len(history)}')
    print('\r')
    print(f'gaussian center: {X_OFFSET}, {Y_OFFSET}')
    print(f'remaining error: {error}')

    history_x = [i[0] for i in history]
    history_y = [i[1] for i in history]
    history_center_x = [i[0] for i in history_center]
    history_center_y = [i[1] for i in history_center]

    X, Y = prep_normal_mesh()
    D = calc_distribution(X, Y)

    # Plotting

    # Define plot frame
    fig, ax = plt.subplots()

    # Plot x, y of spiral
    ax.plot(sample_x, sample_y)
    ax.plot(history_x, history_y)
    ax.plot(history_center_x, history_center_y)

    # Plot x, y of distribution - offset for cell width
    px = X - (SPACING / 2)
    py = Y - (SPACING / 2)
    ax.pcolormesh(px, py, D, shading='auto')
    plt.xlim((MIN, MAX))
    plt.ylim((MIN, MAX))
    plt.show()

    # %%
