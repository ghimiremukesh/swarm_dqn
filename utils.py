import torch
import os
import glob
from itertools import combinations_with_replacement
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# a = list(combinations_with_replacement([0, 1], 4))
# b = list(combinations_with_replacement([1, 0], 4))
# c = sorted(list(set(a+b)))

a = ["%04d" % (int(bin(i)[2:])) for i in range(16)]
b = [list(a[i]) for i in range(len(a))]
c = np.array(b, dtype=np.int32)
ACTION_MAP = {key: value for key, value in enumerate(c)}


def soft_update(target, source, tau):
    """
    Soft update method to update target network
    :param target: target network in double dqn
    :param source: Q-Network in double dqn
    :param tau: ratio of weights in the target
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0-tau) + param.data * tau)

def save_model(model, step, logs_path, types, MAX_MODEL):
    """Saves the weights of the model.

    Args:
      model (toch.nn.Module): the model to be saved.
      step (int): the number of updates so far.
      logs_path (str): the path to save the weights.
      types (str): the decorater of the file name.
      MAX_MODEL (int): the maximum number of models to be saved.
    """
    start = len(types) + 1
    os.makedirs(logs_path, exist_ok=True)
    model_list = glob.glob(os.path.join(logs_path, "*.pth"))
    if len(model_list) > MAX_MODEL - 1:
        min_step = min([int(li.split("/")[-1][start:-4]) for li in model_list])
        os.remove(os.path.join(logs_path, "{}-{}.pth".format(types, min_step)))

    logs_path = os.path.join(logs_path, "{}-{}.pth".format(types, step))
    torch.save(model.state_dict(), logs_path)
    print("  => Save {} after [{}] updates".format(logs_path, step))

def save_obj(obj, filename):
  """Saves the object into a pickle file.

  Args:
      obj (object): the object to be saved.
      filename (str): the path to save the object.
  """
  with open(filename + ".pkl", "wb") as f:
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def plot(x1_values, x2_values, time_steps):
    # Define the bounding box sizes for each region
    box_sizes = [1, 1, 1, 1]

    # Multiply proportions by the total number of swarms
    total_swarms = 1000
    x1_counts = [[int(prop * total_swarms) for prop in timestep] for timestep in x1_values]
    x2_counts = [[int(prop * total_swarms) for prop in timestep] for timestep in x2_values]

    # Create subplots for each time step
    fig, axs = plt.subplots(time_steps, 2, figsize=(12, 4 * time_steps))

    # Generate plots for each time step
    for t in range(time_steps):
        # axs[t].set_xlim(-2, 2)
        # axs[t].set_ylim(-2, 2)
        # axs[t, 0].set_xlabel('X-axis')
        # axs[t, 0].set_ylabel('Y-axis')
        axs[t, 0].set_title(f'Time Step {t}, t = {(t) * 0.05}')
        axs[t, 0].xaxis.set_tick_params(labelbottom=False)
        axs[t, 0].yaxis.set_tick_params(labelleft=False)

        # Hide X and Y axes tick marks
        axs[t, 0].set_xticks([])
        axs[t, 0].set_yticks([])

        # Plot bounding boxes for each region
        corners = [(0, 0), (0, 2 * max(box_sizes)), (2 * max(box_sizes), 2 * max(box_sizes)), (2 * max(box_sizes), 0)]
        color1 = 'red'
        color2 = 'blue'

        for i, corner in enumerate(corners):
            rectangle = Rectangle(corner, box_sizes[i], box_sizes[i], edgecolor='black', facecolor='none')
            axs[t, 0].add_patch(rectangle)

            # Plot dots representing swarms in each region

            # alpha1 = x1_counts[t][i]/1000
            # alpha2 = x2_counts[t][i]/1000
            if x1_counts[t][i] > x2_counts[t][i]:
                alpha1 = 1
                alpha2 = 0.3
            else:
                alpha2 = 1
                alpha1 = 0.3
            x1_swarm_dots = np.random.uniform(corner[0], corner[0] + box_sizes[i], x1_counts[t][i])
            y1_swarm_dots = np.random.uniform(corner[1], corner[1] + box_sizes[i], x1_counts[t][i])
            axs[t, 0].scatter(x1_swarm_dots, y1_swarm_dots, color=color1, alpha=alpha1)

            x2_swarm_dots = np.random.uniform(corner[0], corner[0] + box_sizes[i], x2_counts[t][i])
            y2_swarm_dots = np.random.uniform(corner[1], corner[1] + box_sizes[i], x2_counts[t][i])
            axs[t, 0].scatter(x2_swarm_dots, y2_swarm_dots, color=color2, alpha=alpha2)

            # Add region labels
            if i == 0 or i == 3:  # regions 1 and 3
                axs[t, 0].annotate(f"Region {i + 1}", (corner[0] + box_sizes[i] / 2, corner[1] + 1.1 * box_sizes[i]),
                                   ha='center', va='center')
            else:
                axs[t, 0].annotate(f"Region {i + 1}", (corner[0] + box_sizes[i] / 2, corner[1] - box_sizes[i] / 4),
                                   ha='center', va='center')
        # Plot bar chart comparing the number of swarms in each region
        regions = ['Region 1', 'Region 2', 'Region 3', 'Region 4']
        counts_x1 = x1_counts[t]
        counts_x2 = x2_counts[t]
        bar_width = 0.35

        x = np.arange(len(regions))
        axs[t, 1].bar(x, counts_x1, width=bar_width, label='Attackers', color='red', alpha=0.5)
        axs[t, 1].bar(x + bar_width, counts_x2, width=bar_width, label='Defenders', color='blue', alpha=0.5)

        axs[t, 1].set_xticks(x + bar_width / 2)
        axs[t, 1].set_xticklabels(regions)
        axs[t, 1].set_ylabel('Number of Swarms')
        axs[t, 1].set_title('Number of Swarms in Each Region')
        axs[t, 1].legend()
    plt.tight_layout()
    plt.show()
