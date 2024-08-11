# Q-Learning with Taxi-v3

This repository contains code for training and evaluating a Q-learning agent on the Taxi-v3 environment from OpenAI's Gym. The project is divided into two main notebooks:

1. **Q-learning.ipynb**: This notebook trains a Q-learning agent using various hyperparameters and saves the Q-table to disk.
2. **Testing.ipynb**: This notebook evaluates saved Q-tables, visualizes the performance, and displays the best-performing Q-table.

## Requirements

- Python 3.x
- Google Colab
- Gym
- Numpy
- Matplotlib
- Seaborn
- Pandas

You can install the required libraries using pip:

```bash
pip install gym numpy matplotlib seaborn pandas
```

## Setup

1. **Mount Google Drive**

   Ensure you have Google Colab set up and your Google Drive is mounted:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

   The notebooks assume that the relevant files are stored in `/content/drive/MyDrive/RML/KA/`.

## Notebooks

### 1. Q-learning.ipynb

This notebook performs the following tasks:

- Creates and initializes the Taxi-v3 environment.
- Initializes the Q-table and allows users to input hyperparameters.
- Runs the training loop with epsilon-greedy exploration.
- Saves the Q-table after training.

#### Hyperparameters

- `epsilon`: Exploration rate.
- `alpha`: Learning rate.
- `gamma`: Discount factor.
- `max_epsilon`: Maximum exploration rate.
- `min_epsilon`: Minimum exploration rate.
- `decay_rate`: Rate of decay for epsilon.

### 2. Testing.ipynb

This notebook evaluates the Q-tables saved from `Q-learning.ipynb`:

- Loads Q-tables from the specified directory.
- Tests each Q-table over multiple episodes to compute average moves, penalties, and testing time.
- Visualizes the results, including:
  - Dependence of average penalties and moves on epsilon and epochs.
  - Heatmaps for average testing and training times.
  - Visualization of the best-performing Q-table in the Taxi-v3 environment.

## Usage

1. **Train Q-Table**

   Run `Q-learning.ipynb` to train the Q-learning agent. It will prompt you to enter values for the hyperparameters. The final Q-table will be saved to the specified directory.

2. **Evaluate Q-Tables**

   Run `Testing.ipynb` to evaluate the saved Q-tables. This notebook will provide visual insights into the performance of different Q-tables and identify the best-performing one.

## Visualization

The `Testing.ipynb` notebook includes several plots:

- **Average Penalties vs. Epsilon for Different Epochs**
- **Average Moves vs. Epsilon for Different Epochs**
- **Dependence of Testing Time on Moves for Different Epochs**
- **Heatmaps for Average Testing and Training Times**

It also includes a section to visualize the Taxi-v3 environment using the best Q-table.

## Notes

- Ensure that you have appropriate permissions and paths set for accessing files in Google Drive.
- Adjust the file paths in the notebooks if you store your files in a different directory.

## Acknowledgements

- OpenAI Gym for the Taxi-v3 environment.
- Google Colab for providing a convenient environment for running the notebooks.
