import matplotlib.pyplot as plt
import pandas as pd
import re

# Read the loss results from the file
file_path = '/home/robina/jli/DL/DL2025_final_project/models/Paint-CUT/loss_results.txt'

# Initialize lists to store the data
epochs = []
G_adv_losses = []
G_losses = []
D_Y_losses = []
NCE_losses = []
NCE_Y_losses = []

# Read the file and extract the relevant data
with open(file_path, 'r') as file:
    for line in file:
        match = re.search(r'\[Epoch (\d+)\]\[Iter \d+\] defaultdict\(.*?, \{\'G_adv\': (.*?), \'D_Y\': (.*?), \'G\': (.*?), \'NCE\': (.*?), \'NCE_Y\': (.*?)\}', line)
        if match:
            epochs.append(int(match.group(1)))
            G_adv_losses.append(float(match.group(2)))
            D_Y_losses.append(float(match.group(3)))
            G_losses.append(float(match.group(4)))
            NCE_losses.append(float(match.group(5)))
            NCE_Y_losses.append(float(match.group(6)))

# Create a DataFrame for easier analysis
data = pd.DataFrame({
    'Epoch': epochs,
    'G_adv': G_adv_losses,
    'G': G_losses,
    'D_Y': D_Y_losses,
    'NCE': NCE_losses,
    'NCE_Y': NCE_Y_losses
})

# Group by epoch and calculate the average for each metric
average_data = data.groupby('Epoch').mean().reset_index()

# Plotting learning curves for averaged values of G_adv, G, D_Y, NCE, and NCE_Y
plt.figure(figsize=(12, 10))

# Plot G_adv
plt.subplot(5, 1, 1)  # 5 rows, 1 column, 1st subplot
plt.plot(average_data['Epoch'], average_data['G_adv'], label='G_adv Loss', color='blue')
plt.title('Averaged Learning Curve for G_adv')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()

# Plot G
plt.subplot(5, 1, 2)  # 5 rows, 1 column, 2nd subplot
plt.plot(average_data['Epoch'], average_data['G'], label='G Loss', color='orange')
plt.title('Averaged Learning Curve for G')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()

# Plot D_Y
plt.subplot(5, 1, 3)  # 5 rows, 1 column, 3rd subplot
plt.plot(average_data['Epoch'], average_data['D_Y'], label='D_Y Loss', color='green')
plt.title('Averaged Learning Curve for D_Y')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()

# Plot NCE
plt.subplot(5, 1, 4)  # 5 rows, 1 column, 4th subplot
plt.plot(average_data['Epoch'], average_data['NCE'], label='NCE Loss', color='red')
plt.title('Averaged Learning Curve for NCE')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()

# Plot NCE_Y
plt.subplot(5, 1, 5)  # 5 rows, 1 column, 5th subplot
plt.plot(average_data['Epoch'], average_data['NCE_Y'], label='NCE_Y Loss', color='purple')
plt.title('Averaged Learning Curve for NCE_Y')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('averaged_learning_curves.png')  # Save the figure
plt.show()  # Display the figure
