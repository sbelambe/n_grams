import matplotlib.pyplot as plt

epochs = list(range(1, 21))

# RNN
val_accuracy = [
    0.1431, 0.1942, 0.3394, 0.3025, 0.3744, 0.4014, 0.4467, 0.4031, 
    0.4478, 0.4819, 0.4858, 0.4819, 0.4608, 0.4542, 0.4181, 0.4917, 
    0.4964, 0.4653, 0.4944, 0.4794
]

# LSTM 
val_accuracy = [
    0.4942, 0.7353, 0.7414, 0.7586, 0.7733, 0.7772, 0.7978, 0.7897, 
    0.7742, 0.7672, 0.7853, 0.7878, 0.7758, 0.7864, 0.7886, 0.7839, 
    0.7989, 0.7700, 0.7900, 0.7797
]

plt.figure(figsize=(10, 6))
plt.plot(epochs, val_accuracy, marker='o', linestyle='-', color='b', label='Validation Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy over Epochs')

# Show the grid for better readability
plt.grid(True)

# Show the plot
plt.legend()
plt.show()
