import matplotlib.pyplot as plt

# Example data for learning curves
epochs = [1, 2, 3, 4, 5]
train_accuracy = [0.6, 0.7, 0.8, 0.85, 0.9]
val_accuracy = [0.5, 0.65, 0.75, 0.78, 0.8]

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracy, label='Training Accuracy', marker='o')
plt.plot(epochs, val_accuracy, label='Validation Accuracy', marker='o')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()
