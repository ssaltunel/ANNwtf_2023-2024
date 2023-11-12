from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load the digits dataset
digits = load_digits()

#dataset description
print(digits['DESCR'])

# Extract the data into (input, target) tuples
data = list(zip(digits.data, digits.target))

#checking
fig, axes = plt.subplots(1, 10, figsize=(20, 10))
for ax, (image, label) in zip(axes, data[:10]):
    ax.imshow(image.reshape(8, 8), cmap='gray')
    ax.set_title(f'Target: {label}')

data_rescaled = data * 255
data_rescaled = data_rescaled(np.uin)
plt.show()


reshaped_data = [(image.reshape(64), label) for image, label in data]
