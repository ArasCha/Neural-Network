import numpy as np
import matplotlib.pyplot as plt
import h5py


def display_dataset(X: list, Y: list):
    plt.scatter(X[:,0], X[:,1], c=Y)

def display_error_graph(errors: list):
	plt.plot(errors)
	plt.show()

def display_decision_frontier(W, b):

    x1 = np.linspace(0, 5, 100)
    x2 = ( -W[0] * x1 - b ) / W[1]

    plt.plot(x1, x2, c="green", lw=3)


def load_data():
    train_dataset = h5py.File('datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) # your train set features
    y_train = np.array(train_dataset["Y_train"][:]) # your train set labels

    test_dataset = h5py.File('datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) # your train set features
    y_test = np.array(test_dataset["Y_test"][:]) # your train set labels
    
    return X_train, y_train, X_test, y_test


def show_images(X: np.ndarray, Y: np.ndarray, nb=10):
	"""
	X: set of images or single image
	Y: values for each image or value of image
	nb: Number of images to show
	"""
	
	plt.figure(figsize=(16, 8))

	def plot_image(img, value, i=1):
		plt.subplot(4, 5, i)
		plt.imshow(img, cmap='gray')
		plt.title(value)
		plt.tight_layout()

	if X.ndim == 3:
		for i in range(1, nb+1):
			plot_image(X[i], Y[i], i)

	elif X.ndim == 2:
		plot_image(X, Y)
	
	else: Exception("Enter an image or a set of images (dimension wasn't 2 or 3)")

	plt.show()


def normalize_images(X_images: np.ndarray) -> np.ndarray:
	"""
	X_images: set of images or single image
	Returns the image(s) flatten and normalized (from range 255 to 1)
	"""
	
	if X_images.ndim == 3: # it's a list of images
		X_flattened = X_images.reshape(X_images.shape[0], X_images.shape[1] * X_images.shape[2]) # flattening the 2nd and 3nd dimensions together
	
	elif X_images.ndim == 2: # it's a single images
		X_flattened = X_images.flatten()
		X_flattened = X_flattened.reshape(1, len(X_flattened))

	else:
		raise Exception("Enter an image or a set of images (dimension wasn't 2 or 3)")
	
	return X_flattened.astype(np.float64) / 255
