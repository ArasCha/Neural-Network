from sklearn.datasets import make_blobs



def display_dataset(X: list, Y: list):
    import matplotlib.pyplot as plt
    plt.scatter(X[:,0], X[:,1], c=Y)
    plt.show()


if __name__ == "__main__":

    X, Y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
    """
    X: features. n_features = 2 so X has 2 columns
    Y: 1 if the plant is toxic, 0 if not. Feature we try to predict
    """

    Y = Y.reshape((len(Y), 1)) # Making sure that the feature to predict is a vector (1 dimension)

    # display_dataset(X, Y)

    # see https://www.youtube.com/watch?v=5TpBe7KTAHE