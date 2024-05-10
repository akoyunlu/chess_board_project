"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

This solution will run but the dimensionality reduction and
the classifier are not doing anything useful, so it will
produce a very poor result.

version: v1.0
"""
from typing import List

import numpy as np
import scipy.linalg


N_DIMENSIONS = 10

KNN = 8


def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
    """Classify a set of feature vectors using a training set.

    This dummy implementation simply returns the empty square label ('.')
    for every input feature vector in the test data.

    Note, this produces a surprisingly high score because most squares are empty.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
        """

    #Identify the features and select the desired features from the train and test data
    features = np.arange(0, train.shape[1])
    train = train[:, features]
    test = test[:, features]

    #Identify the unique labels and show them in the train labels 
    labels,train_label_int = np.unique(train_labels, return_inverse =True )

    #Measure the euclidean distance between the train and test data
    distances = np.sqrt(np.sum((train[:, np.newaxis] - test) ** 2, axis=2))

    #Identify the K nearest neighbours get the labels of the neighbours
    indices = np.argpartition(distances, kth=KNN-1, axis=0)[:KNN]
    neighbour_labels = train_label_int[indices]


    # Set the weights of the neighbours based on the inverse of their distance 
    weights = 1 / distances[indices, np.arange(distances.shape[1])]

    #Identify the label for each test data considering the weighted majority of the neighbours
    weighted_labels = np.array([np.argmax(np.bincount(i, weights=weights[:, j]))
                                     for j, i in enumerate(neighbour_labels.T)])
    
    updated_labels= labels[weighted_labels]


    return updated_labels.tolist()
  

# The functions below must all be provided in your solution. Think of them
# as an API that it used by the train.py and evaluate.py programs.
# If you don't provide them, then the train.py and evaluate.py programs will not run.
#
# The contents of these functions are up to you but their signatures (i.e., their names,
# list of parameters and return types) must not be changed. The trivial implementations
# below are provided as examples and will produce a result, but the score will be low.


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    The feature vectors are stored in the rows of 2-D array data, (i.e., a data matrix).
    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    # """


    #Initialises the array to store the reduced data
    pca_data = np.empty((0, data.shape[1]))
    covx = np.cov(data, rowvar=0)
    N = covx.shape[0]
    
    v = np.fliplr(scipy.linalg.eigh(covx, eigvals=(N - 10, N - 1))[1])

    #if the lenght of the model is 1 then the eigenvector is stored in the v.tolist() 
    #if the lenght is bigger than 1, it stays on the model["eigenvector"]
    model["eigenvector"] = v.tolist() if len(model) == 1 else model["eigenvector"]
    
    #reduces the data by dot producting the data with the eigenvector
    pca_data= np.dot((data - np.mean(data)), np.array(model["eigenvector"]))
    
    #I used this to test the shape of the data for both the PCA and the KNN
    # print (pca_data.shape)
    return pca_data
    
   



def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:


    """Process the labeled training data and return model parameters stored in a dictionary.

    Note, the contents of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """
    # The design of this is entirely up to you.
    # Note, if you are using an instance based approach, e.g. a nearest neighbour,
    # then the model will need to store the dimensionally-reduced training data and labels.
   
    model = {}
    model["labels_train"] = labels_train.tolist()
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train"] = fvectors_train_reduced.tolist()
  

    return model
   

def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """
    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        fvectors[i, :] = image.reshape(1, n_features)

    return fvectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Get some data out of the model. It's up to you what you've stored in here
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    # Call the classify function.
    labels = classify(fvectors_train, labels_train, fvectors_test)
    return labels





def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    In the dummy code below, we just re-use the simple classify_squares function,
    i.e. we ignore the ordering.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """



    return classify_squares(fvectors_test, model)


