# Name: Ethan Lan
# Number: 000960215
# Assignment 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import StandardScaler

class KmeansClustering:
    def __init__(self, k):
        self.k = k      # K = the number of desired clusters
        self.centroids = None

    # Calculates the Euclidean distances
    @staticmethod
    def calc_euclidean_distances(datapoint, centroids):
        return np.sqrt(np.sum((centroids - datapoint) ** 2, axis=1))

    # K-means algorithm
    def kmeans(self, data, dataset_num, max_iterations=300):
        # Initialize centroids
        self.centroids = np.random.uniform(np.amin(data, axis=0), np.amax(data, axis=0), size=(self.k, data.shape[1]))

        # Variable to keep track of iterations
        iterations = 0

        # Loop for predefined max iterations
        for _ in range(max_iterations):
            iterations += 1
            c_assignments = [] # Create array to hold cluster assignments for datapoints

            # Assign each datapoint to a cluster
            for i, datapoint in enumerate(data):
                # Calculate the Euclidean distances
                distances = KmeansClustering.calc_euclidean_distances(datapoint, self.centroids)

                # Determine the centroid the datapoint is closest to and assign it to its cluster
                cluster_num = np.argmin(distances)
                c_assignments.append(cluster_num)

            c_assignments = np.array(c_assignments)

            if iterations == 1:
                orig_centroids = self.centroids

                # Print original results
                print("\nOriginal Centroids:\n", orig_centroids)
                print("\nOriginal Cluster Assignments:\n", c_assignments)

                # Plot all original data points and centroids while coloring them appropriately
                if dataset_num == 1:
                    plt.scatter(data[:, 0], data[:, 1], s=50, c=c_assignments)
                    plt.scatter(orig_centroids[:, 0], orig_centroids[:, 1], s=500, c=range(len(orig_centroids)),
                                marker='*')
                elif dataset_num == 2:
                    plt.scatter(data[:, 2], data[:, 3], s=50, c='blue')
                    plt.scatter(orig_centroids[:, 2], orig_centroids[:, 3], s=500, c=range(len(orig_centroids)),
                                marker='*')

                # Include grid, title, and then display resulting scatter plot
                plt.grid(True)
                plt.title('Original Scatter Plot of the Data')
                plt.show()

            # Create array to hold the indices of the datapoints for each cluster
            cluster_indices = []

            # For each cluster, add its associated datapoint indices
            for i in range(self.k):
                cluster_indices.append(np.argwhere(c_assignments == i).flatten())

            # Create array to hold new cluster centers (aka: new centroid positions)
            cluster_centers = []

            # Take all datapoints for each cluster and adjust the cluster center (centroid) positions accordingly
            for i, indices in enumerate(cluster_indices):
                # If cluster has no indices then don't change cluster center position
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])

                # If cluster has indices then update cluster center position by finding
                # the mean of all correlated datapoints to that centroid
                else:
                    cluster_centers.append(data[indices].mean(axis=0))

            # If the centroids are barely moving anymore then end iterations early since
            # miniscule changes won't make much of a difference at this point
            if np.max(self.centroids - np.array(cluster_centers)) < 0.000001:
                break

            # If centroids are still moving by a large margin then set new centroid positions and continue iterating
            else:
                self.centroids = np.array(cluster_centers)

        # Return finalized centroids and all datapoint cluster assignments
        return self.centroids, c_assignments, orig_centroids

    # Cleans the dataset to be used
    @staticmethod
    def clean_data(lines):
        cleaned_data = []

        for line in lines:
            # Split the line by commas
            items = line.strip().split(',')
            cleaned_row = []

            for item in items:
                # Remove everything except for decimal data points
                cleaned_data_point = re.sub(r'[^0-9.]', '', item)

                # Convert to float if the cleaned item is not null and is an actual valid number
                if cleaned_data_point and cleaned_data_point != '.':
                    cleaned_row.append(float(cleaned_data_point))

            # Add cleaned row to the cleaned data array
            cleaned_data.append(cleaned_row)

        # Return the cleaned data
        return cleaned_data

    # Main program loop that handles printing menu options and plotting results
    @staticmethod
    def program_loop():
        # Loop menu to decide which dataset to use
        while True:
            dataset = int(input(f"Select the dataset to use (Enter 1 or 2):\n"
                                f"1. kmtest\n"
                                f"2. iris\n"))

            if dataset == 1:
                data = pd.read_csv('kmtest.csv', header=None, sep='\s+').values
                break

            if dataset == 2:
                data = pd.read_csv('iris.csv', header=None, sep='\s+')
                lines = data.astype(str).values.flatten().tolist()
                data = np.array(KmeansClustering.clean_data(lines))
                break

            print("Invalid input.. Try again!")

        while True:
            input_normalize = int(input(f"Normalize the data?\n"
                                        f"Enter 1 to normalize or 2 to NOT normalize the data: "))

            # Normalize the data using z-score normalization
            if input_normalize == 1:
                zscore_scaler = StandardScaler()
                data = zscore_scaler.fit_transform(data)
                print("Data is normalized..\n")
                print(f"K-Means Clustering With Normalization:")
                break

            # Leave data as is (Data won't be normalized)
            if input_normalize == 2:
                print("Data won't be normalized..\n")
                print(f"K-Means Clustering Without Normalization:")
                break

            print("Invalid input.. Try again!")

        # Obtain the k value from the user
        k_value = int(input("Enter your desired k value (k = number of desired clusters) -- "))

        # Print the dataset
        print("Data contents:\n", data)

        # Run K-Means
        centroids, cluster_assignments, original_centroids = KmeansClustering(k=k_value).kmeans(data, dataset)

        # Print final results
        print("\nFinal Centroids:\n", centroids)
        print("\nDistances Between Original and Final Centroids:\n", KmeansClustering.calc_euclidean_distances(original_centroids, centroids)
)

        # Plot all data points and centroids while coloring them appropriately
        if dataset == 1:
            plt.scatter(data[:, 0], data[:, 1], s=50, c=cluster_assignments)
            plt.scatter(centroids[:, 0], centroids[:, 1], s=500, c=range(len(centroids)), marker='*')
        elif dataset == 2:
            plt.scatter(data[:, 2], data[:, 3], s=50, c=cluster_assignments)
            plt.scatter(centroids[:, 2], centroids[:, 3], s=500, c=range(len(centroids)), marker='*')

        # Include grid, title, and then display resulting scatter plot
        plt.grid(True)
        plt.title('Scatter Plot of the Data')
        plt.show()

        print(f"\nRun program again?")

        # Loop to set exit condition depending on users choice
        while True:
            exit_condition = int(input("Enter 0 to quit and 1 to run again: "))

            if exit_condition == 1 or exit_condition == 0:
                return exit_condition
            else:
                print("Invalid input.. Try again!")

# Loop the program until user decides to exit the program
while True:
    e = KmeansClustering.program_loop()

    if e == 0:
        print("Ending Program..")
        break

    if e == 1:
        print("Running Program Again..")