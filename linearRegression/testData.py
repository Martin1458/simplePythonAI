import numpy as np

# Generate an array of 5 lists, each containing 2 random integers between 0 and 100 (inclusive)
input_list = np.random.randint(0, 101, (50, 2))
#print(input_list)

# Create a new array containing the sum of each list in input_list
sum_input_list = np.sum(input_list, axis=1)

# Create a list of lists, where each sublist contains a single element from sum_input_list
output_list = np.array([[sum_input_list[i]] for i in range(sum_input_list.shape[0])])
#print(output_list)