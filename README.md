# Genetic Algorithm for Feature Selection

This Python code implements a genetic algorithm (GA) for feature selection. Feature selection is an important step in machine learning where irrelevant or redundant features are identified and removed from the dataset, improving model performance and reducing computational complexity.

## Usage:

1. **Initialize the GeneticAlgorithm object:** Create an instance of the `GeneticAlgorithm` class with the desired population size and number of generations.
2. **Evolve the population:** Call the `evolve` method with the feature matrix `X` and target vector `y` to run the genetic algorithm and obtain the best set of features.
3. **Obtain the best features:** The `evolve` method returns the best individual (binary string representing selected features) found by the genetic algorithm.

## Features:

- Automatically discovers the optimal subset of features from the input data.
- Uses genetic operators like selection, crossover, and mutation to evolve the population.
- Supports customization of population size, mutation rate, and other parameters.

## Requirements:

- Python 3.x
- NumPy

