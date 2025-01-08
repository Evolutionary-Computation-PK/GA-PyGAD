import os

from numpy import ndarray
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime


class GraphCreator:
    def __init__(self, generated_data_folder: str = "generated_data"):
        self.generated_data_folder = generated_data_folder
        if self.generated_data_folder and not os.path.exists(self.generated_data_folder):
            os.makedirs(self.generated_data_folder)

    def create_graphs(self, fitness_values: ndarray, search_minimum: bool):
        generations = np.arange(1, fitness_values.shape[0] + 1)
        self.create_fitness_values_graph(generations, fitness_values, search_minimum)
        self.create_standard_deviation_graph(generations, fitness_values, search_minimum)
        self.create_mean_graph(generations, fitness_values, search_minimum)

    def create_fitness_values_graph(self, generations: ndarray, fitness_values: ndarray, search_minimum: bool) -> None:
        best_solutions = np.max(fitness_values, axis=1)
        if search_minimum:
            best_solutions = 1 / best_solutions
        self.save_data(generations, best_solutions, "funkcja_celu_w_kolejnej_iteracji.txt")
        self.plot_data(generations, np.array([best_solutions]).reshape(-1, 1),
                       "Wykres zależności wartości funkcji celu od kolejnej iteracji",
                       "generacja",
                       "wartość funkcji celu",
                       "wykres_funkcji_celu_w_kolejnej_iteracji.png")

    def create_standard_deviation_graph(self, generations: ndarray, fitness_values: ndarray, search_minimum: bool) -> None:
        if search_minimum:
            fitness_values = 1 / fitness_values
        std_fitness_values = np.std(fitness_values, axis=1)
        self.save_data(generations, std_fitness_values, "funkcja_odchylenia_standardowego.txt")
        self.plot_data(generations, std_fitness_values,
                       "Wykres odchylenia standardowego w poszczególnych epokach",
                       "generacja",
                       "odchylenie standardowe",
                       "wykres_odchylenia.png")

    def create_mean_graph(self, generations: ndarray, fitness_values: ndarray, search_minimum: bool) -> None:
        if search_minimum:
            fitness_values = 1 / fitness_values
        means = np.mean(fitness_values, axis=1)
        self.save_data(generations, means, "średnie.txt")
        self.plot_data(generations,  means,
                       "Wykres średniej w poszczególnych epokach",
                       "generacja",
                       "średnia",
                       "wykres_średniej.png")

    def save_data(self, x: ndarray, y: ndarray, filename: str) -> None:
        data = np.column_stack((x, y))
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_name = f'{current_time}_{filename}'
        np.savetxt(self.generated_data_folder+"/"+file_name, data, fmt='%d', header='x y', comments='')

    def plot_data(self, x: ndarray, y: ndarray, title: str, xlabel: str, ylabel: str, filename: str) -> None:
        plt.figure()
        plt.plot(x, y, label=ylabel)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        plt.savefig(self.generated_data_folder+"/"+current_time+"_"+filename)
