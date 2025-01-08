import numpy as np


class BinaryUtils:
    @staticmethod
    def get_binary_length(start_interval: float, end_interval: float, precision: int) -> int:
        """
        Calculates the binary length required to represent a number in a given range with specified precision.

        :param start_interval: Start of the interval.
        :param end_interval: End of the interval.
        :param precision: Number of decimal places.
        :return: Rounded up binary length.
        """
        number_of_intervals = (end_interval - start_interval) * pow(10, precision)
        binary_length = np.log2(number_of_intervals)
        rounded_up_binary_length = np.ceil(binary_length)
        return int(rounded_up_binary_length)

    @staticmethod
    def decode_number(binary_array: np.ndarray) -> int:
        """
        Decodes a binary array to a decimal number.

        :param binary_array: Binary array.
        :return: Decoded decimal number.
        """
        return np.sum(binary_array * (2 ** np.arange(binary_array.size)[::-1]))

    @staticmethod
    def decode_binary_representation(start_interval: float, end_interval: float, number_of_genes: int, binary_repr: np.ndarray) -> float:
        """
        Decodes a single array of 0 and 1, from binary to decimal.

        :param start_interval: Start of the interval for binary_array decimal value.
        :param end_interval: End of the interval for binary_array decimal value.
        :param number_of_genes: Number of genes in the binary array.
        :param binary_repr: Binary representation of a chromosome.
        :return: Decimal value of the chromosome.
        """
        return start_interval + (BinaryUtils.decode_number(binary_repr) * (end_interval - start_interval)) / (
                2 ** number_of_genes - 1)

    @staticmethod
    def decode_individual(individual: np.ndarray, start_interval, end_interval, number_of_variables):
        """
        Decodes single array of 0 and 1 (which represents multiple values) to array of decimal values.

        :param individual: Individual with possible multiple values to decode.
        :param start_interval: Start of the interval for variables decimal value.
        :param end_interval: End of the interval for variables decimal value.
        :param number_of_variables: Number of values in the binary array.
        """
        binary_variables_values = np.split(individual, number_of_variables)
        return np.array([BinaryUtils.decode_binary_representation(start_interval, end_interval, len(binary_variable), binary_variable)
                         for binary_variable in binary_variables_values])
