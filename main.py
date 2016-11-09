# coding: utf-8

import collections
import itertools
import math
import operator
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D


class FeatureVector:
    """
    Вектор признаков
    """

    def __init__(self, features):
        """
        :param features: Признаки (количественные) объекта
        :type features: list

        """

        self.features = features
        self.center = None

    def __eq__(self, other):
        return self.features == other.features and self.center == other.center


# Содержит расстояние между двумя векторами
FeatureVectorPair = collections.namedtuple('FeatureVectorPair', 'first second metrics')


# Содержит расстояние от заданного вектора до вершины
CenterMetricsPair = collections.namedtuple('CenterMetricsPair', 'center_vector metrics')


class KMeans:
    """
    Классификатор К средних
    """

    # Максимальное количество итераций при обучении
    MAX_ITER = 150

    @classmethod
    def euclede_metrics(cls, first_vector, second_vector):
        """
        Метрика Евклида

        :param first_vector: Первый вектор признаков
        :param second_vector: Второй вектор признаков
        :return: float

        """

        return math.sqrt(KMeans.euclede_metrics_sqr(first_vector, second_vector))

    @classmethod
    def euclede_metrics_sqr(cls, first_vector, second_vector):
        """
        Метрика Евклида в квадрате

        :param first_vector: Первый вектор признаков
        :param second_vector: Второй вектор признаков
        :return: float

        """

        return sum([(first - second) ** 2 for first, second in zip(first_vector.features, second_vector.features)])

    @classmethod
    def manhattan_metrics(cls, first_vector, second_vector):
        """
        Манхэттенская метрика

        :param first_vector: Первый вектор признаков
        :param second_vector: Второй вектор признаков
        :return: float

        """

        return sum([abs(first - second) for first, second in zip(first_vector.features, second_vector.features)])

    @classmethod
    def chebyshev_metrics(cls, first_vector, second_vector):
        """
        Метрика Чебышева

        :param first_vector: Первый вектор признаков
        :param second_vector: Второй вектор признаков
        :return: float

        """

        return max([abs(first - second) for first, second in zip(first_vector.features, second_vector.features)])

    def __init__(self, n_clusters, train):
        """
        :param n_clusters_list: Список возможного количества кластеров
        :param train: Обучающая выборка
        """

        self.n_clusters = n_clusters

        # Обучающая выборка
        self.train = train

        # Центры кластеров
        self.centers = None
        # Кластеры
        self.clusters = None

    def fit(self, metrics=None):
        """
        Обучение классификатора

        :param metrics: Используемая метрика
        :return:

        """

        if metrics is None:
            metrics = KMeans.euclede_metrics

        # Массив расстояний между всеми возможными точками, отсортированный по убыванию величины расстояния
        centers = sorted([FeatureVectorPair(first, second, metrics(first, second))
                          for first, second in itertools.product(self.train, repeat=2)],
                         key=operator.itemgetter(2))[::-1]
        centers = [center.first for center in centers]

        # Для сравнения с прощлым шагом
        tmp_clusters = None

        iteration_counter = 0

        while True:
            iteration_counter += 1

            # Выбор K центров
            current_centers = centers[:self.n_clusters]

            # Распределение по кластерам
            current_clusters = list()

            for feature_vector in self.train:
                min_metrics_center = sorted([CenterMetricsPair(center_vector, metrics(feature_vector, center_vector))
                                             for idx, center_vector in enumerate(current_centers)],
                                            key=operator.itemgetter(1))[0]

                feature_vector.center = min_metrics_center.center_vector

                current_clusters.append(feature_vector)

            if iteration_counter == self.MAX_ITER:
                return None, None

            if tmp_clusters is None or tmp_clusters != current_clusters:
                tmp_clusters = current_clusters
            else:
                break

        self.centers = current_centers
        self.clusters = current_clusters

    def draw_plot(self):
        """
        Рисование графика по трем первым признакам выборки

        Если модель не обучена, в консоль выводится сообщение и ничего не рисуется.
        :return:

        """

        if self.centers is None or self.clusters is None:
            print('Модель не обучена. Перед рисованием графика необходимо вызвать метод .fit()')
            return

        figure = plt.figure()

        # Цветовая палитра
        colors = cm.rainbow(np.linspace(0, 1, len(self.centers)))

        # 111 - сетка 1х1, первый subplot
        cluster_subplot = figure.add_subplot(111, projection='3d')

        for center, color in zip(self.centers, colors):
            x_clusters = [x.features[0] for x in self.clusters if x.center.features == center.features]
            y_clusters = [x.features[1] for x in self.clusters if x.center.features == center.features]
            z_clusters = [x.features[2] for x in self.clusters if x.center.features == center.features]

            cluster_subplot.scatter(x_clusters, y_clusters, z_clusters, color=color)

        plt.show()


def read_features_from_file(file_path):
    """
    Чтение выборки из файла.
    Данные должны быть разделены одним пробелом, содержать только числа и не содержать пропущенных значений

    :param file_path: Полный путь к файлу для чтения
    :return: list of FeatureVector
    :raise ValueError:

    """

    if not os.path.exists(file_path):
        raise ValueError('Невозможно открыть файл {}'.format(file_path))

    result = list()

    for line in open(file_path, mode='r', encoding='u8').readlines():
        result.append(FeatureVector([int(x) for x in line.split(' ')]))

    return result


def main():
    file_path = 'KMeansDataSet.txt'

    kmeans = KMeans(5, read_features_from_file(file_path))

    kmeans.fit()

    kmeans.draw_plot()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        exit(1)