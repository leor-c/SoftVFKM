import numpy as np
import pandas as pd

class SoftVectorFieldKMeans:
    '''
    Class for my soft vector field K Means algorithm
    '''
    def __init__(self, k, trajectoriesFile, gridRes, smoothnessWeight=0.05, certaintyWeight=0.35):
        '''
        %   Constructor - initialization and running of the algorithm.
            %   k - num of clusters
            %   trajectoriesFile - dataset filename
            %   gridRes - Grid resolution. given gridRes we get a (gridRes X gridRes) grid,
            %   smoothnessWeight - optional. default is 0.05, you can pass
            %                       [] to get default value.
            %   certaintyWeight - The amount of certainty of the model
            %                     about how well a trajectory is related
            %                     to a cluster.
            %                     typically between 0 to 1, the higher
            %                     the more confident it is. default
            %                     is 0.35. if you dont want to give a
            %                     value, enter [] as value.
        :param k:
        :param trajectoriesFile:
        :param gridRes:
        :param smoothnessWeight:
        :param certaintyWeight:
        '''
        self.numOfClusters = k
        self.vectorFields = []

        self.trajectoriesFile = trajectoriesFile
        self.readInput(trajectoriesFile)

        from math import sqrt
        LMBDA_diag = 0.5 * ((1 / sqrt(2)) + (1 / sqrt(6)))
        LMBDA_rest = 0.5 * ((1 / sqrt(2)) - (1 / sqrt(6)))
        self.Lambda = np.array([[LMBDA_diag, LMBDA_rest], [LMBDA_rest, LMBDA_diag]])

        self.gridRes = gridRes
        self.smoothnessWeight = smoothnessWeight
        self.certaintyWeight = certaintyWeight

    def readInput(self, trajectoriesFile):
        '''
        %   read the input file containing the trajectories and the
            %   grid bounds. the first line should containt the bound aka
            %   xmin xmax ymin ymax tmin tmax. and the rest is x y t of
            %   samples of trajectories where trajectories are seperated by
            %   0 0 0 line.
        :param trajectoriesFile:
        :return:
        '''
