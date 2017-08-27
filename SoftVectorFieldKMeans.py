import numpy as np
import pandas as pd
import scipy as sc
import VFKMGrid

class SoftVectorFieldKMeans:
    '''
    Class for my soft vector field K Means algorithm
    %   A class for the Soft Vector Field K-Means algorithm.
    %   The difference is that each trajectory is associated with each
    %   vector field with probability t_i,j and the clusters are built
    %   using those probabilities as weights.

    %   How to use this class:
    %   ~~~~~~~~~~~~~~~~~~~~~
    %   1. call constructor (detailed in C'tor definition)
    %   2. call convergeClusters method
    %   3. call plotResults method
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
        self.vectorFields = {}

        self.gridBounds = []
        self.totalTime = 0
        self.trajectories = {}
        self.trajectoriesFaces = {}
        self.trajectoriesBary = {}
        self.trajectoriesC = {}
        self.trajectoriesb = {}

        self.trajectoriesFile = trajectoriesFile
        self.readInput(trajectoriesFile)

        from math import sqrt
        LMBDA_diag = 0.5 * ((1 / sqrt(2)) + (1 / sqrt(6)))
        LMBDA_rest = 0.5 * ((1 / sqrt(2)) - (1 / sqrt(6)))
        self.Lambda = np.array([[LMBDA_diag, LMBDA_rest], [LMBDA_rest, LMBDA_diag]])

        self.gridRes = gridRes
        self.smoothnessWeight = smoothnessWeight
        self.certaintyWeight = certaintyWeight

        #   create the grid:
        self.grid = VFKMGrid.VFKMGrid([self.gridBounds[0], self.gridBounds[1]],
                                      [self.gridBounds[2], self.gridBounds[3]],
                                      [self.gridBounds[4], self.gridBounds[5]],
                                      gridRes, gridRes)
        #%   crop to fit grid:
        self.cropToFitGrid()

        #   obtain the Laplace-Beltrami operator:
        self.Laplacian = self.grid.getLaplacian()

        #   tessellate trajectories by grid:
        self.tessellateTrajByGrid()

        #   compute the time of each trajectory:
        self.trajectoriesTimes = self.computeTrajectoriesTimes()

        #   compute C_tilde, b_tilde of each trajectory:
        self.calculateTrajMatrices()


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
        rawData = pd.read_csv(trajectoriesFile, sep=' ')
        self.gridBounds = [float(v) for v in rawData.columns.values]
        rawSamples = rawData.iloc[:, 0:3].as_matrix()
        #print(self.gridBounds)
        #print(rawSamples)
        #   also ontain the total timespan T:
        self.totalTime = 0
        #   now we need to extract the trajectories:
        self.trajectories = {}
        self.trajectoriesFaces = {}
        self.trajectoriesBary = {}
        self.trajectoriesC = {}
        self.trajectoriesb = {}
        curTraj = 0
        firstRow = 0

        for i in range(rawSamples.shape[0]):
            if np.linalg.norm(rawSamples[i,:]) == 0:
                '''
                CHECK that the trajectory has at least 2 points:
                '''
                MinNumOfPtsInCurve = 2
                if i - firstRow < MinNumOfPtsInCurve:
                    print('Oh Oo, input file has trajectory with less than 2 points! DISCARDING')
                else:
                    self.trajectories[curTraj] = rawSamples[firstRow:i, :]
                    self.totalTime += (rawSamples[i-1, 2] - rawSamples[firstRow, 2])
                firstRow = i+1
                curTraj += 1
        #print(self.trajectories.items())


    def cropToFitGrid(self):
        '''
        %   crop the trajectories to fit the grid. practically, for
        %   each trajectory, retain only portions within the grid.
        :return:
        '''
        n_traj = len(self.trajectories)
        omit = []

        for k in range(n_traj):
            t = self.trajectories[k]
            for i in range(t.shape[0]):
                if not self.grid.isInGrid(t[i, :]):
                    omit.append(k)
                    break

        print('Removing ' + str(len(omit)) + ' trajectories out of bounds.')
        for idx in omit:
            del self.trajectories[idx]


    def tessellateTrajByGrid(self):
        '''
        %   iterate over the trajectories and tessellate each of them by
        %   the grid (insert new points). Assuming each trajectory
        %   contains more than 1 point.
        %   also count total number of samples:
        :return:
        '''

        for i in range(len(self.trajectories)):
            # %   tesselate each single trajectory:
            self.trajectories[i], self.trajectoriesFaces[i], self.trajectoriesBary[i] = self.grid.tesselateTrajectory(self.trajectories[i])


    def initializeAssignment(self):
        '''
        %   initialize the assignment function that associates each
        %   trajectory with a cluster. should be as diverse as
        %   possible.
        :return:
        '''


    def computeTrajectoriesTimes(self):
        '''
        %   return a matrix of size |Num of Trajectories| where in each
        %   cell is the time of the trajectory:
        :return:
        '''
        trajectoriesTimes = []
        for i in range(len(self.trajectories)):
            t = self.trajectories[i]
            trajectoriesTimes.append(t[-1,2]-t[0,2])
        return trajectoriesTimes


    def calculateTrajMatrices(self):
        '''
        %   calc all C_tilde, b_tilde for all trajectories:
        :return:
        '''
        for i in range(len(self.trajectories)):
            #   tesselate each single trajectory
            C, b, n_rows = self.getValueConstrains([self.trajectories[i]], [self.trajectoriesBary[i]])
            self.trajectoriesC[i] = C
            self.trajectoriesb[i] = b


    def getValueConstrains(self, VFtrajectories, VFtrajectoriesBary):
        '''
        %   iterate over all segments and construct the matrices. use
        %   the given trajectories set (subset of the global one).
        :return:
        :param VFtrajectories:
        :param VFtrajectoriesBary:
        '''
        nTraj = len(VFtrajectories)
        C_tilde = {}
        b_tilde = {}
        n_rows = {}

        for traj_i in range(nTraj):
            traj = VFtrajectories[traj_i]
            traj_bary = VFtrajectoriesBary[traj_i]
            nSamples = traj.shape[0]
            nSegments = nSamples - 1
            CallSeg = {}
            b_allSeg = {}

            startBary = traj_bary[0, :]
            for seg_i in range(nSegments):
                endBary = traj_bary[seg_i + 1, :]

                #   compute Cs_tilde:
                Cs = np.array([startBary, endBary])
                '''
                %   IMPORTANT: I will not divide by T right here, I
                %   will divide only once at the end!
                '''
                omega_tag = np.sqrt(traj[seg_i + 1, 2] - traj[seg_i, 2])
                CallSeg[seg_i] = omega_tag * np.dot(self.Lambda, Cs)

                #   compute bs_tilde:
                bs = (traj[seg_i + 1, 0:2] - traj[seg_i, 0:2]) / (traj[seg_i + 1, 2] - traj[seg_i, 2])
                b_allSeg[seg_i] = omega_tag * np.dot(self.Lambda, np.tile(bs, (2, 1)))

                startBary = endBary

            C_tilde[traj_i] = np.vstack([CallSeg[i] for i in range(len(CallSeg))])
            b_tilde[traj_i] = np.vstack([b_allSeg[i] for i in range(len(b_allSeg))])
            n_rows[traj_i] = C_tilde[traj_i].shape[0]

        '''
        %   need to complete calc by multiply with sqrt((1-lambda) / T):
            %   Note: I use weighted T (for each cluster) so this will be
            %   added later on, ALSO, I will add the (1-lambda) later
        '''
        C_tilde = np.vstack([C_tilde[i] for i in range(len(C_tilde))])
        b_tilde = np.vstack([b_tilde[i] for i in range(len(b_tilde))])
        return C_tilde, b_tilde, n_rows






