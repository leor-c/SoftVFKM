import numpy as np
import pandas as pd
import scipy.sparse as sc
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
    def __init__(self, k, trajectoriesFile, gridRes, smoothnessWeight=0.05, certaintyWeight=0.35, verbose=1):
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
            #   verbose - the amount of info the algorithm prints. higher value = more info
        :param k:
        :param trajectoriesFile:
        :param gridRes:
        :param smoothnessWeight:
        :param certaintyWeight:
        '''
        #   set a variable for amount of info shown:
        self.verbose = verbose

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
        self.logInfo('Tessellating trajectories...', 2)
        self.tessellateTrajByGrid()

        #   compute the time of each trajectory:
        self.trajectoriesTimes = self.computeTrajectoriesTimes()

        #   compute C_tilde, b_tilde of each trajectory:
        self.logInfo('Computing matrices...', 2)
        self.calculateTrajMatrices()
        self.C_tilde, self.b_tilde, self.trajectories_n_rows = self.getValueConstrains(self.trajectories,
                                                                                     self.trajectoriesBary)
        #   save everything...
        #pd.DataFrame(self.trajectories).to_csv('SVFKM_'+self.trajectoriesFile)

        #   set the initial assignment of trajectories to clusters:
        self.initializeAssignment()

        #   compute probabilities to weights matrix:
        self.probaToWeightMatrix = self.getProbToWeightsMax()


    def logInfo(self, info, logLevel=1):
        if self.verbose >= logLevel:
            print(info)


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
                #   remove seequences of points with same coordinates:
                trajectory = rawSamples[firstRow:i, :]
                _, uniqueCoordsIdxs = np.unique(trajectory[:, 0:2], return_index=True, axis=0)
                uniqueCoordsIdxs = np.sort(uniqueCoordsIdxs)
                trajectory = trajectory[uniqueCoordsIdxs, :]

                MinNumOfPtsInCurve = 2
                if trajectory.shape[0] < MinNumOfPtsInCurve:
                    self.logInfo('Oh Oo, input file has trajectory with less than 2 points! DISCARDING')
                else:
                    self.trajectories[curTraj] = trajectory
                    self.totalTime += (trajectory[-1, 2] - trajectory[0, 2])
                    curTraj += 1
                firstRow = i+1

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

        self.logInfo('Removing ' + str(len(omit)) + ' trajectories out of bounds.')
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
        #   1. pick random trajectory:
        n_traj = len(self.trajectories)
        import random
        random.seed()
        t1 = random.randint(0, n_traj-1)

        #   2. generate the 1st VF:
        VFs = []
        VFs.append(self.findVectorFieldForTraj(t1))

        for vf_i in range(1, self.numOfClusters):
            maxErr = -1
            maxIdx = -1
            for traj_j in range(n_traj):
                err = self.assignTrajToVF(VFs, traj_j)
                minErr = np.min(err)
                if minErr > maxErr:
                    maxErr = minErr
                    maxIdx = traj_j

            VFs.append(self.findVectorFieldForTraj(maxIdx))

        #   calculate probabilities matrix (assignment):
        self.probabilitiesMatrix = self.computeProbabilities(VFs)


    def fitVectorField(self, VF_idx):
        '''
        %   generate the vector field with only the trajectories
        %   assigned to it. VF_idx is the index of the trajectories
        %   associated with VF.
        %   VF is matrix of size 2V * 2 where each row is (vx,vy)
        %   vector at a vertex.
        %   In this function I'll solve Ax=b according to the paper.
        %   A = (lambdaL * L^T*L + C~^T*C~)
        %   b = C~^T * b~
        :param VF_idx:
        :return:
        '''
        smoothness = (self.Laplacian.T @ self.Laplacian) * self.smoothnessWeight
        #   obtain the value constraint expression:
        #   multiply by weights:
        weights = (self.probaToWeightMatrix @ self.probabilitiesMatrix[:, VF_idx, np.newaxis])
        weightedTimes = (self.trajectoriesTimes.T @ self.probabilitiesMatrix[:, VF_idx, np.newaxis])
        coefficient = np.sqrt((1 - self.smoothnessWeight))
        assert weights.shape[1] == 1
        W = (coefficient * np.sqrt(weights / weightedTimes))
        #cur_C_tilde = np.multiply((coefficient * np.sqrt(weights / weightedTimes)), self.C_tilde)
        cur_C_tilde = self.C_tilde.multiply(W)
        cur_b_tilde = np.multiply(W, self.b_tilde)

        A = smoothness + (cur_C_tilde.T @ cur_C_tilde)
        b = (cur_C_tilde.T @ cur_b_tilde)

        #VF_x = np.linalg.lstsq(A, b[:, 0])[0]
        #VF_y = np.linalg.lstsq(A, b[:, 1])[0]
        from scipy.sparse.linalg import lsqr
        VF_x = np.array(lsqr(A, b[:, 0])[0])
        VF_y = np.array(lsqr(A, b[:, 1])[0])

        VF = np.hstack([VF_x.reshape((VF_x.size,1)), VF_y.reshape((VF_x.size,1))])
        return VF


    def findVectorFieldForTraj(self, trajIdx):
        '''
        %   generate the vector field with only the trajectories
        %   assigned to it. VF_idx is the index of the trajectories
        %   associated with VF.
        %   VF is matrix of size 2V * 2 where each row is (vx,vy)
        %   vector at a vertex.
        %   In this function I'll solve Ax=b according to the paper.
        %   A = (lambdaL * L^T*L + C~^T*C~)
        %   b = C~^T * b~
        :param trajIdx:
        :return:
        '''
        smoothness = (self.Laplacian.T @ self.Laplacian) * self.smoothnessWeight
        #   obtain the value constraint expression:
        cur_C_tilde = self.trajectoriesC[trajIdx].toarray()
        cur_b_tilde = self.trajectoriesb[trajIdx]

        A = smoothness + (cur_C_tilde.T @ cur_C_tilde)
        b = (cur_C_tilde.T @ cur_b_tilde)

        VF_x = np.linalg.lstsq(A, b[:, 0])[0]
        VF_y = np.linalg.lstsq(A, b[:, 1])[0]

        VF = np.hstack([np.reshape(VF_x,(A.shape[1],1)), np.reshape(VF_y,(A.shape[1],1))])
        #TODO - verify this is a good VF structure
        return VF


    def assignTrajToVF(self, VFs, trajectory_idx):
        '''
        %   calculate the error for each of the given VFs, and return
        %   the index of the best fitting VF.
        %   VFs - a cell array of vector field matrices
        %   trajectory - matrix of samples
        :param VFs: a cell array of vector field matrices
        :param trajectory_idx: index of trajectory
        :return: errors of the given trajectory on each of the VFs
        '''
        errors = np.array([self.calculateTrajectoryFit(VF, trajectory_idx) for VF in VFs])
        return errors

    def calculateTrajectoryFit(self, VF, trajectory_idx):
        '''
        %   calculate the error of the given trajectory on the given
        %   vector field.
        %   trajectory - matrix of sample points (single element of the
        %   set)
        %   VF - matrix of size V x 2 (two coordinates x,y)
        %[C,b] = obj.getValueConstrains({trajectory}, {trajectoryBary});
        :param VF:
        :param trajectory_idx:
        :return:
        '''
        C = self.trajectoriesC[trajectory_idx]
        b = self.trajectoriesb[trajectory_idx]

        '''
        %   because both C,b are multiplied by  sqrt(1-lambda) we can
        %   ignore this
        '''
        err = np.zeros((2,))
        vf_x = np.reshape(VF[:, 0],(VF.shape[0],1))
        vf_y = np.reshape(VF[:, 1],(VF.shape[0],1))
        b_x = np.reshape(b[:, 0],(b.shape[0],1))
        b_y = np.reshape(b[:, 1],(b.shape[0],1))
        CTC = (C.T @ C)
        err[0] = ((vf_x.T @ CTC) @ vf_x) - 2 * ((b_x.T @ C) @ vf_x) + (b_x.T @ b_x)
        err[1] = ((vf_y.T @ CTC) @ vf_y) - 2 * ((b_y.T @ C) @ vf_y) + (b_y.T @ b_y)
        err = np.sum(err)
        return err


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
        return np.array(trajectoriesTimes)


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
        C_tilde = []
        b_tilde = []
        n_rows = {}

        for traj_i in range(nTraj):
            traj = VFtrajectories[traj_i]
            traj_bary = VFtrajectoriesBary[traj_i]
            nSamples = traj.shape[0]
            nSegments = nSamples - 1
            CallSeg = []
            b_allSeg = {}

            startBary = traj_bary[0, :]
            for seg_i in range(nSegments):
                endBary = traj_bary[seg_i + 1, :]

                #   compute Cs_tilde:
                Cs = sc.vstack([startBary, endBary])
                assert Cs.shape == (2, self.grid.numOfV)
                '''
                %   IMPORTANT: I will not divide by T right here, I
                %   will divide only once at the end!
                '''
                omega_tag = np.sqrt(traj[seg_i + 1, 2] - traj[seg_i, 2])
                CallSeg.append(omega_tag * (self.Lambda @ Cs))
                #CallSeg.append(sc.csr_matrix.multiply(sc.csr_matrix.transpose(sc.csr_matrix.dot(sc.csr_matrix.transpose(Cs), sc.csr_matrix(self.Lambda.T))), omega_tag))

                #   compute bs_tilde:
                eps = 1e-7
                bs = np.true_divide((traj[seg_i + 1, 0:2] - traj[seg_i, 0:2]), (traj[seg_i + 1, 2] - traj[seg_i, 2] + eps))
                b_allSeg[seg_i] = omega_tag * (self.Lambda @ np.tile(bs, (2, 1)))

                startBary = endBary
            if nSegments > 1:
                C_tilde.append(sc.csr_matrix(np.vstack(CallSeg)))
                b_tilde.append(np.vstack(list(b_allSeg.values())))
            else:
                C_tilde.append(CallSeg[0])
                b_tilde.append(b_allSeg[0])
            n_rows[traj_i] = C_tilde[traj_i].shape[0]

        '''
        %   need to complete calc by multiply with sqrt((1-lambda) / T):
            %   Note: I use weighted T (for each cluster) so this will be
            %   added later on, ALSO, I will add the (1-lambda) later
        '''
        if len(C_tilde) > 1:
            C_tilde = sc.vstack(C_tilde)
            b_tilde = np.vstack(b_tilde)
        else:
            C_tilde = C_tilde[0]
            b_tilde = b_tilde[0]

        return C_tilde, b_tilde, n_rows


    def computeProbabilities(self, VFs):
        '''
        %   calculate the errors for each trajectory on each VF, and
        %   use a softmax function to turn them into probabilities:
        :param VFs:
        :return:
        '''
        n_traj = len(self.trajectories)
        probabilities = np.zeros((n_traj, len(VFs)))

        for traj_j in range(n_traj):
            traj_errors = self.assignTrajToVF(VFs, traj_j)
            errsSum = np.sum(traj_errors)
            avgErrSum = errsSum / self.numOfClusters
            eps = 1e-8
            relations = avgErrSum / (traj_errors + eps)
            #   move to interval [0,10]:
            bounds = np.array([0, max(relations)])
            moveToInterval = lambda x: self.certaintyWeight * 10 * (x - bounds[0]) / (bounds[1] - bounds[0])
            relations = moveToInterval(relations)
            sumRel = np.sum(np.exp(relations))
            probabilities[traj_j,:] = np.exp(relations) / sumRel

        return probabilities


    def getProbToWeightsMax(self):
        '''
        %   compute the matrix that help calculate the weights for the
        %   C_tilde and b_tilde matrices at each iteration.
        :return:
        '''
        n_rows = self.C_tilde.shape[0]
        n_cols = len(self.trajectories)
        I = np.arange(n_rows)
        J = np.zeros((n_rows,))
        Vals = np.zeros((n_rows,))
        cur_row = 0

        for i in range(len(self.trajectories_n_rows)):
            cur_last = cur_row + self.trajectories_n_rows[i]
            Vals[cur_row: cur_last] = 1
            J[cur_row: cur_last] = i
            cur_row = cur_last

        probaToWeightsMat = sc.csr_matrix((Vals, (I, J)), (n_rows, n_cols))
        return probaToWeightsMat


    def convergeClusters(self, convergenceThreshold=1e-8, maxIterations=50, verbose=1):
        '''
        %   this is the main loop of the algorithm!
        %   perform steps until reached convergence, then the vector
        %   fields and assignment function will be vaild! [=
        :param convergenceThreshold:
        :param maxIterations:
        :param verbose: amount of info to print. higher value = more info
        :return:
        '''
        #   initialize vector fields to zeros if not defined yet:
        if len(self.vectorFields) != self.numOfClusters:
            self.vectorFields = [np.zeros((self.grid.numOfV, 2)) for i in range(self.numOfClusters)]

        stepSize = np.inf
        iter = 1
        while stepSize > convergenceThreshold and iter < maxIterations:
            #   perform a step:
            stepSize = 0
            newVFs = []
            for i in range(self.numOfClusters):
                newVFs.append(self.fitVectorField(i))
                #   calculate step size:
                stepSize = stepSize + np.linalg.norm(newVFs[i] - self.vectorFields[i], ord='fro')

            self.vectorFields = newVFs

            #   recalculate the assignment function
            newProbabilities = self.computeProbabilities(self.vectorFields)

            self.logInfo(str(iter) + ') Current Step Size: ' + str(stepSize))
            iter = iter + 1

            #   update the assignment function:
            self.probabilitiesMatrix = newProbabilities








