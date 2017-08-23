import numpy as np
import scipy.sparse as sc

class VFKMGrid:
    '''
    Class for the grid of the algorithm
    '''
    def __init__(self, X_Interval, Y_Interval, timeInterval, numOfVrtx_x, numOfVrtx_y):
        '''
        Initialize the object (C'tor)
        '''
        self.timeInterval = timeInterval
        self.X_Interval = X_Interval
        self.Y_Interval = Y_Interval
        self.numOfVrtx_x = numOfVrtx_x
        self.numOfVrtx_y = numOfVrtx_y
        self.numOfV = self.numOfVrtx_x * self.numOfVrtx_y

        '''
        make the grid:
        '''
        self.grid_x = np.linspace(self.X_Interval[0], self.X_Interval[1], self.numOfVrtx_x)
        self.grid_y = np.linspace(self.Y_Interval[0], self.Y_Interval[1], self.numOfVrtx_y)

        self.delta_x = np.abs(self.grid_x[1] - self.grid_x[0])
        self.delta_y = np.abs(self.grid_y[1] - self.grid_y[0])

        '''
        %   make a matrix whose a_ij elements are the location of the
        %   i,j vertex
        '''
        x_i, y_i = np.meshgrid(np.arange(0, self.numOfVrtx_x), np.arange(0, self.numOfVrtx_y))
        x_i = np.reshape(x_i, (x_i.size,), order='C')
        y_i = np.reshape(y_i, (y_i.size,), order='C')
        self.vertexMatCoords = np.array([x_i, y_i]).T
        self.vertexLocations = np.array([self.getVertexLocation(x+1, y+1) for (x,y) in zip(x_i,y_i)])
        self.vertexIndexes = np.reshape(np.arange(0, self.numOfVrtx_x * self.numOfVrtx_y), (self.numOfVrtx_x,self.numOfVrtx_y)).T


    def getFaceVertices(self, location):
        '''
        %   Get the vertex indices of the face containing the point
        %   given by world coordinated (lon/lat)
        %   *** the  vertices are orderer (indexed) like in matlab matrix:
        %   columns top to bottom and from the left most column to the
        %   right most one.
        %   indices of faces are COUNTER-CLOCKWISE

        %   (adding 1 because of index shift in matlab)
        :param location:
        :return:
        '''
        i = int(np.floor((self.Y_Interval[1] - location[1]) / self.delta_y))
        j = int(np.floor((location[0] - self.X_Interval[0]) / self.delta_x))
        '''
        %   (i,j) is the top left vertex of the rectangle containing
        %   the point
        '''
        row_idx = j * self.numOfVrtx_y + i
        topLeft = self.vertexLocations[row_idx, :]

        '''
        %   determine if in bottom left triangle or upper right one:
        '''
        x_in_rect = location[0] - topLeft[0]
        y_in_rect = topLeft[1] - location[1]
        if (x_in_rect == 0) or ((y_in_rect / x_in_rect) > (self.delta_y / self.delta_x)):
            #   bottom left triangle
            vertices = np.array([self.vertexIndexes[i, j],
            self.vertexIndexes[i + 1, j],
            self.vertexIndexes[i + 1, j + 1]])
        else:
            # upper right triangle
            vertices = np.array([self.vertexIndexes[i, j],
            self.vertexIndexes[i + 1, j + 1],
            self.vertexIndexes[i, j + 1]])
        return vertices


    def isInGrid(self, location):
        '''

        :param location:
        :return:
        '''
        if self.X_Interval[0] < location[0] < self.X_Interval[1] and self.Y_Interval[0] < location[1] < self.Y_Interval[1]:
            res = True
        else:
            res = False
        return res


    def getVertexLocation(self, i, j=None):
        '''
        %   Get the vertex location (lon\lat etc.) by the indices of
        %   the grid of this vertex (like matrix indices of an
        %   element).

        %   if given in vertex index, define in grid coordinates (i,j):
            if nargin == 2
                [i, j] = obj.vertexMatCoords(i,:);
            end
        :param i:
        :param j:
        :return:
        '''
        if j is None:
            i,j = self.vertexMatCoords[i,:]

        vertex_x_coor = lambda a,b: (self.X_Interval[0] + (b-1)*self.delta_x)
        vertex_y_coor = lambda a,b: (self.Y_Interval[1] - (a-1)*self.delta_y)
        vLoc = [vertex_x_coor(i,j), vertex_y_coor(i,j)]
        return vLoc

    def getBarycentricCoords(self, location, face=None):
        '''
        %   Returns a vector with the barycentric coordinates of the
        %   given location. the vector is of length resX * resY with
        %   zeros in vertices not in the relevant face.
        :param location:
        :param face:
        :return:
        '''
        if face is None:
            face = self.getFaceVertices(location)

        vLocs = np.array([self.vertexLocations[face[0],:], self.vertexLocations[face[1],:], self.vertexLocations[face[2],:]])
        A = np.array([vLocs[:, 0].T, vLocs[:,1].T, np.array([1, 1, 1])])
        b = np.array([[location[0]], [location[1]], [1]])
        lambda_const = np.linalg.solve(A,b)
        print('###')
        print(np.reshape(lambda_const,(lambda_const.size,)))
        barycentricVec = sc.bsr_matrix((np.reshape(lambda_const,(lambda_const.size,)), (face, np.array([0, 0, 0]))))
        return barycentricVec


    def isInSameFace(self, p1, p2, f1=None, f2=None):
        '''
        %   checks if both points are in the same face. points are in
        %   world coordinates (lon/lat etc.)
        :param p1:
        :param p2:
        :param f1:
        :param f2:
        :return:
        '''
        if f1 is None or f2 is None:
            b1 = np.nonzero(self.getBarycentricCoords(p1))
            b2 = np.nonzero(self.getBarycentricCoords(p2))
        else:
            #   to save computation time...
            b1 = np.nonzero(self.getBarycentricCoords(p1, f1))
            b2 = np.nonzero(self.getBarycentricCoords(p2, f2))

        u = np.unique(np.array([b1, b2]))
        return u.size <= 3


    def tesselateTrajectory(self, trajectory):
        '''
        %   Tesselate a given trajectory by the grid edges and return a
        %   new trajectory (in a matrix)
        :param trajectory:
        :return:
        '''
        n_points = trajectory.shape
        tesselatedTrajectory = {}
        faces = {}
        barycentric = {}

        for segment_i in np.arange(0, n_points-1):
            #   tellesate each raw segment:
            s_begin = trajectory[segment_i,:]
            s_end = trajectory[segment_i + 1,:]

            #   check if need to tesselate:
            if not self.isInGrid(s_begin) or not self.isInGrid(s_end):
                continue

            f_begin = self.getFaceVertices(s_begin)
            f_begin_locs = np.array([self.vertexLocations[f_begin[0],:], self.vertexLocations[f_begin[1],:], self.vertexLocations[f_begin[2],:]])
            f_end = self.getFaceVertices(s_end)

            bary_begin = self.getBarycentricCoords(s_begin, f_begin)
            bary_end = self.getBarycentricCoords(s_end, f_end)

            '''
            %   calculate the number of intersections with
            %   horizontal/vertical edges:
            '''
            coords = self.vertexMatCoords[f_begin[0],:]
            i_begin = coords[0]
            j_begin = coords[1]
            coords = self.vertexMatCoords[f_end[0],:]
            i_end = coords[0]
            j_end = coords[1]
            numOfVertical = np.abs(j_end - j_begin)
            numOfHorizontal = np.abs(i_end - i_begin)

            '''
            %   calculate the number of intersections diagonally
            %   using my formula:
            '''
            coords = self.vertexMatCoords[f_begin[1],:]
            i_begin = coords[0]
            j_begin = coords[1]
            coords = self.vertexMatCoords[f_end[1],:]
            i_end = coords[0]
            j_end = coords[1]
            numOfDiag = np.abs((j_end - j_begin) - (i_end - i_begin))

            '''
            %   now lets get the intersection points, but add a
            %   third value to each one, the value of the parameter
            %   s of the line gamma(s) in parametric form. this
            %   will help to sort afterwards.
            '''
            numNewPts = numOfVertical + numOfHorizontal + numOfDiag
            if numNewPts == 0:
                tesselatedTrajectory[segment_i] = np.array([np.hstack((s_begin, 0)), np.hstack((s_end, 1))])
                faces[segment_i] = np.array([f_begin, f_end])
                barycentric[segment_i] = np.array([bary_begin, bary_end])
                continue

            newPoints = np.zeros((numNewPts, 4))
            newPointsFaces = np.zeros((numNewPts, 3))
            newPointsBary = np.zeros((numNewPts, self.numOfV))

            s_vec = s_end - s_begin

            '''
               add vertical intersection points:
            '''
            if numOfVertical > 0:
                x_int = np.array([min(s_begin[0], s_end[0]), max(s_begin[0], s_end[0])])
                horiz_1st = np.ceil(np.abs(x_int[0] - self.X_Interval[0]) / self.delta_x) + 1
                horiz_last = np.floor(np.abs(x_int[1] - self.X_Interval[0]) / self.delta_x) + 1
                known_X = lambda i: (self.X_Interval[0] + (i - 1) * self.delta_x)
                '''
                % ensure correct calculation:
                '''
                assert (horiz_last - horiz_1st + 1 == numOfVertical);
                for horiz_idx in np.arange(horiz_1st, horiz_last+1):
                    '''
                    %   get the intersection point with this edge!
                    '''
                    s = (known_X(horiz_idx) - s_begin[0]) / s_vec[0]
                    new_point = s_begin + s * s_vec
                    new_pt_idx = horiz_idx - horiz_1st
                    newPoints[new_pt_idx,:] = np.hstack((new_point, s))
                    newPointsFaces[new_pt_idx,:] = self.getFaceVertices(new_point)
                    newPointsBary[new_pt_idx,:] = self.getBarycentricCoords(new_point, newPointsFaces[new_pt_idx,:])

            '''
            %   add horizontal intersection points:
            '''
            if numOfHorizontal > 0:

                horizBounds = np.array([np.mod(f_begin[0], self.numOfVrtx_y), np.mod(f_end[0], self.numOfVrtx_y)])[::-1].sort()
                horizBounds = np.arange(horizBounds[0], horizBounds[1], -1)

                assert (horizBounds.size == numOfHorizontal);
                for vert_idx in horizBounds:
                    '''
                    %   get the intersection point with this edge!
                    '''
                    s = (self.vertexLocations[vert_idx, 1] - s_begin[1]) / s_vec[1]
                    new_point = s_begin + s * s_vec
                    new_pt_idx = numOfVertical + horizBounds[0] - vert_idx
                    newPoints[new_pt_idx,:] = np.hstack((new_point, s))
                    newPointsFaces[new_pt_idx,:] = self.getFaceVertices(new_point)
                    newPointsBary[new_pt_idx,:] = self.getBarycentricCoords(new_point, newPointsFaces[new_pt_idx,:])

            '''
            %   add diagonal intersection points:
            '''
            if numOfDiag > 0:
                '''
                %   I denote the line of the diagonal as:
                %   l(t) = P0 + u*t
                %   where u is the direction vector of the diagonal.
                '''
                u = np.array([self.delta_x, -self.delta_y])
                alpha = np.arccos(np.dot(u, np.array([1, 0])) / (np.linalg.norm(u)))
                rotated_s_vec = np.dot(np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]]), s_vec[0:2].T)
                if rotated_s_vec[1] >= 0:
                    '''
                    %   direction is up right, start with vertex 3 of
                    %   the face of the start point
                    '''
                    p0 = f_begin_locs[2,:]
                    p0 = np.tile(p0, (numOfDiag, 1))
                    p0[:, 1] = np.arange(p0[0, 1],(p0[0, 1] + (numOfDiag - 1) * self.delta_y)+1, self.delta_y).T
                else:
                    '''
                    %   direction is down left, start with vertex 2 of
                    %   the face of the start point
                    '''
                    p0 = f_begin_locs[1,:]
                    p0 = np.tile(p0, (numOfDiag, 1))
                    p0[:, 1] = np.arange(p0[0, 1], (p0[0, 1] - (numOfDiag - 1) * self.delta_y)-1, -self.delta_y).T


                for cur_diag in np.arange(0,numOfDiag):
                    '''
                    %   calculate the current coefficient according to
                    %   my formula, then add the point
                    '''
                    s_coeff = lambda p: (u[1] * (s_begin[0] - p[0]) - u[0] * (s_begin[1] - p[1])) / (u[0] * s_vec[1] - u[1] * s_vec[0])
                    s = s_coeff(p0[cur_diag, :])
                    new_point = s_begin + s * s_vec
                    new_pt_idx = numOfVertical + numOfHorizontal + cur_diag
                    newPoints[new_pt_idx,:] = np.hstack((new_point, s))
                    newPointsFaces[new_pt_idx,:] = self.getFaceVertices(new_point)
                    newPointsBary[new_pt_idx,:] = self.getBarycentricCoords(new_point, newPointsFaces[new_pt_idx,:])

            '''
            %   sort by 4th row, and them omit the 4th row:
            '''
            I = np.argsort(newPoints[:, 4])
            newPoints = newPoints[I,:]
            _, ia = np.unique(newPoints, return_index=True, axis=0)
            ia = np.sort(ia)
            newPoints = newPoints[ia, :]

            newPointsFaces = newPointsFaces[I,:]
            newPointsBary = newPointsBary[I,:]
            newPointsFaces = newPointsFaces[ia,:]
            newPointsBary = newPointsBary[ia,:]

            tesselatedTrajectory[segment_i] = np.vstack((np.concatenate((s_begin, [0]),axis=0),
                                                        newPoints,
                                                        np.concatenate((s_end, [1]),axis=0)))
            faces[segment_i] = np.vstack((f_begin, newPointsFaces, f_end))
            barycentric[segment_i] = np.vstack((bary_begin, newPointsBary, bary_end))

        '''
        %   remove empty cells:
        '''
        tesselatedTrajectory = np.vstack([v for k, v in tesselatedTrajectory.items()])
        _, ia = np.unique(tesselatedTrajectory, axis=0, return_index=True)
        ia = np.sort(ia)
        tesselatedTrajectory = tesselatedTrajectory[ia, 0:3]

        faces = np.vstack([v for k, v in faces.items()])
        faces = faces[ia, :]

        barycentric = np.vstack([v for k, v in barycentric.items()])
        barycentric = barycentric[ia, :]

        return tesselatedTrajectory, faces, barycentric


    def getLaplacian(self):
        '''
        %   calculate and return the Laplace-Beltrami operator using
        %   the cotangent-weights
        %   I will use the form L = A^-1*(D-W), but ommit A^-1 because
        %   the areas doesnt really matter here...
        :return:
        '''
        w_vert = self.delta_x / self.delta_y
        w_hor = self.delta_y / self.delta_x
        w_4nbrs = 2 * w_vert + 2 * w_hor

        '''
        %   create D with all weights and then substract from
        %   boundaries.
        '''
        D = sc.eye(self.numOfV).multiply(w_4nbrs)

        '''
        %   substract w_hor from sides:
        '''
        sides = np.hstack([np.arange(0,self.numOfVrtx_y), np.arange((self.numOfV - self.numOfVrtx_y), self.numOfV)])
        D[np.ix_(sides, sides)] = D[np.ix_(sides, sides)] - sc.eye(sides.size).multiply(w_hor)

        '''
        %   substracr w_vert from top,bottom:
        '''
        vsides = np.hstack([np.arange(0, (self.numOfV - self.numOfVrtx_y+1), self.numOfVrtx_y),
                            np.arange(self.numOfVrtx_y - 1, self.numOfV, self.numOfVrtx_y)])
        D[np.ix_(vsides, vsides)] = D[np.ix_(vsides, vsides)] -  sc.eye(vsides.size).multiply(w_vert)

        '''
        %   generate the W matrix:
        '''
        numOfElemInW = 4 * self.numOfV - 2 * self.numOfVrtx_x - 2 * self.numOfVrtx_y
        I = np.zeros(numOfElemInW)
        J = np.zeros(numOfElemInW)
        Vals = np.zeros(numOfElemInW)

        cur_elem = 0
        for i in range(self.numOfV):
            '''
            %   construct each line of the matrix:
            %   check for upper neighbor:
            '''
            if i > 0 and np.mod(i, self.numOfVrtx_y) != 0:
                I[cur_elem] = i
                J[cur_elem] = i - 1
                Vals[cur_elem] = w_vert
                cur_elem = cur_elem + 1
            '''
            %   check for lower neighbor:
            '''
            if i < self.numOfV and np.mod(i, self.numOfVrtx_y) != self.numOfVrtx_y-1:
                I[cur_elem] = i
                J[cur_elem] = i + 1
                Vals[cur_elem] = w_vert
                cur_elem = cur_elem + 1
            '''
            %   check for left neighbor:
            '''
            if i >= self.numOfVrtx_y:
                I[cur_elem] = i
                J[cur_elem] = i - self.numOfVrtx_y
                Vals[cur_elem] = w_hor
                cur_elem = cur_elem + 1
            '''
            %   check for right neighbor:
            '''
            if i < self.numOfV - self.numOfVrtx_y:
                I[cur_elem] = i
                J[cur_elem] = i + self.numOfVrtx_y
                Vals[cur_elem] = w_hor
                cur_elem = cur_elem + 1

        W = sc.bsr_matrix((Vals, (I, J)), shape=(self.numOfV, self.numOfV))

        L = (D - W)

        return L









