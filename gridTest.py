import VFKMGrid
import numpy as np

def simpleTest():
    grid = VFKMGrid.VFKMGrid([-5, 5], [-1, 1], [0, 10], 3, 3)

    #   test matrices:
    print(grid.vertexMatCoords)
    print(grid.vertexLocations)
    print(grid.vertexIndexes)


    print('delta_x=' + str(grid.delta_x) + ', delta_y=' + str(grid.delta_y))
    L = grid.getLaplacian()
    print(L)
    face = grid.getFaceVertices([0.1, -0.9])
    print(face)
    assert (np.linalg.norm(face - np.array([4, 5, 8])) == 0)
    face = grid.getFaceVertices([4.9, -0.1])
    assert (np.linalg.norm(face - np.array([4, 8, 7])) == 0)
    print(face)

    l1 = grid.getVertexLocation(2,2)
    assert l1[0] == 0
    assert l1[1] == 0

    bar1 = grid.getBarycentricCoords([0, 0])
    print(bar1)

    bar2 = grid.getBarycentricCoords([4, 0.8])
    print(bar2)





if __name__ == '__main__':
    simpleTest()