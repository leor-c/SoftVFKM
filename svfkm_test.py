import SoftVectorFieldKMeans

def simpleTest():
    svfkm = SoftVectorFieldKMeans.SoftVectorFieldKMeans(9, 'curves.txt', 10, smoothnessWeight=0.1, certaintyWeight=1, verbose=2)
    svfkm.convergeClusters(1e-7, 50)
    svfkm.plotResults()

if __name__ == '__main__':
    simpleTest()