import SoftVectorFieldKMeans

def simpleTest():
    svfkm = SoftVectorFieldKMeans.SoftVectorFieldKMeans(5, 'curves.txt', 3, verbose=2)

if __name__ == '__main__':
    simpleTest()