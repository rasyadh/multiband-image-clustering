import numpy as np
import cv2
import matplotlib

# use for non windows
# matplotlib.use('Agg')

import matplotlib.pyplot as plt

class MultibandImageCluster:
    def __init__(self, path):
        self.path = path
        np.set_printoptions(threshold=np.nan)

    def __destroy__(self):
        cv2.destroyAllWindows()

    def read_images(self):
        # Read image in grayscale
        # path = 'project/static/landsat7/'
        gb = {
            'gb1': cv2.imread(self.path + 'gb1.jpg', 0),
            'gb2': cv2.imread(self.path + 'gb2.jpg', 0),
            'gb3': cv2.imread(self.path + 'gb3.jpg', 0),
            'gb4': cv2.imread(self.path + 'gb4.jpg', 0),
            'gb5': cv2.imread(self.path + 'gb5.jpg', 0),
            'gb7': cv2.imread(self.path + 'gb7.jpg', 0)
        }
        return gb

    def feature_space_transformation(self, gb):
        feature_space = []
        for i in range(0, len(gb['gb1'])):
            for j in range(0, len(gb['gb1'])):
                pixel = np.asarray([
                    gb['gb1'][i][j], gb['gb2'][i][j], gb['gb3'][i][j], 
                    gb['gb4'][i][j], gb['gb5'][i][j], gb['gb7'][i][j]
                ])
                feature_space.append(pixel)
        feature_space = np.asarray(feature_space)
        return feature_space
        
    def KMeans_clustering(self, feature_space, cluster, iteration):
        feature_space = np.float32(feature_space)
        criteria = (cv2.TERM_CRITERIA_MAX_ITER, iteration, 1.0)
        ret, label, center = cv2.kmeans(feature_space, cluster, None, criteria, iteration, cv2.KMEANS_PP_CENTERS)

        arrclus = ['cA', 'cB', 'cC', 'cD', 'cE', 'cF']
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        clusters = {}

        for i in range(0, cluster):
            clusters[arrclus[i]] = feature_space[label.ravel() == i]

        for i in range(0, cluster):
            plt.scatter(clusters[arrclus[i]][:,0], clusters[arrclus[i]][:,1], c=colors[i])

        plt.scatter(center[:,0], center[:,1], s = 80, c = 'k', marker = 's')
        plt.xlabel('Height'),plt.ylabel('Weight')
        plt.savefig('project/static/multiband/cluster' + str(cluster) + '.jpg')
        return label

    def image_creation(self, feature, feature_space, label, cluster):
        feature = feature.tolist()
        img_creation = [[0, 0, 0] for i in range(len(feature))]

        colors = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 255, 0], [255, 0, 255], [0, 255, 255]]

        for i in range(0, cluster):
            for j in feature_space[label.ravel() == i]:
                j = j.tolist()
                indeks = feature.index(j)
                img_creation[indeks] = colors[i]
        
        img = []
        num_pixel = -1
        for i in range(0, 32):
            row = []
            for j in range(0, 32):
                num_pixel = num_pixel + 1
                piksel = img_creation[num_pixel]
                row.append(piksel)
            img.append(row)

        img = np.asarray(img, dtype=np.uint8)
        cv2.imwrite('project/static/multiband/result-multiband' + str(cluster) + '.jpg', img)
