import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
def k_means_(df,k):
    m=len(df)
    cludter_centers=np.zeros((k,2))
    index=np.random.choice(range(m))
    cludter_centers[0][0]=np.copy(df['x'][index])
    cludter_centers[0][1]=np.copy(df['y'][index])
    d=[0.0 for i in range(m)]
    for i in range(1,k):
        sum_all=0.0
        for j in range(m):
            d[i]=nearest(df['x'][i],df['y'][i],cludter_centers)
            sum_all+=d[i]
        sum_all *= random.random()
        for k,n in enumerate(d):
            sum_all-=n
            if sum_all>0:
                continue
            cludter_centers[i][0]=np.copy(df['x'][k])
            cludter_centers[i][1]=np.copy(df['y'][k])
            break
    return cludter_centers
def nearest(x,y,cludter_centers):
    min_dist=2**31
    m=np.shape(cludter_centers)[0]
    for i in range(m):
        d=np.sqrt((x-cludter_centers[i][0])**2+(y-cludter_centers[i][1])**2)
        if min_dist>d:
            min_dist=d
    return min_dist
def assignment(df, centroids, colmap):
    for i in range(len(centroids[0])):

        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
    distance_from_centroid_id = ['distance_from_{}'.format(i) for i in range((len(centroids[0])))]
    df['closest'] = df.loc[:, distance_from_centroid_id].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df
def update(df, centroids):
    for i in range(len(centroids[0])):
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return centroids
def main():
    df = pd.DataFrame({
        'x': [12, 20, 28, 18, 10, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72, 23],
        'y': [39, 36, 30, 52, 54, 20, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24, 77]
    })

    k = 3
    centroids = k_means_(df,k)
    colmap = {0: 'r', 1: 'g', 2: 'b'}
    df = assignment(df, centroids, colmap)

    plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
    for i in range(len(centroids[0])):
        plt.scatter(*centroids[i], color=colmap[i], linewidths=6)
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.show()

    for i in range(10):
        key = cv2.waitKey()
        plt.close()

        closest_centroids = df['closest'].copy(deep=True)
        centroids = update(df, centroids)

        plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
        for i in range(len(centroids[0])):
            plt.scatter(*centroids[i], color=colmap[i], linewidths=6)
        plt.xlim(0, 80)
        plt.ylim(0, 80)
        plt.show()

        df = assignment(df, centroids, colmap)

        if closest_centroids.equals(df['closest']):
            break


if __name__ == '__main__':
    main()




