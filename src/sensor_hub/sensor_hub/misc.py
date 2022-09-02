import numpy as np


def distance_point_to_segment(P, A, B):
    AP = P-A
    BP = P-B
    AB = B-A
    if np.dot(AB, AP) >= 0 and np.dot(-AB, BP)>=0:
#         print('-1:',AP,BP,AB,A,B)
        return np.abs(np.cross(AP, AB))/np.linalg.norm(AB), np.dot(AP, AB)/np.dot(AB, AB) * AB + A
    d_PA = np.linalg.norm(AP)
    d_PB = np.linalg.norm(BP)
#     print('0:',d_PA,d_PB,A,B)
    if d_PA < d_PB:
        return d_PA, A
    return d_PB, B

def min_distance_cuboids(cub1, cub2):
    
    minD = 1e5
    for i in range(4):
        for j in range(4):
            d, Q = distance_point_to_segment(cub1[i, :2], cub2[j,:2], cub2[j+1, :2])
            if d < minD:
                minD = d
                minP = cub1[i, :2]
                minQ = Q
#                 print('1:',d, Q)
    for i in range(4):
        for j in range(4):
            d, Q = distance_point_to_segment(cub2[i, :2], cub1[j,:2], cub1[j+1, :2])
            
            if d < minD:
                minD = d
                minP = cub2[i,:2]
                minQ = Q
#                 print('2:',d, Q)
    return minP, minQ, minD