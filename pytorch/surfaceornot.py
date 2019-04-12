depth = np.random.rand(160, 160)
plane = np.zeros_like(depth).astype('uint16')
shape = np.squeeze(depth.shape)
center = (shape/2).astype('int32')

nof_points = 3
left, right = int(center[1]/2), center[1] + int(center[1]/2)
top, down = int(center[0]/2), center[0] + int(center[0]/2)

xs = np.random.choice(np.arange(left, right), nof_points * 3)
ys = np.random.choice(np.arange(top, down), nof_points * 3)
zs = np.float32((np.random.randint(200,400) * np.random.random(nof_points * 3)) % np.random.randint(11,18))
xs.sort();ys.sort()

plane[ys, xs] = zs
print('X: {}'.format(xs))
print('Y: {}'.format(ys))
print('Z: {}'.format(zs))

plt.imshow(plane, 'gray')
plt.show()


print('X: {}'.format(xs))
print('Y: {}'.format(ys))
print('Z: {}'.format(zs))

plt.imshow(plane, 'gray')
plt.show()

'''
已知三点p1（x1,y1,z1），p2(x2,y2,z2)，p3(x3,y3,z3)，要求确定的平面方程。关键在于求出平面的一个法向量，为此做向量
$$p1p2=(x2-x1,y2-y1,z2-z1),$$ $$p1p3=(x3-x1,y3-y1,z3-z1),$$平面法线和这两个向量垂直，因此法向量n：
![](https://img-my.csdn.net/uploads/201304/10/1365596136_2705.png)
'''

def generate_verticalvectors(points):
    '''points is a nx3 array, each row implies a point coordinates
    '''
    p0 = points[0, :]
    p1 = points[1, :]
    p2 = points[2, :]
    v01 = p1-p0  # np.array([p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]])
    v02 = p2-p0  # np.array([p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]])
    '''
    verticalvector = np.array([v01[1]*v02[2]-v02[1]*v01[2],
                     v01[2]*v02[0]-v02[2]*v01[0],
                     v01[0]*v02[1]-v02[0]*v01[1]])
    '''
    verticalvector = np.cross(v01, v02)
    return verticalvector
    
tri1 = np.stack([xs[:3], ys[:3], zs[:3]], axis=1)
tri2 = np.stack([xs[3:6], ys[3:6], zs[3:6]], axis=1)

vector1 = generate_verticalvectors(tri1)
vector2 = generate_verticalvectors(tri2)
print('Vertical Vector:\nvector1: {}\nvector2: {}'.format(vector1, vector2))

num = vector1@vector2.T
denom = np.linalg.norm(vector1) * np.linalg.norm(vector2)  
cos = num / denom #余弦值 
