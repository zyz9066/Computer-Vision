import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from skimage import io
from scipy import interpolate

path = 'assignment_2_data/'

class ImgReg(object):
    def __init__(self, I=None, J=None):
        if I is not None and J is not None:
            assert(isinstance(I, np.ndarray) and isinstance(J, np.ndarray))
            assert(I.shape == J.shape), 'I and J must have the same shape.'
        self.__eps = np.finfo('float32').eps
        self.__I = I
        self.__J = J
        
    def load(self, I):
        self.__I = I
        
    def joint_hist(self, I=None, J=None, bins=None, normed=False):
        if I is None:
            I = self.__I
        if J is None:
            J = self.__J
        assert(isinstance(I, np.ndarray) and isinstance(J, np.ndarray))
        assert(isinstance(normed, bool))
        assert(I.shape == J.shape), 'I and J must have the same shape.'
        
        sample = np.asarray([I.ravel(), J.ravel()]).T
        D = 2
        
        nbin = np.empty(D, int)
        edges = D*[None]
        
        if bins is None:
            bins = [I.max(), J.max()]
    
        try:
            M = len(bins)
            if M != D:
                raise ValueError('The dimension of bins must be equal to `image.size`.')
        except TypeError:
            # bins is an integer
            bins = D*[bins]
            
        # Create edge arrays
        for i in range(D):
            if np.ndim(bins[i]) == 0:
                if bins[i] < 1:
                    raise ValueError('`bins[%d]` must be positive, when an integer' % i)
                a = sample[:, i]
                if a.size == 0:
                    # handle empty arrays. Can't determine range, so use 0-1.
                    smin, smax = 0, 1
                else:
                    smin, smax = a.min(), a.max()
                    if not (np.isfinite(smin) and np.isfinite(smax)):
                        raise ValueError('detected range of [%d, %d] is not finite' % (smin, smax))
                # expand empty range to avoid divide by zero
                if smin == smax:
                    smin -= 0.5
                    smax += 0.5
                edges[i] = np.linspace(smin, smax, bins[i] + 1)
            elif np.ndim(bins[i]) == 1:
                edges[i] = np.asarray(bins[i])
                if np.any(edges[i][:-1] > edges[i][1:]):
                    raise ValueError('`bins[%d]` must be monotonically increasing, when an array' % i)
            else:
                raise ValueError('`bins[%d]` must be a scalar or 1d array' % i)
                
            nbin[i] = len(edges[i]) + 1 # includes an outlier on each end
            
        # Compute the bin number each sample falls into.
        Ncount = tuple(np.searchsorted(edges[i], sample[:, i], side='right') for i in range(D))
        
        # Using digitize, values that fall on an edge are put in the right bin.
        # For the rightmost bin, we want values equal to the right edge to be
        # counted in the last bin, and not as an outlier.
        for i in range(D):
            # Find which points are on the rightmost edge.
            on_edge = sample[:, i] == edges[i][-1]
            # Shift these points one bin to the left.
            Ncount[i][on_edge] -= 1
            
        # Compute the sample indices in the flattened histogram matrix.
        # This raises an error if the array is too large
        xy = np.ravel_multi_index(Ncount, nbin)
        
        # Compute the number of repetitions in xy and assign it to the
        # flattened histmat.
        hist = np.bincount(xy, minlength=nbin.prod())
        
        # Shape into a proper matrix
        hist = hist.reshape(nbin)
        
        hist = hist.astype(float, casting='safe')
        
        # Remove outliers (indices 0 and -1 for each dimension).
        core = D*(slice(1, -1),)
        hist = hist[core]
            
        if normed:
            # calculate the probability density function
            s = hist.sum()
            hist /= s
            
        if (hist.shape != nbin - 2).any():
            raise RuntimeError('Internal Shape Error')
        
        self.__hist = hist
        
    def get_joint_hist(self):
        return self.__hist
    
    def plot_joint_hist(self):
        hist = self.__hist
        hist[hist==0] = 0.5
        plt.imshow(np.log(hist.T), origin='low', aspect='auto')
        
    def ssd(self, I=None, J=None):
        if I is None:
            I = self.__I
        if J is None:
            J = self.__J
        assert(isinstance(I, np.ndarray) and isinstance(J, np.ndarray))
        assert(I.shape == J.shape), 'I and J must have the same shape.'
        diff = (I.ravel() - J.ravel()).astype('uint64')
        return np.dot(diff, diff)
    
    def corr(self, I=None, J=None):
        if I is None:
            I = self.__I
        if J is None:
            J = self.__J
        assert(isinstance(I, np.ndarray) and isinstance(J, np.ndarray))
        assert(I.shape == J.shape), 'I and J must have the same shape.'
        
        n = I.size
        assert(n >= 2), 'I and J must have length at least 2.'
        
        I = I.ravel()
        J = J.ravel()
        # If an input is constant, the rho is not defined.
        if (I == I[0]).all() or (J == J[0]).all():
            return np.nan
        
        # dtype is the data type for the calculation.
        dtype = type(1.0 + I[0] + J[0])
        
        if n == 2:
            return np.sign(I[1] - I[0])*np.sign(J[1]-J[0])
        
        Imean = I.mean(dtype=dtype)
        Jmean = J.mean(dtype=dtype)
        
        Im = I.astype(dtype) - Imean
        Jm = J.astype(dtype) - Jmean
        
        normIm = np.linalg.norm(Im)
        normJm = np.linalg.norm(Jm)
        
        r = np.dot(Im/normIm, Jm/normJm)
        
        r = max(min(r, 1.0), -1.0)
        
        return r
    
    def mi(self, I=None, J=None):
        if I is None:
            I = self.__I
        if J is None:
            J = self.__J
        self.joint_hist(I, J)
        Hij = self.get_joint_hist()
        
        nzi, nzj = np.nonzero(Hij)
        nz_val = Hij[nzi, nzj]
            
        Hij_sum = Hij.sum()
        pi = Hij.sum(axis=1).ravel()
        pj = Hij.sum(axis=0).ravel()
        log_Hij_nm = np.log(nz_val)
        Hij_nm = nz_val / Hij_sum
        # Don't need to calculate the full outer product, just for non-zeros
        outer = pi.take(nzi).astype('int64', copy=False) * pj.take(nzj).astype('int64', copy=False)
        log_outer = -np.log(outer) + np.log(pi.sum()) + np.log(pj.sum())
        mi = Hij_nm * (log_Hij_nm - np.log(Hij_sum)) + Hij_nm * log_outer
        return np.clip(mi.sum(), 0.0, None)
    
    def rigid_transform(self, I=None, theta=0, u=[0,0]):
        if I is None:
            I = self.__I
        xs, ys = np.arange(I.shape[0]), np.arange(I.shape[1])
        
        # matrix
        theta = np.deg2rad(theta)
        c, s = np.cos(theta), np.sin(theta)
        m = np.array([[c, -s, -u[1]], [s, c, -u[0]], [0, 0, 1]])
        
        pts = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)
        pts = np.hstack([pts, np.ones((pts.shape[0],1))])
        out = m.dot(pts.T).T[:, :-1].reshape(I.shape[0], I.shape[1], 2)
        
        return interpolate.interpn((xs, ys), I, out, bounds_error=False, fill_value=0)
    
    def register(self, J, mode='rigid', optimizer='gd', lr=1e-7, gamma=0.9, beta=0.9, iters=100):
        u, t = np.zeros(2), 0
        cost_history = np.zeros(iters)
        I = self.__I
        x, y = np.arange(I.shape[0]), np.arange(I.shape[1])
        vu, vt = np.zeros(2), 0
        for itr in range(iters):
            
            if optimizer == 'nag':
                vu *= gamma
                vt *= gamma
                curr = self.rigid_transform(J, theta=t-vt, u=u-vu)
            else:
                curr = self.rigid_transform(J, theta=t, u=u)
                
            c, s = np.cos(np.deg2rad(t)), np.sin(np.deg2rad(t))
            gy, gx = np.gradient(curr)
            dx, dy, dt = 0., 0., 0.
            if mode != 'rotation':
                dx = -((curr - I) * gx).sum() * 2
                dy = -((curr - I) * gy).sum() * 2
            if mode != 'translation':
                dt = -2 * ((curr - I) * (gy*(x*c-y*s) - gx*(x*s+y*c))).sum()
            
            du = np.array([dx, dy])
            if optimizer == 'gd':
                cu = lr * du
                ct = lr * dt
            elif optimizer == 'vgd':
                cu = lr * du / I.size
                ct = lr * dt / I.size
            elif optimizer == 'momentum':
                vu = gamma * vu + lr * du
                vt = gamma * vt + lr * dt
                cu, ct = vu, vt
            elif optimizer == 'nag':
                vu += lr * du
                vt += lr * dt
                cu, ct = vu, vt
            elif optimizer == 'adagrad':
                vu += du**2
                vt += dt**2
                cu = (lr / (np.sqrt(vu) + self.__eps)) * du
                ct = (lr / (np.sqrt(vt) + self.__eps)) * dt
            elif optimizer == 'rmsprop':
                vu = beta * vu + (1-beta) * du**2
                vt = beta * vt + (1-beta) * dt**2
                cu = (lr / (np.sqrt(vu) + self.__eps)) * du
                ct = (lr / (np.sqrt(vt) + self.__eps)) * dt
            u -= cu
            t -= ct
            cost_history[itr] = self.ssd(curr, I)
            
        return t%360, u, cost_history


'''
Part 1: Joint histogram
'''

I1 = io.imread(path + 'I1.png')
J1 = io.imread(path + 'J1.png')

for i in range(1, 7):
    if i == 1:
        globals()['I%s' % i] = io.imread(path + 'I' + str(i) +'.png')
        globals()['J%s' % i] = io.imread(path + 'J' + str(i) +'.png')
    else:
        globals()['I%s' % i] = io.imread(path + 'I' + str(i) +'.jpg')
        globals()['J%s' % i] = io.imread(path + 'J' + str(i) +'.jpg')

'''
a) JointHist(I, J, bins)
'''

def JointHist(I=None, J=None, bins=None, plot=False):
    img_reg = ImgReg(I, J)
    img_reg.joint_hist(bins=bins)
    if plot:
        img_reg.plot_joint_hist()
    else:
        return img_reg.get_joint_hist()

H = JointHist(I1, J1, bins=256)


'''
b) verify
'''

print(H.sum() == I1.size)

'''
c) show the joint histogram
'''

for i in range(1, 7):
    plt.subplot(2, 3, i)
    JointHist(globals()['I%s' % i], globals()['J%s' % i], plot=True)
plt.tight_layout()


'''
Part 2: similarity criteria
'''

'''
a) sum squared difference
'''

def SSD(I, J):
    img_reg = ImgReg(I, J)
    return img_reg.ssd()

print(SSD(I1, J1))

'''
b) pearson correlation coefficient
'''

def corr(I, J):
    img_reg = ImgReg(I, J)
    return img_reg.corr()

print(corr(I1, J1))

'''
c) mutual information
'''

def MI(I, J):
    img_reg = ImgReg(I, J)
    return img_reg.mi()

print(MI(I1, J1))

'''
d) results
'''

print('No.\tSSD\t\t\tcorr\t\tMI')
for i in range(1, 7):
    I = globals()['I%s' % i]
    J = globals()['J%s' % i]
    print('%d\t%.3e\t%.3f\t%.2f' % (i, SSD(I, J), corr(I, J), MI(I, J)))
    
'''
Part 3: spatial transforms
'''

class Mat3d(object):
    def __init__(self):
        # epsilon for testing whether a number is close to zero
        self.__eps = np.finfo('float32').eps
        # axis sequences for Euler angles
        self.__next_axis = [1, 2, 0, 1]
        # map axes to inner axis, parity, repetition, frame
        self.__axes = {
            'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
            'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
            'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
            'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
            'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
            'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
            'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
            'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}
        
    def __euler_mat(self, angles=[0,0,0], axes='sxyz'):
        firstaxis, parity, repetition, frame = self.__axes[axes]
        
        ai, aj, ak = angles
        i = firstaxis
        j = self.__next_axis[i+parity]
        k = self.__next_axis[i-parity+1]
        
        if frame:
            ai, ak = ak, ai
        if parity:
            ai, aj, ak = -ai, -aj, -ak
    
        si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
        ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
        cc, cs = ci*ck, ci*sk
        sc, ss = si*ck, si*sk
    
        M = np.identity(4)
        if repetition:
            M[i, i] = cj
            M[i, j] = sj*si
            M[i, k] = sj*ci
            M[j, i] = sj*sk
            M[j, j] = -cj*ss+cc
            M[j, k] = -cj*cs-sc
            M[k, i] = -sj*ck
            M[k, j] = cj*sc+cs
            M[k, k] = cj*cc-ss
        else:
            M[i, i] = cj*ck
            M[i, j] = sj*sc-cs
            M[i, k] = sj*cc+ss
            M[j, i] = cj*sk
            M[j, j] = sj*ss+cc
            M[j, k] = sj*cs-sc
            M[k, i] = -sj
            M[k, j] = cj*si
            M[k, k] = cj*ci
        return M
    
    def translation_mat(self, direction=[0,0,0]):
        M = np.identity(4)
        M[:3, 3] = direction[:3]
        return M
    
    def rotation_mat(self, angle=0., axis=[1,0,0], point=None):
        sina = np.sin(angle)
        cosa = np.cos(angle)
        axis /= np.linalg.norm(axis)
        # rotation matrix around unit vector
        R = np.diag([cosa, cosa, cosa])
        R += np.outer(axis, axis) * (1. - cosa)
        axis *= sina
        R += np.array([[0., -axis[2], axis[1]],
                       [axis[2], 0., -axis[0]],
                       [-axis[1], axis[0], 0.]])
        M = np.identity(4)
        M[:3, :3] = R
        if point is not None:
            # rotation not around origin
            point = np.array(point, dtype='float64', copy=False)
            M[:3, 3] = point - R.dot(point)
        return M
    
    def scale_mat(self, factor=1., origin=None, axis=None):
        if axis is None:
            # uniform scaling
            M = np.diag([factor]*3 + [1])
            if origin is not None:
                M[:3, 3] = origin
                M[:3, 3] *= 1. - factor
        else:
            # nonuniform scaling
            axis /= np.linalg.norm(axis)
            factor = 1. - factor
            M = np.identity(4)
            M[:3, :3] -= factor * np.outer(axis, axis)
            if origin is not None:
                M[:3, 3] = factor * np.dot(origin, axis) * axis
        return M
    
    def shear_mat(self, angle=0., axis=[1,0,0], point=[0,0,0], normal=[0,0,1]):
        normal /= np.linalg.norm(normal)
        axis /= np.linalg.norm(axis)
        if np.abs(np.dot(normal, axis)) > self.__eps:
            raise ValueError('axis and normal vectors are not orthogonal')
        angle = np.tan(angle)
        M = np.identity(4)
        M[:3, :3] += angle * np.outer(axis, normal)
        M[:3, 3] = -angle * np.dot(point[:3], normal) * axis
        return M
        
    def comp_mat(self, scale=None, angles=None, translate=None, shear=None):
        M = np.identity(4)
        if translate is not None:
            T = np.identity(4)
            T[:3, 3] = translate
            M = M.dot(T)
        if angles is not None:
            R = self.__euler_mat(angles)
            M = M.dot(R)
        if scale is not None:
            S = np.diag(scale+[1])
            M = M.dot(S)
        if shear is not None:
            Z = np.identity(4)
            Z[1, 2] = shear[2]
            Z[0, 2] = shear[1]
            Z[0, 1] = shear[0]
            M = M.dot(Z)
        M /= M[3, 3]
        return M
    
    def decomp_matrix(self, matrix):
        M = np.array(matrix, dtype='float64', copy=True).T
        if np.abs(M[3, 3]) < self.__eps:
            raise ValueError('`M[3, 3]` is zero')
        M /= M[3, 3]
            
        angles = np.zeros(3)
        
        translate = M[3, :3].copy()
        M[3, :3] = 0.
        
        row = M[:3, :3].copy()
        scale = np.linalg.norm(row[0])
        row /= scale
        
        if np.dot(row[0], np.cross(row[1], row[2])) < 0:
            scale = np.negative(scale)
            row = np.negative(row)
            
        angles[1] = np.arcsin(-row[0, 2])
        if np.cos(angles[1]):
            angles[0] = np.arctan2(row[1, 2], row[2, 2])
            angles[2] = np.arctan2(row[0, 1], row[0, 0])
        else:
            angles[0] = np.arctan2(-row[2, 1], row[1, 1])
            angles[2] = 0.0
            
        return scale, np.rad2deg(angles), translate
    

def warpAffine(p, m):
    return m.dot(np.hstack((p, np.ones((p.shape[0], 1)))).T).T[:, :-1]

def plot3d(pts, res=None, x=None, y=None, z=None):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(pts[:, 0], pts[:, 1], pts[:, 2], facecolor=(0,0,0,0), edgecolor='b', linewidth=0.5)
    if res is not None:
        ax.scatter3D(res[:, 0], res[:, 1], res[:, 2], facecolor=(0,0,0,0), edgecolor='r', linewidth=0.5)
    if x is not None:
        ax.set_xlim3d(x[0], x[1])
    if y is not None:
        ax.set_ylim3d(y[0], y[1])
    if z is not None:
        ax.set_zlim3d(z[0], z[1])

'''
a) points
'''

pts = np.array(np.meshgrid(np.arange(21), np.arange(21), np.arange(5))).T.reshape(-1, 3)

plot3d(pts, x=[0,20], y=[0,20], z=[0,20])

'''
b) rigid_transform()
'''

def rigid_transform(theta=0, omega=0, phi=0, p=0, q=0, r=0):
    angles = np.deg2rad([theta, omega, phi])
    matg = Mat3d()
    return matg.comp_mat(angles=angles, translate=[p, q, r])
    
m = rigid_transform(theta=0, omega=0, phi=0, p=5, q=10, r=15)
res = warpAffine(pts, m)

plot3d(pts, res, x=[0,30], y=[0,30], z=[0,30])

'''
c) affine_transform()
'''

def affine_transform(s=1, theta=0, omega=0, phi=0, p=0, q=0, r=0):
    angles = np.deg2rad([theta, omega, phi])
    matg = Mat3d()
    return matg.comp_mat(scale=[s]*3, angles=angles, translate=[p, q, r])

m = affine_transform(s=0.5, theta=90, omega=0, phi=20, p=-10, q=-5, r=0)
res = warpAffine(pts, m)

plot3d(pts, res, x=[0,30], y=[0,30], z=[0,30])

'''
d) type of transformation
'''

def get_matrices(matrix):
    matg = Mat3d()
    return matg.decomp_matrix(matrix)

M1 = np.array([[0.9045, -0.3847, -0.1840, 10.0000],
               [0.2939, 0.8750, -0.3847, 10.0000],
               [0.3090, 0.2939, 0.9045, 10.0000],
               [0, 0, 0, 1.0000]])
M2 = np.array([[-0.0000, -0.2598, 0.1500, -3.0000],
               [0.0000, -0.1500, -0.2598, 1.5000],
               [0.3000, -0.0000, 0.0000, 0],
               [0, 0, 0, 1.0000]])
M3 = np.array([[0.7182, -.3727, -0.5660, 1.8115],
               [-1.9236, -4.6556, -2.5512, 0.2873],
               [-0.6426, -1.7985, -1.6283, 0.7404],
               [0, 0, 0, 1.0000]])

def print_res(res):
    for s in res:
        print(s)
        
print('Order: scale, rotation, translation')
print('M1:')
print_res(get_matrices(M1))
print('M2:')
print_res(get_matrices(M2))
print('M3:')
print_res(get_matrices(M3))


'''
Part 4: simple 2d registration
'''

I1 = io.imread(path + 'BrainMRI_1.jpg')
I2 = io.imread(path + 'BrainMRI_2.jpg')
I3 = io.imread(path + 'BrainMRI_3.jpg')
I4 = io.imread(path + 'BrainMRI_4.jpg')

for i in range(1, 5):
    globals()['I%s' % i] = io.imread(path + 'BrainMRI_' + str(i) +'.jpg')
    
def register(I, J, mode='rigid', optim='gd', lr=1e-6, iters=100):
    img_reg = ImgReg(I)
    return img_reg.register(J, mode=mode, optimizer=optim, lr=lr, iters=iters)
        
'''
a) translation()
'''

def translation(I, p=0, q=0, plot=False):
    img_reg = ImgReg()
    res = img_reg.rigid_transform(I, u=[p, q])
    if plot:
        plt.imshow(res)
    else:
        return res

translation(I1, 20, 50, plot=True)


'''
b) translation search
'''
t, u, ch = register(I1, I4, mode='translation', lr=1e-7, iters=500)

translation(I4, u[0], u[1], plot=True)

for i in range(2, 5):
    t, u, ch = register(I1, globals()['I%s' % i], mode='translation')
    print(u)
    plt.subplot(1, 3, i-1)
    plt.plot(ch)

'''
c) rotation()
'''

def rotation(I, theta=0, plot=False):
    img_reg = ImgReg()
    res = img_reg.rigid_transform(I, theta=theta)
    if plot:
        plt.imshow(res)
    else:
        return res

rotation(I1, 15, plot=True)


'''
d) rotation search
'''
    
t, u, ch = register(I1, I2, mode='rotation', lr=1e-10, iters=1000)
rotation(I2, t, plot=True)

for i in range(2, 5):
    t, u, ch = register(I1, globals()['I%s' % i], mode='rotation', lr=1e-9)
    print(t)
    plt.subplot(1, 3, i-1)
    plt.plot(ch)


'''
d) gradient descent
'''

def rigid_transform2d(I, theta=0, p=0., q=0., plot=False):
    img_reg = ImgReg()
    res = img_reg.rigid_transform(I, theta=theta, u=[p,q])
    if plot:
        plt.imshow(res)
    else:
        return res

t, u, ch = register(I1, I4, mode='rigid', lr=1e-9, iters=1000)
rigid_transform2d(I4, t, u[0], u[1], plot=True)

optim = ['gd', 'momentum', 'nag', 'adagrad', 'rmsprop']
for op in optim:
    t, u, ch = register(I1, I4, mode='rigid', optim=op, lr=1e-10, iters=900)
    plt.plot(ch, label=op)
plt.legend()

optim = ['adagrad', 'rmsprop']
for op in optim:
    t, u, ch = register(I1, I2, mode='rigid', optim=op, lr=0.1, iters=900)
    plt.plot(ch, label=op)
plt.legend()

t, u, ch = register(I1, I4, mode='rigid', optim='adagrad', lr=1, iters=900)

t, u, ch = register(I1, I4, mode='rigid', optim='rmsprop', lr=0.1, iters=1000)
rigid_transform2d(I2, t, u[0], u[1], plot=True)

for i in range(2, 5):
    t, u, ch = register(I1, globals()['I%s' % i])
    print(t, u)
    plt.plot(ch)


for i in range(2, 5):
    t, u, ch = register(I1, globals()['I%s' % i], optim='nag')
    print(t, u)
    plt.plot(ch)