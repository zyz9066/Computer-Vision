import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


'''
Part 1: Python viewer
'''

'''
Part 1a:
'''

class Viewer:
    def __init__(self, fig):
        self.__fig = fig
        self.__views = ('sagittal', 'coronal', 'axial', 'all')
        self.__data = None
        self.set_params()
        
    def __connect(self):
        # connect to mouse and keypad
        self.__cidpress = self.__fig.canvas.mpl_connect('key_press_event',
                                                      self.__key_press)
        self.__cidscroll = self.__fig.canvas.mpl_connect('scroll_event',
                                                       self.__mouse_scroll)
        
    def __disconnect(self):
        # disconnect stored connection ids
        self.__fig.canvas.mpl_disconnect(self.__cidpress)
        self.__fig.canvas.mpl_disconnect(self.__cidscroll)
        
    # keypad callback
    def __key_press(self, event):
        if self.__view != self.__views[3]:
            if event.key == self.__keys[0]:
                self.__index -= 1
            elif event.key == self.__keys[1]:
                self.__index += 1
            
            if self.__view == self.__views[0]:
                self.__index %= self.__data.shape[0]
                self.__img = self.__data[self.__index, :, :]
            elif self.__view == self.__views[1]:
                self.__index %= self.__data.shape[1]
                self.__img = self.__data[:, self.__index, :]
            elif self.__view == self.__views[2]:
                self.__index %= self.__data.shape[2]
                self.__img = self.__data[:, :, self.__index]
            self.__img = np.rot90(self.__img)
            self.__array = self.__img
            
            if self.__histeq:
                self.__img = self.__hist_eq(self.__img)
                self.__array = self.__img
                
            if self.__fft:
                self.__array = np.fft.fftshift(np.fft.fft2(self.__img))
                self.__img = np.abs(np.log(self.__array))
                    
            self.__im.set_data(self.__img)
            self.__ax.set_title('%s %d' % (self.__view, self.__index))
            
        elif self.__view == self.__views[3]:
            if event.key == self.__keys[0]:
                self.__indices -= 1
            elif event.key == self.__keys[1]:
                self.__indices += 1
            
            self.__indices[0] %= self.__data.shape[0]
            self.__imgs[0] = np.rot90(self.__data[self.__indices[0], :, :])
            self.__arrays[0] = self.__imgs[0]
            if self.__histeq:
                self.__imgs[0] = self.__hist_eq(self.__imgs[0])
                self.__arrays[0] = self.__imgs[0]
            if self.__fft:
                self.__arrays[0] = np.fft.fftshift(np.fft.fft2(self.__imgs[0]))
                self.__imgs[0] = np.abs(np.log(self.__array))
            self.__ims[0].set_data(self.__imgs[0])
            self.__axes[0].set_title('%s %d' % (self.__views[0],
                                                self.__indices[0]))
            
            self.__indices[1] %= self.__data.shape[1]
            self.__imgs[1] = np.rot90(self.__data[:, self.__indices[1], :])
            self.__arrays[1] = self.__imgs[1]
            if self.__histeq:
                self.__imgs[1] = self.__hist_eq(self.__imgs[1])
                self.__arrays[1] = self.__imgs[1]
            if self.__fft:
                self.__arrays[1] = np.fft.fftshift(np.fft.fft2(self.__imgs[1]))
                self.__imgs[1] = np.abs(np.log(self.__arrays[1]))
            self.__ims[1].set_data(self.__imgs[1])
            self.__axes[1].set_title('%s %d' % (self.__views[1],
                                                self.__indices[1]))
            
            self.__indices[2] %= self.__data.shape[2]
            self.__imgs[2] = np.rot90(self.__data[:, :, self.__indices[2]])
            self.__arrays[2] = self.__imgs[2]
            if self.__histeq:
                self.__imgs[2] = self.__hist_eq(self.__imgs[2])
                self.__arrays[2] = self.__imgs[2]
            if self.__fft:
                self.__arrays[2] = np.fft.fftshift(np.fft.fft2(self.__imgs[2]))
                self.__imgs[2] = np.abs(np.log(self.__arrays[2]))
            self.__ims[2].set_data(self.__imgs[2])
            self.__axes[2].set_title('%s %d' % (self.__views[2],
                                                self.__indices[2]))
         
        self.__fig.canvas.draw()
      
    # mouse wheel callback
    def __mouse_scroll(self, event):
        if self.__view != self.__views[3]:
            if event.button == self.__keys[0]:
                self.__index -= 1
            elif event.button == self.__keys[1]:
                self.__index += 1
                
            if self.__view == self.__views[0]:
                self.__index %= self.__data.shape[0]
                self.__img = self.__data[self.__index, :, :]
            elif self.__view == self.__views[1]:
                self.__index %= self.__data.shape[1]
                self.__img = self.__data[:, self.__index, :]
            elif self.__view == self.__views[2]:
                self.__index %= self.__data.shape[2]
                self.__img = self.__data[:, :, self.__index]
            self.__img = np.rot90(self.__img)
            self.__array = self.__img
            
            if self.__histeq:
                self.__img = self.__hist_eq(self.__img)
                self.__array = self.__img
            
            if self.__fft:
                self.__array = np.fft.fftshift(np.fft.fft2(self.__img))
                self.__img = np.abs(np.log(self.__array))
                
            self.__im.set_data(self.__img)
            self.__ax.set_title('%s %d' % (self.__view, self.__index))
            
        elif self.__view == self.__views[3]:
            ax = event.inaxes
            if ax is not None:
                view = ax.get_title().partition(' ')[0]
                
                if view == self.__views[0]:
                    if event.button == self.__keys[0]:
                        self.__indices[0] -= 1
                    elif event.button == self.__keys[1]:
                        self.__indices[0] += 1
                    self.__indices[0] %= self.__data.shape[0]
                    self.__imgs[0] = np.rot90(self.__data[self.__indices[0], :, :])
                    self.__arrays[0] = self.__imgs[0]
                    if self.__histeq:
                        self.__imgs[0] = self.__hist_eq(self.__imgs[0])
                        self.__arrays[0] = self.__imgs[0]
                    if self.__fft:
                        self.__arrays[0] = np.fft.fftshift(np.fft.fft2(self.__imgs[0]))
                        self.__imgs[0] = np.abs(np.log(self.__arrays[0]))
                    self.__ims[0].set_data(self.__imgs[0])
                    self.__axes[0].set_title('%s %d' % (self.__views[0],
                                                        self.__indices[0]))
                
                elif view == self.__views[1]:
                    if event.button == self.__keys[0]:
                        self.__indices[1] -= 1
                    elif event.button == self.__keys[1]:
                        self.__indices[1] += 1
                    self.__indices[1] %= self.__data.shape[1]
                    self.__imgs[1] = np.rot90(self.__data[:, self.__indices[1], :])
                    self.__arrays[1] = self.__imgs[1]
                    if self.__histeq:
                        self.__imgs[1] = self.__hist_eq(self.__imgs[1])
                        self.__arrays[1] = self.__imgs[1]
                    if self.__fft:
                        self.__arrays[1] = np.fft.fftshift(np.fft.fft2(self.__imgs[1]))
                        self.__imgs[1] = np.abs(np.log(self.__arrays[1]))
                    self.__ims[1].set_data(self.__imgs[1])
                    self.__axes[1].set_title('%s %d' % (self.__views[1],
                                                        self.__indices[1]))
                elif view == self.__views[2]:
                    if event.button == self.__keys[0]:
                        self.__indices[2] -= 1
                    elif event.button == self.__keys[1]:
                        self.__indices[2] += 1
                    self.__indices[2] %= self.__data.shape[2]
                    self.__imgs[2] = np.rot90(self.__data[:, :, self.__indices[2]])
                    self.__arrays[2] = self.__imgs[2]
                    if self.__histeq:
                        self.__arrays[2] = self.__hist_eq(self.__imgs[2])
                        self.__arrays[2] = self.__imgs[2]
                    if self.__fft:
                        self.__arrays[2] = np.fft.fftshift(np.fft.fft2(self.__imgs[2]))
                        self.__imgs[2] = np.abs(np.log(self.__arrays[2]))
                    self.__ims[2].set_data(self.__imgs[2])
                    self.__axes[2].set_title('%s %d' % (self.__views[2],
                                                        self.__indices[2]))
                
        self.__fig.canvas.draw()
        
        # histogram equalization
    def __hist_eq(self, array, bins=256):
        # get image histogram
        hist, bins = np.histogram(array.ravel(), bins=bins, density=True)
        # cumulative distribution function
        cdf = hist.cumsum()
        # normalize
        cdf = 255 * cdf / cdf[-1]
        # use linear interpolation of cdf to find new pixel value
        arrayeq = np.interp(array.ravel(), bins[:-1], cdf)
        
        return arrayeq.reshape(array.shape)  
    
    def load(self, data):
        assert (type(data) is np.memmap), 'Invalid data type!'
        self.__data = data
        
    def set_params(self, index=None, view='axial', histeq=False,
                   cmap=None, value_range=(None, None), aspect=None, fft=False):
        if index is None:
            self.__index = np.random.randint(np.iinfo(np.uint16).max)
        else:
            assert (type(index) is int), 'Not integer type!'
            self.__index = index
            
        assert (view in set(self.__views)), 'Invalid view type!'
        self.__view = view
        assert (type(histeq) is bool), 'Not boolean type!'
        self.__histeq = histeq
        assert (type(fft) is bool), 'Not boolean type!'
        self.__fft = fft
        self.__cmap = cmap
        self.__vmin, self.__vmax = value_range
        self.__aspect = aspect
    
    def process(self):
        assert (type(self.__data) is not None), 'Invalid data!'
        
        if self.__view != self.__views[3]:
            if self.__view == self.__views[0]:
                self.__index %= self.__data.shape[0]
                self.__img = self.__data[self.__index, :, :]
            elif self.__view == self.__views[1]:
                self.__index %= self.__data.shape[1]
                self.__img = self.__data[:, self.__index, :]
            elif self.__view == self.__views[2]:
                self.__index %= self.__data.shape[2]
                self.__img = self.__data[:, :, self.__index]
            self.__img = np.rot90(self.__img)
            self.__array = self.__img
            
            if self.__histeq:
                self.__img = self.__hist_eq(self.__img)
                self.__array = self.__img
            
            self.__extent = None
            if self.__fft:
                self.__array = np.fft.fftshift(np.fft.fft2(self.__array))
                self.__img = np.abs(np.log(self.__array))
                freq_x = np.fft.fftfreq(self.__img.shape[0])
                freq_y = np.fft.fftfreq(self.__img.shape[1])
                self.__extent = (freq_x.min(), freq_x.max(),
                                     freq_y.min(), freq_y.max())
            
        elif self.__view == self.__views[3]:
            self.__indices = self.__index * np.ones(self.__data.ndim, dtype=int)
            self.__imgs = [None] * self.__data.ndim
            self.__arrays = [None] * self.__data.ndim
            self.__extents = [None] * self.__data.ndim
            
            self.__indices[2] %= self.__data.shape[2]
            self.__imgs[2] = np.rot90(self.__data[:, :, self.__indices[2]])
            self.__arrays[2] = self.__imgs[2]
            if self.__histeq:
                self.__imgs[2] = self.__hist_eq(self.__imgs[2])
                self.__arrays[2] = self.__imgs[2]
            if self.__fft:
                self.__arrays[2] = np.fft.fftshift(np.fft.fft2(self.__imgs[2]))
                self.__imgs[2] = np.abs(np.log(self.__arrays[2]))
                freq_x = np.fft.fftfreq(self.__imgs[2].shape[0])
                freq_y = np.fft.fftfreq(self.__imgs[2].shape[1])
                self.__extents[2] = (freq_x.min(), freq_x.max(),
                                     freq_y.min(), freq_y.max())         
            
            self.__indices[0] %= self.__data.shape[0]
            self.__imgs[0] = np.rot90(self.__data[self.__indices[0], :, :])
            self.__arrays[0] = self.__imgs[0]
            if self.__histeq:
                self.__imgs[0] = self.__hist_eq(self.__imgs[0])
                self.__arrays[0] = self.__imgs[0]
            if self.__fft:
                self.__arrays[0] = np.fft.fftshift(np.fft.fft2(self.__imgs[0]))
                self.__imgs[0] = np.abs(np.log(self.__arrays[0]))
                freq_x = np.fft.fftfreq(self.__imgs[0].shape[0])
                freq_y = np.fft.fftfreq(self.__imgs[0].shape[1])
                self.__extents[0] = (freq_x.min(), freq_x.max(),
                                     freq_y.min(), freq_y.max())            
            
            self.__indices[1] %= self.__data.shape[1]
            self.__imgs[1] = np.rot90(self.__data[:, self.__indices[1], :])
            self.__arrays[1] = self.__imgs[1]
            if self.__histeq:
                self.__imgs[1] = self.__hist_eq(self.__imgs[1])
                self.__arrays[1] = self.__imgs[1]
            if self.__fft:
                self.__arrays[1] = np.fft.fftshift(np.fft.fft2(self.__imgs[1]))
                self.__imgs[1] = np.abs(np.log(self.__arrays[1]))
                freq_x = np.fft.fftfreq(self.__imgs[1].shape[0])
                freq_y = np.fft.fftfreq(self.__imgs[1].shape[1])
                self.__extents[1] = (freq_x.min(), freq_x.max(),
                                     freq_y.min(), freq_y.max())
        
    def get_array(self):
        if self.__view != self.__views[3]:
            return self.__array
        elif self.__view == self.__views[3]:
            return self.__arrays
        
    def plot(self):
        self.__keys = ('up', 'down')
        self.__connect()
        
        if self.__view != self.__views[3]:
            self.__ax = self.__fig.add_subplot(111)
            self.__im = self.__ax.imshow(self.__img, cmap=self.__cmap, extent=self.__extent,
                                         vmin=self.__vmin, vmax=self.__vmax, aspect=self.__aspect)
            self.__ax.set_title('%s %d' % (self.__view, self.__index))
            
        elif self.__view == self.__views[3]:
            self.__axes = [None] * self.__data.ndim
            self.__ims = [None] * self.__data.ndim
            
            self.__axes[2] = self.__fig.add_subplot(221)
            self.__ims[2] = self.__axes[2].imshow(self.__imgs[2], cmap=self.__cmap,
                                                  extent=self.__extents[2], aspect=self.__aspect,
                                                  vmin=self.__vmin, vmax=self.__vmax)
            self.__axes[2].set_title('%s %d' % (self.__views[2], self.__indices[2]))
            
            self.__axes[0] = self.__fig.add_subplot(222)
            self.__ims[0] = self.__axes[0].imshow(self.__imgs[0], cmap=self.__cmap,
                                                  extent=self.__extents[0], aspect=self.__aspect,
                                                  vmin=self.__vmin, vmax=self.__vmax)
            self.__axes[0].set_title('%s %d' % (self.__views[0], self.__indices[0]))
            
            self.__axes[1] = self.__fig.add_subplot(223)
            self.__ims[1] = self.__axes[1].imshow(self.__imgs[1], cmap=self.__cmap,
                                                  extent=self.__extents[1], aspect=self.__aspect,
                                                  vmin=self.__vmin, vmax=self.__vmax)
            self.__axes[1].set_title('%s %d' % (self.__views[1], self.__indices[1]))
        
            plt.tight_layout()
            
                
# viewer function
def viewer(data, index=None, view='axial', histeq=False, fft=False, cmap=None,
           vmin=None, vmax=None, aspect=None, ret_val=False):
        
    fig = plt.figure()
    global view_obj
    view_obj = Viewer(fig)
    view_obj.load(data)
    view_obj.set_params(index=index, view=view, histeq=histeq, fft=fft,
                        cmap=cmap, value_range=(vmin, vmax), aspect=aspect)
    view_obj.process()
    if ret_val:
        return view_obj.get_array()
    else:
        view_obj.plot()
                
 
# load t1
t1_data = nib.load('images/t1.nii').get_fdata()


viewer(t1_data, index=250, view='axial', vmin=20, vmax=60)

viewer(t1_data, index=250, view='axial', cmap='hot')

viewer(t1_data, index=250, view='coronal', histeq=True)

viewer(t1_data, view='all')



'''
Part 2: Modalities and frequency-domain filtering
'''

'''
Part 2a:
'''

# load t2
t2_data = nib.load('images/t2.nii').get_fdata()

viewer(t2_data, index=250, view='axial')

# t2 2d FFT
viewer(t2_data, index=250, view='axial', fft=True)


'''
Part 2b:
'''

# Gaussian filter
def gaussian_kernel(size, sigma=10, ptype='low', dim=2, verbose=False):
        
    if dim == 1:
        norm_v = np.vectorize(lambda x: np.exp(-(x/sigma)**2 / 2) /\
                              (np.sqrt(2*np.pi)*sigma))
        kernel = norm_v(np.linspace(-size // 2, size // 2, size)) 
    elif dim == 2:
        sz_x, sz_y = size
        X, Y = np.mgrid[:sz_x, :sz_y]
        xpr = X - int(sz_x) // 2
        ypr = Y - int(sz_y) // 2
        kernel = np.exp(-((xpr**2+ypr**2) / (2*sigma**2))) / (2*np.pi*sigma**2)
    
    if ptype == 'high':
        kernel = 1 - kernel
        
    if verbose:
        plt.title('sigma=%d' % sigma)
        if dim == 1:
            plt.plot(kernel)
        elif dim == 2:
            plt.imshow(kernel)
    
    return kernel

# load swi
swi_data = nib.load('images/swi.nii').get_fdata()


# Gaussian blur
rotim = viewer(swi_data, index=250, view='axial', fft=True, ret_val=True)

count = 1
for sigma in range(1, 25, 5):
    gaussfilt = gaussian_kernel(rotim.shape, sigma=sigma)
    plt.subplot(2, 3, count)
    plt.imshow(np.abs(np.fft.ifft2(np.fft.ifftshift(gaussfilt*rotim))))
    count += 1
plt.suptitle('Gaussian blur')
plt.tight_layout()


'''
Part 2c:
'''

# Circular pass filter mask
def cpf(size, r=10, ptype='high', verbose=False):
    rows, cols = size
    crow, ccol = int(rows / 2), int(cols / 2)

    cntr = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    
    if ptype == 'band':
        mask_area = np.logical_and(((x-cntr[0])**2 + (y-cntr[1])**2 >= r[0]**2),
                                   ((x-cntr[0])**2 + (y-cntr[1])**2 <= r[1]**2))
    else:
        mask_area = (x - cntr[0]) ** 2 + (y - cntr[1]) ** 2 <= r**2
        
    if ptype == 'high':
        mask = np.ones((rows, cols))
        mask[mask_area] = 0
    elif ptype == 'low' or ptype == 'band':
        mask = np.zeros((rows, cols))
        mask[mask_area] = 1
        
    if verbose:
        plt.imshow(mask)
        plt.title("Pass filter")
        
    return mask

def separate_axial(data, index=250):
        
    # raw image
    plt.subplot(221)
    plt.imshow(np.rot90(data[:, :, index]))
    
    # fft
    plt.subplot(222)
    
    rotim = np.fft.fftshift(np.rot90(np.fft.fftn(data, axes=(0,1))[:, :, index]))
    freq_x = np.fft.fftfreq(rotim.shape[0])
    freq_y = np.fft.fftfreq(rotim.shape[1])
    
    plt.imshow(np.abs(np.log(rotim)), extent=(freq_x.min(), freq_x.max(),
                                           freq_y.min(), freq_y.max()))
    
    # edge enhancement
    plt.subplot(223)
    hpf = cpf(rotim.shape, ptype='high')
    plt.imshow(np.abs(np.fft.ifft2(np.fft.ifftshift(hpf*rotim))))
    
    # smoothing
    plt.subplot(224)
    lpf = cpf(rotim.shape, ptype='low')
    plt.imshow(np.abs(np.fft.ifft2(np.fft.ifftshift(lpf*rotim))))
    
    plt.suptitle('axial %d' % index)
    plt.tight_layout()

# separate plot t1
separate_axial(t1_data, index=250)
# separate plot t2
separate_axial(t2_data, index=250)
# separate plot swi
separate_axial(swi_data, index=250)

# load tof
tof_data = nib.load('images/tof.nii').get_fdata()

# separater plot tof
separate_axial(tof_data, index=50)

# load bold
bold_data = nib.load('images/bold.nii').get_fdata()

# separate plot bold
separate_axial(bold_data, index=30)

'''
Square filter
'''

def square_kernel(size, a=21, ptype='low', verbose=False):
    rows, cols = size
    xline = np.ones(rows)
    yline = np.ones(cols)
    xf = gaussian_kernel(rows, sigma=a, dim=1, ptype=ptype)
    yf = gaussian_kernel(cols, sigma=a, dim=1, ptype=ptype)
    xkernel = np.outer(xline.T, yf.T)
    ykernel = np.outer(xf.T, yline.T)
    kernel = np.minimum(xkernel, ykernel)
    
    if verbose:
        plt.imshow(kernel)
        plt.title("Square filter")
 
    return kernel

rotim = viewer(swi_data, index=250, view='axial', fft=True, ret_val=True)
freq_x = np.fft.fftfreq(rotim.shape[0])
freq_y = np.fft.fftfreq(rotim.shape[1])

count = 1
for a in range(1, 25, 5):
    plt.subplot(2, 3, count)
    plt.imshow(square_kernel(rotim.shape, a=a),
               extent=(freq_x.min(), freq_x.max(), freq_y.min(), freq_y.max()))
    count += 1
plt.suptitle('Square filter')
plt.tight_layout()

# Square blur

count = 1
for a in range(1, 25, 5):
    sfilt = square_kernel(rotim.shape, a=a)
    plt.subplot(2, 3, count)
    plt.imshow(np.abs(np.fft.ifft2(np.fft.ifftshift(sfilt*rotim))))
    count += 1
plt.suptitle('Square filter smoothing')
plt.tight_layout()

'''
Mystery image
'''
viewer(bold_data, view='axial', cmap='gray')