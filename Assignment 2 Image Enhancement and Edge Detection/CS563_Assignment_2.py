'''
    File name: CS563_Assignment_2.py
    Author: Tianye Zhao
    Test by: Zhaoxuan Qin
    Date created: 2/28/2019
    Date last modified: 3/13/2019
    Python Version: 3.7.2
    IDE: Python IDLE
    Libraries: numpy, opencv, pillow, matplotlib
    ================================
    There may be slight difference between report and program, since this program is
    edited continuously till final submission.
    
    User should follow the exact steps to improve the appearance of images:
    1. image loading;
    2. contrast enhancement by linear/power transformation or histogram equalization;
    3. smoothing with chosen filters;
    4. edge detection through different methods.
    Otherwise, it will report error if previous step is not operated.
'''

#======================
# imports
#======================
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import cv2 as cv
import numpy as np
import PIL.Image, PIL.ImageTk
from matplotlib import pyplot as plt

#======================
# Create class
#======================
class App:

    def __init__(self, master):
    	#======================
	# procedural code
	#======================
        # Add a title 
        master.title('CS563 Assignment 2')
        
	# ---------------------------------------------------------------
        # Creating a Menu Bar
        master.option_add('*tearOff', False)
        menuBar = tk.Menu(master)
        master.config(menu=menuBar)
	#---------------------------------------------
	# Add menu items
        fileMenu = tk.Menu(menuBar)
        
        menuBar.add_cascade(label='File', menu=fileMenu)
        fileMenu.add_command(label='Open', command = self.select_image)
        fileMenu.add_separator()
        fileMenu.add_command(label='Save', command=self.save_image) 
        fileMenu.add_separator()
        fileMenu.add_command(label='Exit', command=lambda: self._quit(master))
        #---------------------------------------------
        # Add tool Menu to the Menu Bar
        toolMenu = tk.Menu(menuBar)
        menuBar.add_cascade(label='Tool', menu=toolMenu)

        # Negate image
        self.negate = tk.IntVar()
        toolMenu.add_checkbutton(label='Negate', variable=self.negate, command=self.inverte)

        # Histogram
        toolMenu.add_separator()
        histMenu = tk.Menu(toolMenu)
        toolMenu.add_cascade(menu=histMenu, label='Histogram')
        histMenu.add_command(label='Original', command=self.show_original_hist)
        histMenu.add_command(label='Adjusted', command=self.show_adjusted_hist)
		
        # Histogram Equalization
        toolMenu.add_separator()
        histEqualMenu = tk.Menu(toolMenu)
        toolMenu.add_cascade(menu=histEqualMenu, label='HistEqual')
        histEqualMenu.add_radiobutton(label='None', command=self.recover_hist)
        histEqualMenu.add_radiobutton(label='HistEqual', command=self.histogram_equalization)
        histEqualMenu.add_radiobutton(label='CLAHE', command=self.Clahe)

        # Find Contours
        toolMenu.add_separator()
        toolMenu.add_command(label='Contours', command=self.draw_contours)
	#---------------------------------------------
        
        # Help
        helpMenu = tk.Menu(menuBar)
        menuBar.add_cascade(label='Help', menu=helpMenu)
        helpMenu.add_command(label='About', command=self.show_info)
	# ---------------------------------------------------------------
		
        # Main Frame
        self.main_frame = ttk.Frame(master)
        self.main_frame.pack()
        #---------------------------------------------
        # initialize Image Frame, this panel will store our original and adjusted image
        self.img_frame = ttk.Frame(self.main_frame)
        self.img_frame.grid(column=0, row=0)
		
        # Create a canvas that can fit the above image
        self.canvas = tk.Canvas(self.img_frame, width=150, height=100)
        self.canvas.pack()
	#---------------------------------------------
        # Edit Frame, this panel stores the choices for user to select
        self.edit_frame = ttk.Frame(self.main_frame)
        self.edit_frame.grid(column=1, row=0)
        # ---------------------------------------------------------------
        
        # Linear Transformation frame
        self.linear_frame = ttk.LabelFrame(self.edit_frame, text='Linear')
        self.linear_frame.grid(row=0, column=0)
        #---------------------------------------------
        self.alpha = tk.DoubleVar()
        self.alpha.set(1.0)
        ttk.Label(self.linear_frame, text='a:').grid(column=0, row=0, stick='e')
        ttk.Spinbox(self.linear_frame, from_=0.1, to=5, increment=0.1, width=5, textvariable=self.alpha,
                    command=lambda: self.linear_transforms(self.alpha.get(),self.beta.get())).grid(row=0,column=1)
        #---------------------------------------------
        self.beta = tk.IntVar()
        self.beta.set(0)
        ttk.Label(self.linear_frame, text='b:').grid(column=0, row=1, stick='e')
        ttk.Spinbox(self.linear_frame, from_=-50, to=50, increment=1, width=5, textvariable=self.beta,
                    command=lambda: self.linear_transforms(self.alpha.get(),self.beta.get())).grid(row=1,column=1)
        
        for child in self.linear_frame.winfo_children():
            child.grid_configure(padx=3, pady=3, stick='w')
        # ---------------------------------------------------------------
        
        # Gamma Correction frame
        self.gamma_frame = ttk.LabelFrame(self.edit_frame, text='Gamma')
        self.gamma_frame.grid(row=0, column=1)
        
        self.gamma = tk.DoubleVar()
        self.gamma.set(1.0)
        ttk.Label(self.gamma_frame, textvariable=self.gamma).pack()
        ttk.Scale(self.gamma_frame, orient = tk.HORIZONTAL, length=100, variable=self.gamma,
                  from_=0.1, to = 5.0, command=lambda x: self.adjust_gamma(self.gamma.get())).pack()
        # ---------------------------------------------------------------

        # Filter Frame
        self.filter_frame = ttk.LabelFrame(self.edit_frame, text='Filter')
        self.filter_frame.grid(row=1, column=0)
	#---------------------------------------------
        self.filter_type = tk.StringVar()
        ttk.Radiobutton(self.filter_frame, text='None', value='None', variable=self.filter_type,
                        command=self.recover).grid(row=0, column=0, columnspan=2)
        #---------------------------------------------
        ttk.Radiobutton(self.filter_frame, text='Average', value='Average', variable=self.filter_type,
                        command=self.average_filter).grid(row=1, column=0, columnspan=2)
        #---------------------------------------------
        ttk.Radiobutton(self.filter_frame, text='Gaussian', value='Gaussian', variable=self.filter_type,
                        command=self.gaussian_filter).grid(row=2, column=0, columnspan=2)
        #---------------------------------------------
        ttk.Radiobutton(self.filter_frame, text='Median', value='Median', variable=self.filter_type,
                        command=self.median_filter).grid(row=3, column=0, columnspan=2)
        #---------------------------------------------
        ttk.Radiobutton(self.filter_frame, text='Bilateral', value='Bilateral', variable=self.filter_type,
                        command=self.bilateral_filter).grid(row=4, column=0, columnspan=2)
        #---------------------------------------------
        self.filterSize = tk.IntVar()
        self.filterSize.set(5)
        ttk.Label(self.filter_frame, text='Size:').grid(column=0, row=5)
        ttk.Spinbox(self.filter_frame, from_=3, to=31, increment=2, width=3,
                    textvariable=self.filterSize).grid(row=5, column=1)
	#---------------------------------------------
        for child in self.filter_frame.winfo_children():
            child.grid_configure(padx=3, stick='w')
        # ---------------------------------------------------------------

        # Edge Detect Frame
        self.edge_frame = ttk.LabelFrame(self.edit_frame, text='Edge')
        self.edge_frame.grid(row=1, column=1)
        #---------------------------------------------
        self.edge_detector = tk.StringVar()
        self.sobelSize = tk.IntVar()
        self.sobelSize.set(5)
        ttk.Radiobutton(self.edge_frame, text='Sobel', value='Sobel', variable=self.edge_detector,
                        command=lambda: self.sobel_edge(self.sobelSize.get())).grid(row=0, column=0)
        ttk.Spinbox(self.edge_frame, from_=5, to=15, increment=2, width=3,
                    textvariable=self.sobelSize).grid(row=0, column=1)
        #---------------------------------------------
        ttk.Radiobutton(self.edge_frame, text='Scharr', value='Scharr', variable=self.edge_detector,
                        command=self.scharr_edge).grid(row=1, column=0, columnspan=2)
        #---------------------------------------------
        ttk.Radiobutton(self.edge_frame, text='Laplacian', value='Laplacian', variable=self.edge_detector,
                        command=self.laplacian_edge).grid(row=2, column=0, columnspan=2)
        #---------------------------------------------
        ttk.Radiobutton(self.edge_frame, text='Prewitt', value='Prewitt', variable=self.edge_detector,
                        command=self.prewitt_edge).grid(row=3, column=0, columnspan=2)
        #---------------------------------------------
        ttk.Radiobutton(self.edge_frame, text='Canny', value='Canny', variable=self.edge_detector,
                        command=self.canny_edge).grid(row=4, column=0, columnspan=2)
		#---------------------------------------------
        for child in self.edge_frame.winfo_children():
            child.grid_configure(padx=3, stick='w')

        # ---------------------------------------------------------------
        # Thresholding Frame
        self.thres_frame = ttk.LabelFrame(self.edit_frame, text='Threshold')
        self.thres_frame.grid(row=0, column=2, rowspan=2)
        #---------------------------------------------
        self.thres_type = tk.StringVar()
        ttk.Radiobutton(self.thres_frame, text='None', value='None', variable=self.thres_type,
                        command=self.no_thresh).grid(row=0, column=0, columnspan=2)
        #---------------------------------------------
        ttk.Radiobutton(self.thres_frame, text='Otsu', value='Otsu', variable=self.thres_type,
                        command=self.otsu_thresh).grid(row=1, column=0, columnspan=2)
        #---------------------------------------------
        ttk.Radiobutton(self.thres_frame, text='Mean', value='Mean', variable=self.thres_type,
                        command=lambda: self.adapt_mean(self.block_size.get(),self.c.get())).grid(row=2, column=0, columnspan=2)
        #---------------------------------------------
        ttk.Radiobutton(self.thres_frame, text='Gaussian', value='Gaussian', variable=self.thres_type,
                        command=lambda: self.adapt_gaussian(self.block_size.get(),self.c.get())).grid(row=3, column=0, columnspan=2)
        #---------------------------------------------
        self.block_size = tk.IntVar()
        self.block_size.set(5)
        ttk.Label(self.thres_frame, text='Size:').grid(row=4, column=0)
        ttk.Spinbox(self.thres_frame, from_=5, to=31, increment=2, width=3,
                    textvariable=self.block_size).grid(row=4, column=1)
        #---------------------------------------------
        self.c = tk.IntVar()
        self.c.set(1)
        ttk.Label(self.thres_frame, text='c:').grid(row=5, column=0)
        ttk.Spinbox(self.thres_frame, from_=1, to=5, increment=1, width=3,
                    textvariable=self.c).grid(row=5, column=1)
        #---------------------------------------------
        
        for child in self.thres_frame.winfo_children():
            child.grid_configure(padx=3, stick='w')
        # ---------------------------------------------------------------
        for child in self.edit_frame.winfo_children():
            child.grid_configure(padx=5, pady=5)
        # ---------------------------------------------------------------
    
    #======================
    # functions
    #======================
    # Exit GUI cleanly
    def _quit(self, win):
        win.quit()      # win will exit when this function is called
        win.destroy()
        exit()
    
    # Load an image using OpenCV
    def select_image(self):
     
        # open a file chooser dialog and allow the user to select an input image
        self.image_path = filedialog.askopenfilename(initialdir='',title='Choose an image',
                                          filetypes=(('PGM','*.pgm'),('PBM','*.pbm'),('PPM','*.ppm')))

        # ensure a file path was selected
        if len(self.image_path) > 0:
            self.original = cv.imread(self.image_path, 0)
            # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
            self.height, self.width = self.original.shape
            
            # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage,
            # convert the images to PIL format and then to ImageTk format
            self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.original))
        
            # Add a PhotoImage to the Canvas
            self.canvas.config(width=self.width, height=self.height)
            self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
    # ---------------------------------------------------------------

    def save_image(self):
     
        # open a file chooser dialog and allow the user to select an input image
        self.filename = filedialog.asksaveasfilename(initialdir='',title='Select an image',
                                          filetypes=(('JPEG','*.jpg;*.jpeg'),
                                                     ('GIF','*.gif'),
                                                     ('PNG','*.png')))
        cv.imwrite(self.filename, self.blur)
        
    def inverte(self):

        self.original = cv.bitwise_not(self.original)
        self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.original))
        
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
    # ---------------------------------------------------------------

    def draw_contours(self):
        plt.clf()
        contours, _ = cv.findContours(self.binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        self.detect = np.ones((self.height, self.width), 'uint8')
        self.detect *= (2**8-1)

        cv.drawContours(self.detect, contours, -1, 0, 1)
        plt.imshow(self.detect, cmap='gray')
        plt.title('Contours Image'), plt.xticks([]), plt.yticks([])
        plt.show()
        
    # ---------------------------------------------------------------

    def otsu_thresh(self):
        _, self.binary = cv.threshold(self.blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.binary))
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

    def no_thresh(self):
        self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.blur))
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

    def adapt_mean(self, size, c):
        self.binary = cv.adaptiveThreshold(self.blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, size, c)
        self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.binary))
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

    def adapt_gaussian(self, size, c):
        self.binary = cv.adaptiveThreshold(self.blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV, size, c)
        self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.binary))
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
    # ---------------------------------------------------------------
            
    def linear_transforms(self, alpha, beta):
                
        self.adjusted = np.zeros(self.original.shape, self.original.dtype)
        for y in range(self.original.shape[0]):
            for x in range(self.original.shape[1]):
                self.adjusted[y,x] = np.clip(alpha*self.original[y,x]+beta, 0, 255)
            
        self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.adjusted))
        
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
    # ---------------------------------------------------------------
    
    # Callback for the "Gamma Correct"
    def adjust_gamma(self, gamma):
    	# build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
        self.gamma.set('%0.2f' % self.gamma.get()) 
        self.table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype('uint8')
     
        # apply gamma correction using the lookup table
        self.adjusted = cv.LUT(self.original, self.table)
        self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.adjusted))
        # update the image panel
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
    # ---------------------------------------------------------------

    def laplacian_edge(self):
        plt.clf()
        self.edges = cv.Laplacian(self.blur, cv.CV_64F)
        plt.imshow(self.edges, cmap='gray')
        plt.title('Laplacian Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    def sobel_edge(self, kernelSize):
        plt.clf()
        self.sobelx = cv.Sobel(self.blur, cv.CV_64F, 1, 0, ksize=kernelSize)
        self.sobely = cv.Sobel(self.blur, cv.CV_64F, 0, 1, ksize=kernelSize)
        self.sobel = self.sobelx + self.sobely
        plt.subplot(131), plt.imshow(self.sobelx, cmap='gray')
        plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
        plt.subplot(132), plt.imshow(self.sobely, cmap='gray')
        plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
        plt.subplot(133), plt.imshow(self.sobel, cmap='gray')
        plt.title('Sobel'), plt.xticks([]), plt.yticks([])
        plt.show()

    def scharr_edge(self):
        plt.clf()
        self.scharrx = cv.Scharr(self.blur, cv.CV_64F, 1, 0)
        self.scharry = cv.Scharr(self.blur, cv.CV_64F, 0, 1)
        self.scharr = self.scharrx + self.scharry
        plt.subplot(131), plt.imshow(self.scharrx, cmap='gray')
        plt.title('Scharr X'), plt.xticks([]), plt.yticks([])
        plt.subplot(132), plt.imshow(self.scharry, cmap='gray')
        plt.title('Scharr Y'), plt.xticks([]), plt.yticks([])
        plt.subplot(133), plt.imshow(self.scharr, cmap='gray')
        plt.title('Scharr'), plt.xticks([]), plt.yticks([])
        plt.show()

    def prewitt_edge(self):
        plt.clf()
        self.kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        self.kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        self.prewittx = cv.filter2D(self.blur, -1, self.kernelx)
        self.prewitty = cv.filter2D(self.blur, -1, self.kernely)
        self.prewitt = self.prewittx + self.prewitty
        plt.subplot(131), plt.imshow(self.prewittx, cmap='gray')
        plt.title('Prewitt X'), plt.xticks([]), plt.yticks([])
        plt.subplot(132), plt.imshow(self.prewitty, cmap='gray')
        plt.title('Prewitt Y'), plt.xticks([]), plt.yticks([])
        plt.subplot(133), plt.imshow(self.prewitt, cmap='gray')
        plt.title('Prewitt'), plt.xticks([]), plt.yticks([])
        plt.show()
        
    def canny_edge(self):
        plt.clf()
        retVal, _ = cv.threshold(self.blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        self.edges = cv.Canny(self.blur, int(0.33*retVal), retVal)
        plt.imshow(self.edges, cmap='gray')
        plt.title('Canny Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()
    # ---------------------------------------------------------------
        
    def show_original_hist(self):
        plt.clf()
        self.hist = cv.calcHist([self.original], [0], None, [256], [0,256])
        plt.plot(self.hist)
        plt.title('Original Image')
        plt.xlim([0,256])
        plt.show()

    def show_adjusted_hist(self):
        plt.clf()
        self.hist = cv.calcHist([self.adjusted], [0], None, [256], [0,256])
        plt.plot(self.hist)
        plt.title('Adjusted Image')
        plt.xlim([0,256])
        plt.show()
    # ---------------------------------------------------------------

    def show_info(self):
        tk.messagebox.showinfo(title='About',
                           message='CS563 Assignment 2\nAuthor: Tianye Zhao')
    # ---------------------------------------------------------------

    def recover_hist(self):
        self.adjusted = self.original
        self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.adjusted))
        
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

    def histogram_equalization(self):
        self.adjusted = cv.equalizeHist(self.original)
        self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.adjusted))
        
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

    def Clahe(self):
        self.clahe = cv.createCLAHE()
        self.adjusted = self.clahe.apply(self.original)
        self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.adjusted))
        
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
    # ---------------------------------------------------------------
	
	# Functions that let user blur the image
    def recover(self):
        self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.adjusted))
        
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
        
    def gaussian_filter(self):

        self.blur = cv.GaussianBlur(self.adjusted, (self.filterSize.get(),self.filterSize.get()), 0)
        self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.blur))
        
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

    def average_filter(self):

        self.blur = cv.blur(self.adjusted, (self.filterSize.get(),self.filterSize.get()))
        self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.blur))
        
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

    def median_filter(self):

        self.blur = cv.medianBlur(self.adjusted, self.filterSize.get())
        self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.blur))
        
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

    def bilateral_filter(self):

        self.blur = cv.bilateralFilter(self.adjusted, self.filterSize.get(), self.filterSize.get()*2, self.filterSize.get()/2)
        self.photo = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(self.blur))
        
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
    # ---------------------------------------------------------------


def main():
    # Create a window and pass it to the Application object
    # Create instance
    root = tk.Tk()
    app = App(root)
    #======================
    # Start GUI
    #======================
    root.mainloop()

if __name__ == '__main__':
    main()

