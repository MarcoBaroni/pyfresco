import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.widgets import PolygonSelector
import matplotlib.animation as animation
from matplotlib.backend_bases import MouseButton
from matplotlib.ticker import FuncFormatter

import numpy as np
import pandas as pd

from scipy.stats import median_abs_deviation as MAD
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.signal import savgol_filter, argrelextrema, find_peaks

from kneed import KneeLocator

import torch
import torch.nn.functional as F
from torch import nn

class SpectraExtract():
    """
    Class to extract the spectra by selecting pixels or collections of pixels on a given RGB map.
    
    Parameters
    ----------
    img : python spectral.io.bsqfile.BsqFile object
        Spectral reflectances datacube.
    Nbands : int
        Number of wavelengths used (basically, len(wavelength)).
    wavelength : list or array
        CRISM observed wavelengths.
    MIN : float
        Minimum wavelength value to be taken into consideration.
    MAX : float
        Maximum wavelength value to be taken into consideration.
    """
    def __init__(self , img , Nbands , wavelength , MIN , MAX):
        self.img = img
        self.Nbands = Nbands
        self.w = wavelength
        self.MIN = MIN
        self.MAX = MAX
        
    def upload_map(self , name , folder = None):
        """
        Function to upload a pre-made RGB map, saved using RGBImageManipulator.save_map(), that wants to be used to extract the spectra.
        
        Parameters
        ----------
        name : string
            Name of the RGB map that wants to be uploaded.
        folder : string
            Path of the RGB map that want s to be uploaded. If None path is taken as home directory. Default is None.
            
        Returns
        -------
        RGB : 3-dim array
            Uploaded RGB map.
        """

        if folder == None:
            R = np.loadtxt(name + '_R.txt')
            G = np.loadtxt(name + '_G.txt')
            B = np.loadtxt(name + '_B.txt')
        else:
            R = np.loadtxt(folder + name + '_R.txt')
            G = np.loadtxt(folder + name + '_G.txt')
            B = np.loadtxt(folder + name + '_B.txt')    
            
        image = np.zeros((R.shape[0] , R.shape[1] , 3))

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i,j,0] = R[i,j]
                image[i,j,1] = G[i,j]
                image[i,j,2] = B[i,j]
                
        self.RGB = image

        return self.RGB
        
    def limits(self, other_w = None):
        """
        Function to compute the limits of the x-axis and the corresponding indexes to then compute the y-axis limits of an array given the list/array of the x axis.
            
        Parameters
        ----------
        other_w : list, array or None
            If list or array, this will be taken as the wavelength array to use, if None then the CRISM observation wavelengths are used. Default is None.
        
        Returns
        -------
        extremas : list of float
            list of index correspondant do xmin , index correpsondant to xmax, xmin and xmax
        """
        
        if type(other_w) == list or isinstance(other_w , np.ndarray) == True:
            k = other_w
        else:
            k = self.w
        
        a , b = np.zeros(len(k)) , np.zeros(len(k))
        for i in range(len(k)):
            a[i] = np.abs(k[i]-self.MIN)
            b[i] = np.abs(k[i]-self.MAX)
        xmin_ind , xmax_ind = np.argmin(a) , np.argmin(b)
        xmin , xmax = k[xmin_ind] , k[xmax_ind]

        return [xmin_ind , xmax_ind , xmin , xmax]
    
    def find_nearest(self, array, value):
        """
        Function used to find the index of element nearest to a given arbitrary value.
        
        Parameters
        ----------
        array : array
            Array in which to search
        value : float
            Value to search the nearest element inside array.
            
        Returns
        -------
        idx : int
            Index of the nearest value.
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    def polygon_spectra(self, c_line='white', c_marker='white', save_pixel=False, folder=None, name='spectra_coordinates'):
        """
        Function to select the area from which to extract the spectra by drawing a polygon on the RGB image. The spectra are then extracted from the enclosed pixels from the spectral reflectances datacube.
        
        Parameters
        ----------
        c_line : string
            Color of the polygon's sides. Default is 'white'.
        c_marker : string
            Color of the polygon's corners. Default is 'white'.
        save_pixel : bool
            If to save or not the enclosed pixels coordinates in a .txt file. Default is False.
        fodler : string
            Folder in which to save the pixels if save_pixels is True. If None it saves in the home directory. Default is None.
        name : string
            Name with which the pixels coorinates are saved. Default is 'spectra_coordinates'.
            
        Returns
        -------
        spectra : 2-dim array
            Spectra extracted from the spectral reflectance datacube from the pixels enclosed in the polygon drewn on the RGB map.
        L : int
            Amount of pixels selected.
        mask : 2-dim array
            Masked RGB array to enhance polygon position.
        """
        fig, ax = plt.subplots()
        ax.imshow(self.RGB)
        
        def onselect(poly_verts):
            global poly
            poly = poly_verts

        poly_selector = PolygonSelector(ax , onselect , props = {'color': c_line})
        plt.show()

        path = Path(poly)

        y , x = np.mgrid[:self.RGB.shape[0] , :self.RGB.shape[1]]
        points = np.vstack((x.ravel() , y.ravel())).T
        mask = path.contains_points(points)
        mask = mask.reshape(self.RGB.shape[:2])

        points_inside = self.RGB[mask]
        L = len(points_inside)
        print('Number of points inside the drewn polygon: ' , L)

        indices_1 , indices_2 = np.where(mask)

        spectra = np.zeros( ( L , self.Nbands ) )

        for i in range(len(indices_1)):

            spectra[i] = np.transpose( self.img[indices_1[i] , indices_2[i] , :] )[:,0,0]

        if save_pixel == True:
            if folder == None:
                np.savetxt(name + '.txt' , np.array([indices_2 , indices_1] , dtype = int))
            else:
                np.savetxt(folder + name + '.txt' , np.array([indices_2 , indices_1] , dtype = int))
                
        self.spectra = spectra

        return self.spectra , L , mask
    
    def point_spectra(self, N):
        """
        Function to select pixels from which to extract the spectra by selecting points on the RGB image. The spectra are then extracted from the enclosed pixels from the spectral reflectances datacube.
        
        Parameters
        ----------
        N : int
            Number of points that wants to be taken.
            
        Returns
        -------
        spectra : 2-dim array
            Spectra extracted from the spectral reflectance datacube from the pixels corresponding to the points drewn on the RGB map.
        """
        spectra = np.zeros((N , self.Nbands))
    
        plt.imshow(self.RGB)
        coords = plt.ginput(n=N, timeout=0)
        plt.show()

        for i in range(N):
            x = coords[i][0]
            y = coords[i][1]
            spectra[i] = self.img[int(x) , int(y) , :]
        self.spectra = spectra
        return self.spectra

    def square_spectra(self, N, show = False):
        """
        Function to select the area from which to extract the spectra by drawing a square on the RGB image. The spectra are then extracted from the enclosed pixels from the spectral reflectances datacube.
        The square is drewn by selecting the center of the square and giving to it a side length.
        IMPORTANT: THE GIVEN SQUARE SIDE LENGTH MUST BE AN ODD NUMBER!
        
        Parameters
        ----------
        N : int
            Side length of the square (MUST BE AN ODD NUMBER!).
        show : bool
            Whether or not to show the location of the selected square. Default is False.
            
        Returns
        -------
        spectra : 2-dim array
            Spectra extracted from the spectral reflectance datacube from the pixels corresponding to the points enclosed in the square on the RGB map.
        x : list
            X coordinates of the pixels inside the square.
        y : list
            Y coordinates of the pixels inside the square.
        """
        
        if (N % 2) == 0:
            print('WARNING: THE GIVEN SQUARE SIDE LENGTH MUST BE AN ODD NUMBER!')
            return 0 , 0 , 0
    
        spectra = np.zeros((N*N , self.Nbands))

        fig = plt.figure()
        plt.imshow(self.RGB)

        coords = plt.ginput(n=1, timeout=0 , mouse_add=MouseButton.RIGHT, mouse_pop=None, mouse_stop=None)        # Select the pixels with the cursor

        coords_x , coords_y = int(coords[0][0]) , int(coords[0][1])
        
        plt.show()

        x , y , n = int(coords[0][0]) , int(coords[0][1]) , N//2

        X , Y = [] , []

        for i in range(x-n , x+n+1):
            for j in range(y-n , y+n+1):
                X.append(i)
                Y.append(j)
                
        if show == True:
            fig = plt.figure()
            plt.imshow(self.RGB)
            
        for I in range(len(X)):
            spectra[I,:] = self.img[Y[I],X[I],:]
            
            if show == True:
                if X[I] == x-n or X[I] == x+n or Y[I] == y-n or Y[I] == y+n:
                    edges = 'black'
                else:
                    edges = 'white'
                plt.scatter(X[I] , Y[I] , color = 'white' , edgecolor = edges , marker = 's' , s = 10)
                
        self.spectra = spectra
        
        if show == True:
            plt.show()
        
        return self.spectra , x , y
    
    def plot_spectra(self, N, ylim_min = 0, ylim_max = 0.5):
        """
        Function to plot the spectra of the spectra arrays.
        
        Parameters
        ----------
        
        N : int
            Number of spectra
        ylim_min : float
            Minimum y value from which to plot
        ylim_max : float
            Maximum y value from which to plot
        
        Returns
        -------
        None
        """

        for i in range(N):
            plt.plot(self.w , self.spectra[i])
        plt.ylim(ylim_min , ylim_max)
        plt.xlim(self.MIN , self.MAX)
        plt.xlabel('$\lambda$ [nm]')
        plt.ylabel('Relative reflectance')
        plt.show()
        
    def final_spectra(self, mean = True, size = [5,15], c = 'blue'):
        """
        Function to compute and plot the final version of the spectra (mean +- standard deviation or median +- MAD).
        
        Parameters
        ----------
        mean : bool
            If True it computes and plots the mean and the standard deviation, if False it computes and plots the median and the MAD. Default is True.
        size : list of two float
            Size of the plot.
        c : string
            Color of the plot.

        Returns
        -------
        m_spec : array
            Array of mean or median spectrum.
        err_spec : array
            Array of the spectrum' standard deviation or MAD.
        """

        N = len(self.w)

        ref , err = np.zeros(N) , np.zeros(N)
        if mean == True:
            for i in range(N):
                ref[i] = np.mean(self.spectra[:,i])
                err[i] = np.std(self.spectra[:,i])
        else:
            for i in range(N):
                ref[i] = np.median(self.spectra[:,i])
                err[i] = MAD(self.spectra[:,i])

        a , b = np.zeros(N) , np.zeros(N)
        for i in range(N):
            a[i] = np.abs(self.w[i] - self.MIN)
            b[i] = np.abs(self.w[i] - self.MAX)

        xmin_ind , xmax_ind = np.argmin(a) , np.argmin(b)
        xmin , xmax = self.w[xmin_ind] , self.w[xmax_ind]

        #self.w = self.w[ref < 1]
        #err = err[ref < 1]
        #ref = ref[ref < 1]

        refmin , refmax = ref - err , ref + err

        plt.figure(figsize = size)
        plt.plot(self.w , ref , '-' , color = c , label = 'Mean')
        plt.plot(self.w , refmin , '--' , color = c , label = '$\sigma$' , linewidth = 0.5)
        plt.plot(self.w , refmax , '--' , color = c , linewidth = 0.5)
        plt.fill_between(self.w , refmin , refmax , color = c , alpha = 0.2)
        plt.legend(loc = 'best')
        plt.xlim(self.MIN , self.MAX)
        plt.ylim(np.min(refmin[xmin_ind:xmax_ind]) , np.max(refmax[xmin_ind:xmax_ind]))
        plt.xlabel('$\lambda$ [nm]' , fontsize = 20)
        plt.ylabel('Reflectance' , fontsize = 20)
        plt.show()
        
        self.m_spec = ref
        self.err_spec = err

        return self.m_spec , self.err_spec #, [refmin , refmax]7
    
    def save_spectra(self , name , folder , method = 'polygon'):
        '''
        Function to save the extracted spectra with either method and the cut wavelength range.
        
        Parameters
        ----------
        name : string
            Name of the file.
        folder : string or None
            If None it will be saved into the home folder, if a folder path is given, the path must end with the /.
        method : string
            It add a suffix standing for the used extraction method. Default is polygon.
            
        Returns
        -------
        None
        '''
        i , j = self.find_nearest(self.MIN , self.w) , self.find_nearest(self.MAX , self.w)
        if folder == None:
            np.savetxt('Wavelength_from_'+str(self.w[i])+'_to_'+str(self.w[i:j])+'.txt')
            np.savetxt(name + '_' + method + '.txt' , self.spectra)
        else:
            np.savetxt(folder + 'Wavelength_from_'+str(self.w[i])+'_to_'+str(self.w[i:j])+'.txt')
            np.savetxt(folder + name + '_' + method + '.txt' , self.spectra)
    
    def save_spectrum(self , name , folder , method = 'polygon' , mean = True):
        '''
        Function to save the mean/median and std/MAD spectra and the cut wavelength range.
        
        Parameters
        ----------
        name : string
            name of the file.
        folder : string or None
            If None it will be saved into the home folder, if a folder path is given, the path must end with the /.
        method : string
            It add a suffix standing for the used extraction method. Default is polygon.
        mean : Bool
            if True it will add a suffix with mean/std at the end of the filename, if False it will add median/mad instead.
        
        Returns
        -------
        None
        '''
        i , j = self.find_nearest(self.MIN , self.w) , self.find_nearest(self.MAX , self.w)
        if folder == None:
            np.savetxt('Wavelength_from_'+str(self.w[i])+'_to_'+str(self.w[i:j])+'.txt')
            if mean == True:
                np.savetxt(name + '_' + method +'_mean.txt' , self.m_spec)
                np.savetxt(name + '_' + method +'_std.txt' , self.err_spec)
            else:
                np.savetxt(name + '_' + method +'_median.txt' , self.m_spec)
                np.savetxt(name + '_' + method +'_mad.txt' , self.err_spec)
                
        else:
            np.savetxt(folder + 'Wavelength_from_'+str(self.w[i])+'_to_'+str(self.w[i:j])+'.txt')
            if mean == True:
                np.savetxt(folder + name + '_' + method +'_mean.txt' , self.m_spec)
                np.savetxt(folder + name + '_' + method +'_std.txt' , self.err_spec)
            else:
                np.savetxt(folder + name + '_' + method +'_median.txt' , self.m_spec)
                np.savetxt(folder + name + '_' + method +'_mad.txt' , self.err_spec)
