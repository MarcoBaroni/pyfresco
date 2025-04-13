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
class SpectraNorm():
    """
    Class to Normalize the mean target spectrum.
    
    Parameters
    ----------
    RGB : 3-dim array or None
        The RGB image to be used. Inside the class it also possible to upload the RGB map, so it is possible to set this parameter as None.
    Nbands : int
        Number of wavelengths used (basically, len(wavelength)).
    img : python spectral.io.bsqfile.BsqFile object
        Spectral reflectances datacube.
    img_sr : python spectral.io.bsqfile.BsqFile object
        Spectral parameters datacube.
    wavelength : list or array
        CRISM observed wavelengths.
    target : array
        The mean target spectrum.
    error_target : array
        The standard deviation of the target spectrum.
    MIN : float
        Minimum wavelength value to be taken into consideration.
    MAX : float
        Maximum wavelength value to be taken into consideration.
    """
    def __init__(self , RGB , Nbands , img , img_sr , wavelength , target , error_target , spectra , MIN , MAX):
        self.RGB = RGB
        self.Nbands = Nbands
        self.img = img
        self.img_sr = img_sr
        self.w = wavelength
        self.target = target
        self.error_target = error_target
        self.spectra = spectra
        self.MIN = MIN
        self.MAX = MAX

    def upload_spectrum_target(self , name , folder = None , mean = True):
            """
            Function to upload a set of pre-extracted spectra, saved using SpectraExtarct.save_spectra().
            
            Parameters
            ----------
            name : string
                Name of the file that wants to be uploaded, without the format extension.
            folder : string
                Path of the file that want s to be uploaded. If None path is taken as home directory. Default is None.
                
            Returns
            -------
            spectra : 2-dim array
                Uploaded pre-extracted spectra.
            target_spectrum : 1-dim array
                Mean/median of the pre-extracted spectra.
            error_spectrum : 1-dim array
                Standard deviation/median absolute deviation of the pre-extracted spectra.
            """
            if folder == None:
                data = np.genfromtxt(name + '.txt' , dtype = int)
            else:
                data = np.genfromtxt(folder + name + '.txt' , dtype = int)
    
            x , y = data[0,:] , data[1,:]
    
            #a , b = self.find_nearest(self.w , self.MIN) , self.find_nearest(self.w , self.MAX)
    
            spectra = np.array([self.img[xi, yi, :] for xi, yi in zip(x, y)])[:,0,0,:]#np.asarray(self.img)[x , y , a:b]
            print(spectra.shape)
        
            plt.imshow(self.RGB)
            plt.plot(x,y, 'w.')
            plt.show()
    
            if mean == True:
                target_spectrum = np.mean(spectra.T , axis = 1)
                error_spectrum = np.std(spectra.T , axis = 1)
            else:
                target_spectrum = np.median(spectra.T , axis = 1)
                error_spectrum = MAD(spectra.T , axis = 1)
    
            self.target , self.error_target , self.spectra = target_spectrum , error_spectrum , spectra
    
            return self.spectra.T , self.target , self.error_target
        
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
                
        RGB = image

        return RGB
    
    def upload_neutral(self , name , method , folder):
        """
        Function to upload a pre-made median/mad spectrum.
        
        Parameters
        ----------
        name : string
            Name of the file.
        method : string
            It will add a suffix after the name on the base on the method with which the neutral spectra was computed.
        folder : string or None
            If None it will be saved into the home folder, if a folder path is given, the path must end with the /.
        
        Returns
        -------
        med : 1d-array
            The uploaded median spectrum.
        mad : 1d-array
            The MAD of the uplkoaded median spectrum.
        w : 1d-array
            The cut wavelength range.
        """
        
        i , j = self.find_nearest(self.MIN , self.w) , self.find_nearest(self.MAX , self.w)

        if folder == None:
            self.w = np.genfromtxt('Wavelength_from_'+str(self.w[i])+'_to_'+str(self.w[i:j])+'.txt')
            self.med = np.genfromtxt(name + '_' + method +'_median.txt')
            self.mad = np.genfromtxt(name + '_' + method +'_mad.txt')
        else:
            self.w = np.genfromtxt(folder + 'Wavelength_from_'+str(self.w[i])+'_to_'+str(self.w[i:j])+'.txt')
            self.med = np.genfromtxt(folder + name + '_' + method +'_median.txt')
            self.mad = np.genfromtxt(folder + name + '_' + method +'_mad.txt')
        
        return self.med , self.mad , self.w

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

    def neutral_polygon_spectra(self, c_line='white', c_marker='white', save_pixel=False, folder=None, name='neutral_spectra_coordinates'):
        """
        Function to select the area from which to extract the neutral spectra by drawing a polygon on the RGB image. The spectra are then extracted from the enclosed pixels from the spectral reflectances datacube.
        This function is essentially the same as SpectraExtract.polygon_spectra().
        
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
        med : 1-dim array
            Median of the extracted spectra.
        mad : 1-dim array
            MAD of the extracted spectra.
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

        self.neutral = spectra

        self.med = np.median(spectra , axis = 0)
        self.mad = MAD(spectra , axis = 0)

        self.neutral_spectra = spectra

        return self.neutral , self.med , self.mad , L , mask
    
    def neutral_convex_hull(self , interp = 'linear'):
        """
        Function to select as neutral spectrum the convex hull (the calculation of is mutuated from scipy https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html).
        
        Parameters
        ---------
        interp : string {‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’}
            Is the interpolation of the convex hull using interp1d from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html).
            The possible arguments signify: ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of zeroth, first, second or third order; ‘previous’ and ‘next’ simply return the previous or next value of the point; ‘nearest-up’ and ‘nearest’ differ when interpolating half-integers (e.g. 0.5, 1.5) in that ‘nearest-up’ rounds up and ‘nearest’ rounds down. Default is ‘linear’.
        
        Returns
        -------
        neutral : 1-dim array
            Points of the convex hull.
        med : 1-dim array
            Median of the convex hull, in this case is the same as neutral, but for completeness we keeped the same strucutre as the others methods.
        mad : 1-dim array
            MAD of the convex hull, since it only one "spectrum" it is an array filled with zeros.
        """

        I , J = self.find_nearest(self.w , self.MIN) , self.find_nearest(self.w , self.MAX)
        x , y = self.w[I:J] , self.target[I:J]
        points = np.c_[x , y]
        augmented = np.concatenate([points, [(x[0], np.min(y)-1), (x[-1], np.min(y)-1)]], axis=0)
        hull = ConvexHull(augmented , incremental = True)
        continuum_points = points[np.sort([v for v in hull.vertices if v < len(points)])]
        continuum_indexs = np.array(range(0,len(continuum_points) , 1) , dtype = int)

        continuum_function = interp1d(*continuum_points.T , kind = interp)

        self.neutral = continuum_function(x)
        
        self.med = self.neutral
        self.mad = np.zeros(len(self.neutral))

        return self.neutral , self.med , self.mad

    def neutral_all_map_multiple(self , RGBs_names , RGB_path , p , threshold , names):
        """
        Function to obtain the neutral spectra with the so called 'all map' method consisting into creating an array of zeros and ones for each RGB image
        where the ones represents the points for that RGB with the three channel at 0 (+ a threshold), then this arrays are summed together and the neutral spectra
        is obtained by averaging all the spectra that corresponds to the pixels in which we have ones in all the maps (e.g if we have three maps, the final map will have
        only integer values of 0, 1, 2 or 3. We will take only the spectra corresponding to pixels of value 3 (or greater than a specified integer value) and we will do the average between them).
        
        Parameters
        ----------
        RGBs_names : list of strings
            The names of the RGB maps used.
        RGB_path : string
            Path to the RGB maps used.
        p : int
            Minimum number of maps or the number of image with zeros in common. 
            IMPORTANT: CANNOT EXCEED THE NUMBER OF RGB MAPS GIVEN.
        Threshold : float
            Threshold value to use if not so many zero pixels are avaliable in some maps.
        names : list of strings
            Code names of the RGB maps for the plot of zero valued pixels per map.
        
        Returns
        -------
        neutral : 2-dim array
            Array containing the neutral spectra.
        med : 1-dim array
            Median neutral spectrum.
        mad : 1-dim array
            MAD of the neutral spectrum.
        superimposed : 2-dim array
            Array of the superimposed zero valued pixels per map. In this map 0 means no zeros in common on that pixel, 1 means that 1 map have a zero there, 2 means that two maps have a zero in common, etc.
        zero_map : 3-dim array
            Array of the zero pixels per map.
        xx : list
            X coordinates of the pixels containing the neutral spectra.
        yy : list
            Y coordinates of the pixels containing the neutral spectra.
        """

        RGBs = [] 
        for i in range(len(RGBs_names)):
            RGBs.append(self.upload_map(RGBs_names[i] , folder = RGB_path))

        # Extracting dimensions
        X , Y = self.img.shape[0] , self.img.shape[1]

        # Initializing the arrays, zero_map is the array of ones and zeros, xx and yy are lists in which the lists of indexes of zero RGB pixels will be put
        zero_map , xx , yy = np.zeros((X , Y , len(RGBs))) , [] , []

        # Initializing the axis given a certain number of figures
        if len(RGBs) % 2 != 0:
            fig , ax = plt.subplots( 2 , int(len(RGBs)/2 + 0.5) )
        else:
            fig , ax = plt.subplots( 2 , int(len(RGBs)/2) )

        # Double cycle that finds the zeros

        # Initialize parametric values. k is the map index (searching the k-th RGB) and l is an adjoint index used in plotting
        k , l = 0 , 0
        for rgb in RGBs:
            # Initializing empty lists in which tht eindexes of zeros will be inserted
            x , y = [] , []
            for i in range(X):
                for j in range(Y):

                    # Check to not search for NaN values (i.e. the borders)
                    if rgb[i,j,0] != np.nan and rgb[i,j,1] != np.nan and rgb[i,j,2] != np.nan :
                        # Main condition, setting to 1 the zero_map values thata correspond to the searched pixels and appending the indexes
                        if rgb[i,j,0] <= 0. + threshold and rgb[i,j,1] <= 0. + threshold and rgb[i,j,2] <= 0. + threshold :

                            zero_map[i,j,k] = 1
                            x.append(i)
                            y.append(j)
                            
            if len(RGBs) == 2:
                ax[k,0].imshow(zero_map[:,:,k] , cmap = 'Greys_r')
                ax[k,0].axis('off')
                ax[k,0].set_title(names[k])
            else:
                            
                # if/else choice to make the plot loofing good 
                if k < len(RGBs)/2:
                    ax[0,k].imshow(zero_map[:,:,k] , cmap = 'Greys_r')
                    ax[0,k].axis('off')
                    ax[0,k].set_title(names[k])
                else:
                    ax[1,l].imshow(zero_map[:,:,k] , cmap = 'Greys_r')
                    ax[1,l].axis('off')
                    ax[1,l].set_title(names[k])
                    l += 1
                
            # Appending lists of indexes
            xx.append(x)
            yy.append(y)
            k += 1
            
        # If number of axis is odd, delete the last figure (is empty)
        if len(RGBs) % 2 != 0:
            fig.delaxes(ax[1,int(len(RGBs)/2-0.5)])
        # Showing the zero maps
        plt.tight_layout()
        plt.show()

        # Initializing array of the final map
        all_maps = np.zeros((X , Y))

        # Initializing plot
        fig , ax = plt.subplots( 1 , 2 )

        # summing slices of the zero_map into one 2-dim array
        for i in range(len(RGBs)):
            all_maps[:,:] += zero_map[:,:,i]

        # Showing the allmap
        im=ax[0].imshow(all_maps , 'Greys_r')
        ax[0].axis('off')
        ax[0].set_title('Amount of 0 valued pixels per map.')

        superimposed , spec = np.zeros((X , Y)) , []

        # Filtering the allmap showing only the pixels with value equal to the number of RGB images. Those pixels are the pixels with zero value in all RGB maps.
        # In this cycle the spectra are extracted and appended in a empty list.
        for i in range(X):
            for j in range(Y):
                if all_maps[i,j] >= p:
                    superimposed[i,j] = 1
                    spec.append(img[i,j,:])
        
        ax[1].imshow(superimposed , 'Greys_r')
        plt.colorbar(im , ticks = np.arange(0,len(RGBs)+1 , dtype = int))
        ax[1].axis('off')
        ax[1].set_title('Null pixels for each map.')

        plt.show()

        self.neutral = np.array(spec)[:,0,0,:]

        self.med = np.median(spec , axis = 0)[0,0,:]
        self.mad = MAD(spec , axis = 0)[0,0,:]
        
        return self.neutral , self.med , self.mad , superimposed , zero_map , xx , yy

    def neutral_all_map_single(self , threshold):
        """
        Simpler version of SpectraNorm.neutral_all_map_multiple() that takes into account the zero-valued pixels of only one RGB map.
        In this case the used RGB map is the one given in the main or the one uploaded using SpectraNorm.upload_map().
        
        Parameters
        ----------
        Threshold : float
            Threshold value to use if not so many zero pixels are avaliable in some maps.
        
        Returns
        -------
        neutral : 2-dim array
            Array containing the neutral spectra.
        med : 1-dim array
            Median neutral spectrum.
        mad : 1-dim array
            MAD of the neutral spectrum.
        x : list
            X coordinates of the pixels containing the neutral spectra.
        y : list
            Y coordinates of the pixels containing the neutral spectra.
        """

        # Extracting dimensions
        X , Y = self.img.shape[0] , self.img.shape[1]

        # Initializing the arrays, zero_map is the array of ones and zeros, xx and yy are lists in which the lists of indexes of zero RGB pixels will be put
        zero_map , xx , yy = np.zeros( (X , Y) ) , [] , []

        # Double cycle that finds the zeros
        plt.figure()
        x , y = [] , []
        for i in range(X):
            for j in range(Y):

                # Check to not search for NaN values (i.e. the borders)
                if self.RGB[i,j,0] != np.nan and self.RGB[i,j,1] != np.nan and self.RGB[i,j,2] != np.nan :
                    # Main condition, setting to 1 the zero_map values thata correspond to the searched pixels and appending the indexes
                    if self.RGB[i,j,0] <= 0. + threshold and self.RGB[i,j,1] <= 0. + threshold and self.RGB[i,j,2] <= 0. + threshold :

                        zero_map[i,j] = 1
                        x.append(i)
                        y.append(j)

        plt.imshow(zero_map , cmap = 'Greys_r')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        self.neutral = np.array(img[:,:,:])[x , y , :]

        self.med = np.median(self.neutral , axis = 0)
        self.mad = MAD(self.neutral , axis = 0)

        return self.neutral , self.med , self.mad , x , y

    def mineral_mask( self , b, band_n , band_v):
        r"""
        Function to obtain the neutral spectra from the mineral mask method used and defined by Horgan et al., 2020 
        (https://www.sciencedirect.com/science/article/pii/S0019103518306067).

        It works by selecting a set of spectral parameters and selecting pixels for the neutral spectrum by imposing threshold values on the parameters themselves with the following rules:

        .. math::

            \begin{aligned}
                &\text{If reflectance:} \quad 
                \begin{cases}
                    R... \geq \text{threshold} &\Rightarrow \text{Pixel in mineral mask} \\
                    R... \leq \text{threshold} &\Rightarrow \text{Pixel outside the mineral mask}
                \end{cases} \\
                &\text{If band depth (BD):} \quad 
                \begin{cases}
                    BD... \leq \text{threshold} &\Rightarrow \text{Pixel in mineral mask} \\
                    BD... \geq \text{threshold} &\Rightarrow \text{Pixel outside the mineral mask}
                \end{cases}
            \end{aligned}

        Where `R...` are the pure reflectance parameters, and `BD...` symbolizes all other spectral parameter types.

        Parameters
        ----------
        b : int
            Number of bins to divide the histograms into.

        band_n : list of strings
            Names of bands to use in the mineral mask. Accepted names are those given in Viviano-Beck et al., 2014.

        band_v : list or array of float
            Threshold values for each band.

        Returns
        -------
        neutral : 2D array
            Array containing the neutral spectra.

        med : 1D array
            Median neutral spectrum.

        mad : 1D array
            MAD of the neutral spectrum.
        """

        df_sr = pd.DataFrame(self.img_sr[:,:,:].reshape(-1,self.img_sr.shape[-1]),
                    columns=self.img_sr.metadata['band names']
                    )

        headers = list(df_sr.columns.values)

        for i in range(df_sr.shape[1]):
            df_sr.loc[df_sr[headers[i]] > 10 , headers[i]] = np.nan#pd.DataFrame(np.where(df_sr > 10., np.nan, df_sr))

        # tuple with spectral parameters names and their threshold value, this is just a nice way to display the selected parameters and their values
        # should also be easy to change values based on the histograms (see below)

        constraints = []

        for k in range(len(band_n)):
            constraints.append( ( band_n[k] , band_v[k] ) )

        band_names_mask = [i for i,j in constraints]  # retrieving band names
        constr_values = [j for i,j in constraints]    # retrieving constrain values

        # plot histograms

        histograms = df_sr[band_names_mask].hist(bins=b, grid=False, color = 'mediumpurple')

        for i, h in enumerate(histograms.ravel()):
            h.title.set_size(5)
            h.axvline(constr_values[i], color='k', linestyle='--')

        # mineral mask

        # small function used just below to easily get the band indexes
        def get_band_index(img,x):
            return img.metadata['band names'].index(x)

        # applying the mineral mask on the img_sr cube
        constraints_ls = []
        for b,v in constraints:
            if b == 'R770':
                constraints_ls.append((self.img_sr[:,:,get_band_index(img_sr,b)] > v).squeeze())
            else:
                constraints_ls.append((self.img_sr[:,:,get_band_index(img_sr,b)] < v).squeeze())

        constraints_mask = np.array(constraints_ls).sum(axis=0) == len(constraints)
        print(f'{constraints_mask.size=}\n{constraints_mask.sum()=}')

        # mineral mask's selected pixel plot, this plot shows in yellow the pixels selected by the mineral mask

        plt.figure()
        plt.imshow(constraints_mask , interpolation='None' , cmap ='Greys_r')
        plt.show()

        # applying the mineral mask to the img (reflectance) cube
        img_masked = np.copy(self.img[:,:,:])

        img_masked=img_masked[constraints_mask==True,:] 

        self.neutral = img_masked

        self.med = np.median(img_masked , axis = 0)
        self.mad = MAD(img_masked , axis = 0)

        return self.neutral , self.med , self.mad

    def plot_together(self , convex_hull = False):
        """
        Function to plot together the mean target spectrum +- its standard deviation and the median neutral spectrum +- its MAD.
        
        Parameters
        ----------
        convex_hull : bool
            If the convex hull neutral spectrum is used. This is done since the convex hull spectrum is drewn directly onto the x-cut spectrum and so does not need to be cut in the same range as the target. Default is False.
        
        Returns
        -------
        None
        """

        lims = self.limits()
        a , b = lims[0] , lims[1]

        wav = self.w[a:b]
        
        if convex_hull == False:
            neutral , neutralerr = self.med[a:b] , self.mad[a:b]
        else:
            neutral , neutralerr = self.med , self.mad

        plt.figure()
        plt.plot(wav , neutral , 'k-' , label = 'Median Neutral')
        plt.plot(wav , neutral + neutralerr , 'k--')
        plt.plot(wav , neutral - neutralerr , 'k--')
        plt.fill_between(wav , neutral - neutralerr , neutral + neutralerr , color = 'k' , alpha = 0.2)

        plt.plot(wav , self.target[a:b] , 'b-' , label = 'Mean Target')
        plt.plot(wav , self.target[a:b] + self.error_target[a:b] , 'b--')
        plt.plot(wav , self.target[a:b] - self.error_target[a:b] , 'b--')
        plt.fill_between(wav , self.target[a:b] - self.error_target[a:b] , self.target[a:b] + self.error_target[a:b] , color = 'b' , alpha = 0.2)

        plt.xlabel ('$\lambda$ [nm]')

        plt.show()

    def norm_spectra(self , convex_hull = False):
        r"""
        Normalization of the target spectrum over the neutral. 
        The error propagation formula is used for the resulting final error, 
        and thus can lead to some places having complex error due to the presence 
        of the covariance between the spectra. More advanced methods should 
        be used to evaluate the error in those cases, but for most application it is sufficently good like this.
        Anyway, errors that ends up as complex are set to zero for simplicity.

        Calling :math:`N` the normalized spectrum, :math:`\sigma_N` the resulting error of the normalized spectrum,  
        :math:`A` the target mean spectrum, :math:`B` the median neutral spectrum,  
        :math:`\sigma_A` the standard deviation of the target, :math:`\sigma_B` the MAD of the neutral,  
        and :math:`C_{A,B}` the covariance between the two spectra, the normalization is done in the following way:


        .. math::

            \begin{aligned}
                N &= \frac{A}{B} \\
                \sigma_{N} &= \left| \frac{A}{B} \right| \sqrt{ 
                \left(\frac{\sigma_{A}}{B}\right)^{2} + 
                \left(\frac{\sigma_{B}}{B}\right)^{2} - 
                \frac{2C_{A,B}}{A \cdot B} }
            \end{aligned}
        
        Parameters
        ----------
        convex_hull : bool
            If the normalization is done with the convex hull or not. Default is False.
            
        Returns
        -------
        norm : array
            Normalized spectrum.
        error norm : array
            Propagated error of the normalized error.
        """
        lims = self.limits()
        
        if convex_hull == False:

            A , B , dA , dB = self.target[lims[0]:lims[1]] , self.med[lims[0]:lims[1]] , self.error_target[lims[0]:lims[1]] , self.mad[lims[0]:lims[1]]
        
        else:
            
            A , B , dA , dB = self.target[lims[0]:lims[1]] , self.med , self.error_target[lims[0]:lims[1]] , self.mad
        
        mean , err = np.zeros(len(A)) , np.zeros(len(B))

        C = 0.

        for i in range(len(A)):
            C += ( A[i] - np.mean(A) )*( B[i] - np.mean(B) )

        C = C/len(A)

        k = 0

        for i in range(len(A)):
            mean[i] = A[i]/B[i]
            t0 , t1 , t2 , t3 = np.abs(A[i]/B[i]) , (dA[i]/A[i])**2 , (dB[i]/B[i])**2 , 2*C/(A[i]*B[i])
            if  t1 + t2 - t3 >= 0:
                err[i] = t0*np.sqrt( t1 + t2 - t3 )
            else:
                k += 1

        print(k/len(A)*100 , '% of the errors, set to zero for simplicity, are in reality NaN values.')

        self.norm = mean
        self.normerr = err

        return self.norm , self.normerr
    
    def normplot(self , convex_hull = False):
        """
        Function to plot the normalized spectrum.

        Parameters
        ---------
        None
        
        Returns
        -------
        None
        """
        
        lims = self.limits()
        
        a , b = lims[0] , lims[1]
        
        wav = self.w[a:b]
        
        plt.plot(wav , self.norm , 'k-')
        
        if convex_hull == False:
            plt.plot(wav , self.norm+self.normerr , 'k--')
            plt.plot(wav , self.norm-self.normerr , 'k--')
            plt.fill_between(wav , self.norm-self.normerr , self.norm+self.normerr , color = 'black' , alpha = 0.5)
        
        plt.xlabel('$\lambda$[nm]')
        plt.show()
        
    def save_spectrum(self , name , folder , method , normalized = True):
        """
        Function to save the mean/median and std/MAD spectra and the cut wavelength range.
        
        Parameters
        ----------
        name : string
            Name of the file.
        folder : string or None
            If None it will be saved into the home folder, if a folder path is given, the path must end with the /.
        method : string
            It will add a suffix after the name on the base on the method with which the neutral spectra was computed.
            If method is not ply , sam , mam , min or cxh, this function will not work.
            ply stands for polygon, sam for single allmap, mam for mutiple allmap, min for mineral mask and csh for convex hull.
        normalized : bool
            If True it will also save the normalize spectrum, if False not.

        Returns
        -------
        None
        """
        if method != 'ply' or method != 'sam' or method != 'mam' or method != 'min' or method != 'cxh':
            raise ValueError('method parameter must be one between ply, sam, mam , min or cxh!')
        
        i , j = self.find_nearest(self.MIN , self.w) , self.find_nearest(self.MAX , self.w)
        if folder == None:
            np.savetxt('Wavelength_from_'+str(self.w[i])+'_to_'+str(self.w[i:j])+'.txt')
            np.savetxt(name + '_' + method +'_median.txt' , self.m_spec)
            np.savetxt(name + '_' + method +'_mad.txt' , self.err_spec)
            np.savetxt(name + '_' + method +'_norm.txt' , self.err_spec)
        else:
            np.savetxt(folder + 'Wavelength_from_'+str(self.w[i])+'_to_'+str(self.w[i:j])+'.txt')
            np.savetxt(folder + name + '_' + method +'_median.txt' , self.m_spec)
            np.savetxt(folder + name + '_' + method +'_mad.txt' , self.err_spec)
            np.savetxt(folder + name + '_' + method +'_norm.txt' , self.err_spec)

    def moving_average(self , window_size , limiti = True):
        """
        Function to perform a moving average smoothing on the normalized spectrum.
        
        Parameters
        ----------
        window_size : int
            Size of the step taken for the moving mean.
        
        Returns
        -------
        result : 1-dim array
            Moving-mean smoothed normalized spectrum.
        """
        if window_size < 1:
            raise ValueError("Window_size must be at least 1.")

        data = np.asarray(self.norm)
        result = np.empty(len(self.norm))
        half_window = window_size // 2

        for i in range(len(self.norm)):
            start = max(0, i - half_window)
            end = min(len(self.norm), i + half_window + 1)
            result[i] = np.mean(data[start:end])
        self.final_smooth = result

        if limiti == True:
            lims = self.limits()
            a , b = lims[0] , lims[1]
            plt.plot(self.w[a:b] , self.norm , 'b')
            plt.plot(self.w[a:b] , self.final_smooth , 'r')
        else:
            plt.plot(self.w , self.norm , 'b')
            plt.plot(self.w , self.final_smooth , 'r')
        plt.xlabel('$\lambda$[nm]')
        plt.show()
        
        return self.final_smooth
    
    def savgol(self, window, order , limiti = True):
        """
        Function to perform a Savitzky-Golay smoothing on the normalized spectrum as given in https://pubs.acs.org/doi/10.1021/ac60214a047.
        Mutuated from scipy.signal (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html).
        
        Parameters
        ----------
        window : int
            Size of the convolution window.
        order : int
            Order of the convolutional fit.
        
        Returns
        -------
        result : 1-dim array
            Moving-mean smoothed normalized spectrum.
        """
        
        self.final_smooth = savgol_filter(self.norm , window , order)
        self.final_smooth_error = np.zeros(len(self.final_smooth))

        if limiti == True:
            lims = self.limits()
            a , b = lims[0] , lims[1]
            plt.plot(self.w[a:b] , self.norm , 'b')
            plt.plot(self.w[a:b] , self.final_smooth , 'r')
        else:
            plt.plot(self.w , self.norm , 'b')
            plt.plot(self.w , self.final_smooth , 'r')
        plt.xlabel('$\lambda$[nm]')
        plt.show()
        return self.final_smooth

    def bootstrapnorm(self , convexhull , N = 1000 , interp = 'linear' , lower = 2.5 , upper = 97.5):
        """
        Normalization of the target spectrum over the neutral. The error propagation is performed using a bootstrap algorithm. 
        
        Parameters
        ----------
        convex_hull : bool
            If the normalization is done with the convex hull or not. Default is False.
        N : int
            Number of bootstrap iteration. Deafult is 1000.
        interp : string {‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’}
            Is the interpolation of the convex hull using interp1d from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html).
            The possible arguments signify: ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a 
            spline interpolation of zeroth, first, second or third order; 
            ‘previous’ and ‘next’ simply return the previous or next value of the point; 
            ‘nearest-up’ and ‘nearest’ differ when interpolating half-integers (e.g. 0.5, 1.5) in that ‘nearest-up’ rounds up and ‘nearest’ rounds down.
            Default is ‘linear’.
        lower : float
            Lower bound for the percentile calculation. Default is 2.5.
        upper : float
            Upper bound for the percentile calculation. Default is 97.5
            
        Returns
        -------
        norm : array
            Normalized spectrum.
        error norm : array
            Propagated error of the normalized error. Calculated as mean of the upper and lower percentile bounds.
        """
        lims = self.limits()
        target = self.spectra[:,lims[0]:lims[1]] 
        W = self.w[lims[0]:lims[1]]
        bootstrap_ratios = np.zeros( ( N , len(W) ) )

        if convexhull == True:

            def compute_convex_hull(wavelengths, spectra):
                points = np.c_[wavelengths, np.max(spectra, axis=0)]  # Use max spectrum to ensure upper boundary
                augmented = np.vstack([points, [wavelengths[0], np.min(points[:, 1]) - 1], 
                                                   [wavelengths[-1], np.min(points[:, 1]) - 1]])  # Add support points
                hull = ConvexHull(augmented)
                hull_points = points[np.sort([v for v in hull.vertices if v < len(points)])]
                interp_func = interp1d(hull_points[:, 0], hull_points[:, 1], kind='linear', fill_value="extrapolate")
                return interp_func(wavelengths)  # Return interpolated convex hull spectrum

            S2 = compute_convex_hull(W, target)

            plt.plot(W , S2 , 'k')
            for i in range(target.shape[0]):
                plt.plot(W , target[i] , 'b')
            plt.plot(W , np.max(target , axis = 0) , 'r')
            plt.show()

            for i in range(N):
                S1 = target[np.random.choice(len(target) , len(W) , replace = True)]

                S1_resample , S2_resample = np.mean(S1 , axis = 0) , np.median(S2 , axis = 0)

                bootstrap_ratios[i, :] = S1_resample / S2_resample

        else:

            neutral = self.neutral[:,lims[0]:lims[1]]
            
            for i in range(N):
                S1 = target[np.random.choice(len(target) , len(W) , replace = True)]
                
                S2 = neutral[np.random.choice(len(neutral) , len(W) , replace = True)]

                S1_resample , S2_resample = np.mean(S1 , axis = 0) , np.median(S2 , axis = 0)

                bootstrap_ratios[i , :] = S1_resample / S2_resample

        ratio_mean = np.mean(bootstrap_ratios , axis = 0)
        ratio_ci_lower, ratio_ci_upper = np.percentile(bootstrap_ratios , [lower , upper] , axis = 0)

        self.norm = ratio_mean
        self.normerr = (ratio_ci_upper - ratio_ci_lower) / 2

        return self.norm , self.normerr
