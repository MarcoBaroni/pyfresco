"""RGBmap module for FRESCO package.
"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.transform import from_gcps, Affine
from pathlib import Path
from rasterio.transform import from_origin
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin, array_bounds
from rasterio.enums import ColorInterp, Resampling
from rasterio.warp import calculate_default_transform, reproject
from scipy.signal import savgol_filter
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from pathlib import Path

def open_raw(path_img_IF , path_hdr_IF , path_img_SR , path_hdr_SR):
    """
    Function to open the spectral parameters and spectral reflectance datacubes.
    
    Parameters
    ----------
    path_img_IF : string
        Complete path of the reflectance datacube.
    path_hdr_IF : string
        Complete path of the reflectance datacube header.
    path_img_SR : string
        Complete path of the spectral parameters datacube.
    path_hdr_SR : string
        Complete path of the spectral parameter datacube header.
        
    Returns
    -------
    img : python spectral.io.bsqfile.BsqFile object
        Spectral reflectances datacube.
    img_sr : python spectral.io.bsqfile.BsqFile object
        Spectral parameters datacube.
    wavelength : list
        List of the CRISM observation wavelengths.
    sr_names : list
        List of the CRISM spectral parameters.
    """
    
    img = envi.open(path_hdr_IF , path_img_IF)

    img_sr = envi.open(path_hdr_SR , path_img_SR)

    wavelength = np.array(img.metadata['wavelength']).astype(float)

    sr_names = img_sr.metadata['band names']

    return img , img_sr , wavelength , sr_names

class RGBImageManipulator():
    """
    Class to generate the RGB map. With this class it is possible to manually control the contrast of each RGB channel and, after the RGB map is produced, it possible to save it both in .txt and .tiff and also to georeferentiate it.
    
    Parameters
    ----------
    img : python spectral.io.bsqfile.BsqFile object
        The spectral reflectance datacube.
    img_sr : python spectral.io.bsqfile.BsqFile object
        The spectral parameter datacube.
    preset : string
        The preset as given in Viviano-Beck et al., 2014 (https://doi.org/10.1002/2014JE004627). If None then no preset is selected.
    ch1 : int
        If preset is None, this is the index of the spectral parameter used in the Red channel.
    ch2 : int
        If preset is None, this is the index of the spectral parameter used in the Green channel.
    ch3 : int
        If preset is None, this is the index of the spectral parameter used in the Blue channel.
    wavelength : list or array
        The wavelengths detected by CRISM.
    """
    def __init__(self , img , img_sr , preset , ch1 , ch2 , ch3 , wavelength):

        self.img = img
        self.img_sr = img_sr
        self.preset = preset
        self.ch1 = ch1
        self.ch2 = ch2
        self.ch3 = ch3
        self.w = wavelength

    def RGB_Viviano_Beck_2014(self):
        """
        This function uploads descriptions, spectral parameter names and combinations for RGB map as given in Viviano-Beck et al., 2014 (https://doi.org/10.1002/2014JE004627).
        Used in the function RGB_map_slider.
        
        Parameters
        ----------
        preset : string
            Pre-selected spectral parameters for RGB map generation as given in Viviano-Beck et al., 2014.
            
        Returns
        -------
        names : list
            List of the spectral parameters corresponding to the RGB channels.
        descriptions : string
            Description of the RGB map as given in Viviano-Beck et al., 2014.
        indexes : list
            Indexes of the spectral parameters for the RGB map as they sorted in the spectral parameter datacube.
        """
        sr = ['R770', 'RBR', 'BD530_2', 'SH600_2', 'SH770', 'BD640_2', 'BD860_2', 'BD920_2', 'RPEAK1',
              'BDI1000VIS', 'R440', 'IRR1', 'BDI1000IR', 'OLINDEX3', 'R1330', 'BD1300', 'LCPINDEX2', 
              'HCPINDEX2', 'VAR', 'ISLOPE1', 'BD1400', 'BD1435', 'BD1500_2', 'ICER1_2', 'BD1750_2', 
              'BD1900_2', 'BD1900R2', 'BDI2000', 'BD2100_2', 'BD2165', 'BD2190', 'MIN2200', 'BD2210_2',
              'D2200', 'BD2230', 'BD2250', 'MIN2250', 'BD2265', 'BD2290', 'D2300', 'BD2355', 'SINDEX2', 
              'ICER2_2', 'MIN2295_2480', 'MIN2345_2537', 'BD2500_2', 'BD3000', 'BD3100', 'BD3200', 
              'BD3400_2', 'CINDEX2', 'BD2600', 'IRR2', 'IRR3', 'R530', 'R600', 'R1080', 'R1506', 'R2529', 
              'R3920']

        # Descriptions from Viaviano-Beck et al. 2014

        descriptions = {'TRU': 'Enhanced true color.' , 
                        'VNA': 'Photometric correct I/F, used to correlate morphology and spectral variation.' , 
                        'FEM': 'Fe minerals absorption.' , 
                        'FM2': 'Complementary info on Fe minerals.' , 
                        'FAL': 'False color image. Red/orange -> olivine rich , blue/green -> clay , green -> carbonates , gray/brown -> basaltic .' ,
                        'MAF': 'Mafic mineralogy. Green/cyan -> Low Ca Pyroxene , blue/magenta -> High Ca Pyroxene.' , 
                        'HYD': 'Hydrated mineralogy. Magenta -> polyhydrated sulfates , yellow/green -> monohydrated sulfates , blue -> hydrated minerals .' ,
                        'PHY': 'Phyllosilicates.  Red -> non-hydr. Fe/Mg-OH minerals , magenta -> hydr. Fe/Mg-OH inerals , green -> non-hydr. Al/Si-OH minerals , cyan -> hydr. Al/Si-OH minerals , blue -> other hydrated minerals.' , 
                        'PFM': 'Phyllosilicates with Fe and Mg. Red/yellow -> prehnite, chlorite, epidote or Ca/Fe carbonate, cyan -> Fe/Mg smectites of Mg carbonates.' , 
                        'PAL': 'Phyllosilicates qith Al. Red/yellow -> Al smectites or hydrated silica, cyan -> alunite, light/white -> kaolinite.' ,
                        'HYS': 'Hydrated silica. Used to differentiate between Al-phyl and hyd. silica. Light red/yellow -> hydrated silica, yellow -> jarosite,  cyan -> Al-OH minerals, blue -> sulfates, clays, hydr. silica, carbonates or water ice.' ,
                        'ICE': 'H20/CO2 ice. Red -> sulfates, clays, hydr. silica, carbonate, water ice. Green -> Water ice, Blue -> carbon dioxide ice.' ,
                        'IC2': 'Complementary information about H20/CO2 ice. Red -> ice free surface, green -> water ice, blue -> carbon dioxide ice.' ,
                        'CHL': 'Info about chloride deposits. Yellow/green -> hydr. minerals and phyllosilicates, blue -> chloride.' ,
                        'CAR': 'Info about Mg carbonate minerals. Red/magenta -> Fe/Mg phyllosilicates, yellowish-white-bluish -> Mg carbonates, blue -> sulfates, clays, hydr. silica or carbonate.' ,
                        'CR2': 'Info to distinguish carbonate minerals. Red/magenta -> Mg-carbonates, green/cyan -> Fe/Ca carbonates.'
                       }

        names = {'TRU' : [sr.index('R600') , sr.index('R530') , sr.index('R440')] , 
                 'VNA' : [sr.index('R770') , sr.index('R770') , sr.index('R770')] , 
                 'FEM' : [sr.index('BD530_2') , sr.index('SH600_2') , sr.index('BDI1000VIS')] , 
                 'FM2' : [sr.index('BD530_2') , sr.index('BD920_2') , sr.index('BDI1000VIS')] , 
                 'FAL' : [sr.index('R2529') , sr.index('R1506') , sr.index('R1080')] , 
                 'MAF' : [sr.index('OLINDEX3') , sr.index('LCPINDEX2') , sr.index('HCPINDEX2')] , 
                 'HYD' : [sr.index('SINDEX2') , sr.index('BD2100_2') , sr.index('BD1900_2')] , 
                 'PHY' : [sr.index('D2300') , sr.index('D2200') , sr.index('BD1900R2')] , 
                 'PFM' : [sr.index('BD2355') , sr.index('D2300') , sr.index('BD2290')] , 
                 'PAL' : [sr.index('BD2210_2') , sr.index('BD2190') , sr.index('BD2165')] ,
                 'HYS' : [sr.index('MIN2250') , sr.index('BD2250') , sr.index('BD1900R2')] ,
                 'ICE' : [sr.index('BD1900_2') , sr.index('BD1500_2') , sr.index('BD1435')] ,
                 'IC2' : [sr.index('R3920') , sr.index('BD1500_2') , sr.index('BD1435')] ,
                 'CHL' : [sr.index('ISLOPE1') , sr.index('BD3000') , sr.index('IRR2')] , 
                 'CAR' : [sr.index('D2300') , sr.index('BD2500_2') , sr.index('BD1900_2')] ,
                 'CR2' : [sr.index('MIN2295_2480') , sr.index('MIN2345_2537') , sr.index('CINDEX2')]
                }
        
        print( names[self.preset] , [ sr[names[self.preset][0]] , sr[names[self.preset][1]] , sr[names[self.preset][2]] ])
        return names[self.preset] , descriptions[self.preset] , [ sr[names[self.preset][0]] , sr[names[self.preset][1]] , sr[names[self.preset][2]] ]

    def f(self, RGB, min_R, min_G, min_B, max_R, max_G, max_B, clip=False):
        """
        This function uploads the RGB map during the customization. This function is only used inside RGBmapmake.
        
        Parameters
        ----------
        RGB : 3-dim array
            The RGB image to be updated.
        min_R : float
            Minimum value for the Red channel.
        min_G : float
            Minimum value for the Green channel.
        min_B : float
            Minimum value for the Blue channel.
        max_R : float
            Maximum value for the Red channel.
        max_G : float
            Maximum value for the Green channel.
        max_B : float
            Maximum value for the Blue channel.
        clip : bool
            If True it clips the negative values. Default if False.
            
        Returns
        -------
        RGB_raw : 3-dim array
            Updated RGB map
        """
        stretches_min = [min_R , min_G , min_B]
        stretches_max = [max_R , max_G , max_B]

        stretch = np.array([ [min_R , max_R] , [min_G , max_G] , [min_B , max_B] ])

        RGBs = np.where(RGB < stretch[:,0], stretches_min, RGB) 
        RGBs = np.where(RGB > stretch[:,1], stretches_max, RGB) 

        RGB_raw = (RGBs - stretch[:,0]) / (stretch[:,1] - stretch[:,0])

        if clip == True:

            RGB_raw = np.clip(RGB_raw , 0. , 1.)

        return RGB_raw

    def area(self, hist, bins, line1, line2):
        """
        Function to calulate the percentile area of a histogram between two lines. Only used inside RGBmapmake.
        
        Parameters
        ----------
        hist : array
            Histogram of the value of the RGB channel.
        bins : int
            Number of bins of the histogram.
        line1 : matplotlib.axes.Axes
            The line object of the left percentiles.
        line2 : matplotlib.axes.Axes
            The line object of the right percentiles.
            
        Returns
        -------
        Areas : list
            List containing the area of the histogram on the left of the first line and on the right of the second line.
        """
        area1 , area2 , total_area = 0 , 0 , 0
        for S in range(len(bins)-1):

            F = S + 1

            total_area += hist[S]*(bins[F] - bins[S])

            if bins[S] <= line1.get_xdata():
                area1 += hist[S]*(bins[F] - bins[S])
            if bins[S] <= line2.get_xdata():
                area2 += hist[S]*(bins[F] - bins[S])

        area_inferior = area1*100/total_area
        area_superior = area2*100/total_area

        return [area_inferior , area_superior]

    def Labels(self):
        L = {'TRU': [['--' , 'white' , 'white'] , ['--' , 'white' , 'white']],#'Enhanced true color.' , 
          'VNA': [['--' , 'white' , 'white'] , ['--' , 'white' , 'white']],#'Photometric correct I/F, used to correlate morphology and spectral variation.' , 
          'FEM': [['--' , 'white' , 'white'] , ['--' , 'white' , 'white']],#'Fe minerals absorption.' , 
          'FM2': [['--' , 'white' , 'white'] , ['--' , 'white' , 'white']],#'Complementary info on Fe minerals.' , 
          'FAL': [['Olivine' , 'red' , 'orange'] , ['Clay' , 'mediumseagreen' , 'blue'] , ['Carbonates' , 'green' , 'green'] , ['Basalts' , 'gray' , 'brown']] ,
          'MAF': [['Olivine' , 'red' , 'red'] , ['Low Ca Pyroxene' , 'green' , 'cyan'] , ['High Ca Pyroxene' , 'blue' , 'magenta']],
          'HYD': [['Polyhydrated sulfates' , 'magenta' , 'magenta'] , ['Monohydrated sulfates' , 'yellow' , 'green'] , ['Hydrated minerals' , 'blue' , 'blue']],
          'PHY': [['Non hydrated Fe/Mg-OH' , 'red' , 'red'] , ['Hydrated Fe/Mg-OH' , 'magenta' , 'magenta'] , ['Non hydrated Al/Si-OH' , 'green' , 'green'] , ['Hydrated Al/Si-OH' , 'cyan' , 'cyan'] , ['Hydrated minerals' , 'blue' , 'blue']], 
          'PFM': [['Prehnite' , 'red' , 'yellow'] , ['Chlorite' ,  'red' , 'yellow'] , ['Epidote' , 'red' , 'yellow'] , ['Ca/Fe carbonate', 'red' , 'yellow'] , ['Fe/Mg smectites / Mg carbonates' , 'cyan' , 'cyan'] , ['Kaolinite' , 'white' , 'white']] ,
          'PAL': [['Al smectites/Hydrated silica' , 'red' , 'yellow'] , ['Alunite' , 'cyan' , 'cyan'] , ['Kaolinite' , 'white' , 'white']] ,
          'HYS': [['Hydrated silica' , 'red' , 'yellow'] , ['Jarosite' , 'yellow' , 'yellow'] , ['Al-OH minerals' , 'cyan' , 'cyan'] , ['Other hydrates' , 'blue' , 'blue']] ,
          'ICE': [['Other hydrates' , 'red' , 'red'] , ['H2O ice' , 'green' , 'green'] , ['CO2 ice' , 'blue' , 'blue'] ] ,
          'IC2': [['Ice free surface' , 'red' ,'red'] , ['H2O ice' , 'green' , 'green'] , ['CO2 ice' , 'blue' , 'blue'] ] , 
          'CHL': [['Hydr. mineral and phyllosilicates' , 'yellow' , 'green'] , ['Chloride' , 'blue' , 'blue']],
          'CAR': [['Fe/Mg phyllosilicates' , 'red' , 'magenta'] , ['Mg carbonates' , 'yellow' , 'lightblue'] , ['Other hydrates' , 'blue' , 'blue'] ] ,
          'CR2': [['Mg carbonates' , 'red' , 'magenta'] , ['Fe/Ca carbonates' , 'green' , 'cyan']]
         }
        return L

    def RGBmapmake(self, bins , FALSE = None , clip = True , cumhist = False , preset_true_colors = 'TRU' , use_false_color = False ,
                   R_min_in = [0,1]  , R_max_in = [0,1]  ,
                   G_min_in = [0,1]  , G_max_in = [0,1]  ,
                   B_min_in = [0,1]  , B_max_in = [0,1]  ,
                   init_R = [0,1]  , init_G = [0,1]  , init_B = [0,1]  ,
                   slider_step = 0.005 ,
                   slider_height = 0.02 , slider_width = 0.25 , slider_spacing = 0.05):
        """
        Function to perform the customization of the RGB map by moving apposite sliders to enhance constrast between different colors (i.e. spectral parameters).
        To finish the customization it is sufficent to close the plot window.
        
        Parameters
        ----------
        bins : int
            Number of bins to divide the histograms into.
        FALSE : 3-dim array
            RGB image to be used as background. Can be set to None to use the non-stretched TRU RGB map. Default is None.
        clip : bool
            If to clip the negative values or not. Default is True.
        cumhist : bool
            If to use cumulative histograms instead of frequency histograms. Default is False.
        preset_true_colors : string
            If use_false_color is False, here insert the preset name from Viviano-Beck et al., 2014 that wants to be used as background true color RGB image. Default is TRU
        use_false_color : bool
            If to use a pre-computed RGB background map or to select another one from Viviano-Beck et al., 2014 as it is (without stretching). Default is False.
        R_min_in : list of two floats
            Minimum and maximum possible values for the minimum of the Red channel. Default is [0,1].
        R_max_in : list of two floats
            Minimum and maximum possible values for the maximum of the Red channel. Default is [0,1].
        G_min_in : list of two floats
            Minimum and maximum possible values for the minimum of the Green channel. Default is [0,1].
        G_max_in : list of two floats
            Minimum and maximum possible values for the maximum of the Green channel. Default is [0,1].
        B_min_in : list of two floats
            Minimum and maximum possible values for the minimum of the Blue channel. Default is [0,1].
        B_max_in : list of two floats
            Minimum and maximum possible values for the maximum of the Blue channel. Default is [0,1].
        init_R : list of two floats
            Initial values of the Red channel. Default is [0,1].
        init_G : list of two floats
            Initial values of the Green channel. Default is [0,1].
        init_B : list of two floats
            Initial values of the Blue channel. Default is [0,1].
        slider_step : float
            Minimum step done by the slider while interacting with it. Default is 0.005.
        slider_height : float
            Height at which the sliders are posed in the plot. Default is 0.02.
        slider_width : float
            Width of the sliders. Default is 0.25.
        slider_spacing : float
            Space between the sliders. Default is 0.05.
            
        Returns
        -------
        RGB : 3-dim array
            Final RGB map in the form of a numpy array.
        stretches : list of float
            Final stretch values in the following order: final_min_R , final_min_G , final_min_B , final_max_R , final_max_G , final_max_B.
        """
        fig , ax = plt.subplots(1 , 4 , figsize = [10,5] , gridspec_kw={'width_ratios': [1,1,1,3]})
    
        # setting initial stretch to float
        init_min_R, init_min_G, init_min_B = float(init_R[0]) , float(init_G[0]) , float(init_B[0])
        init_max_R, init_max_G, init_max_B = float(init_R[1]) , float(init_G[1]) , float(init_B[1])

        # Creation of the RGB image either from coded ones or from custom select indices
        if self.preset == None:

            sr = ['R770', 'RBR', 'BD530_2', 'SH600_2', 'SH770', 'BD640_2', 'BD860_2', 'BD920_2', 'RPEAK1',
                  'BDI1000VIS', 'R440', 'IRR1', 'BDI1000IR', 'OLINDEX3', 'R1330', 'BD1300', 'LCPINDEX2', 
                  'HCPINDEX2', 'VAR', 'ISLOPE1', 'BD1400', 'BD1435', 'BD1500_2', 'ICER1_2', 'BD1750_2', 
                  'BD1900_2', 'BD1900R2', 'BDI2000', 'BD2100_2', 'BD2165', 'BD2190', 'MIN2200', 'BD2210_2',
                  'D2200', 'BD2230', 'BD2250', 'MIN2250', 'BD2265', 'BD2290', 'D2300', 'BD2355', 'SINDEX2', 
                  'ICER2_2', 'MIN2295_2480', 'MIN2345_2537', 'BD2500_2', 'BD3000', 'BD3100', 'BD3200', 
                  'BD3400_2', 'CINDEX2', 'BD2600', 'IRR2', 'IRR3', 'R530', 'R600', 'R1080', 'R1506', 'R2529', 
                  'R3920']

            sr_channels_number = [self.ch1 , self.ch2 , self.ch3]
            RGB_raw = np.array(self.img_sr[:,:,sr_channels_number].squeeze()).astype(float)
            
            ax[0].set_xlabel(sr[self.ch1] , color = 'red' , fontsize = 15)
            ax[1].set_xlabel(sr[self.ch2] , color = 'green' , fontsize = 15)
            ax[2].set_xlabel(sr[self.ch3] , color = 'blue' , fontsize = 15)

        else:
            sr_channels_number, descr , pars = self.RGB_Viviano_Beck_2014()#self.preset)
            print(sr_channels_number , pars) 
            RGB_raw = np.array(self.img_sr[:,:,sr_channels_number].squeeze())

            ax[0].set_xlabel(str(pars[0]) , color = 'red' , fontsize = 15)
            ax[1].set_xlabel(str(pars[1]) , color = 'green' , fontsize = 15)
            ax[2].set_xlabel(str(pars[2]) , color = 'blue' , fontsize = 15)

        if use_false_color == False:

            # Create the true or false color image
            true_channels , true_descr , true_pars = self.RGB_Viviano_Beck_2014()#preset_true_colors)
            RGB_true_colors = np.array(self.img_sr[:,:,true_channels].squeeze())
            RGB_true_colors[RGB_true_colors > 1.] = np.nan

        else:

            RGB_true_colors = FALSE

        # Remove of the borders and definition of image in function of the stretches
        RGB_raw[RGB_raw > 1.] = np.nan
        RGB_browse_norm = self.f(RGB_raw, init_R[0] , init_G[0] , init_B[0] ,
                                 init_R[1] , init_G[1] , init_B[1] , clip = clip)

        # Choosing between cumulative histogram and frequency histogram
        if cumhist == True:
            hR , biR , _ = ax[0].hist( RGB_raw[:,:,0].ravel() , bins , color = 'r' , alpha = 0.5 , density = True , cumulative = True )
            hG , biG , _ = ax[1].hist( RGB_raw[:,:,1].ravel() , bins , color = 'g' , alpha = 0.5 , density = True , cumulative = True )
            hB , biB , _ = ax[2].hist( RGB_raw[:,:,2].ravel() , bins , color = 'b' , alpha = 0.5 , density = True , cumulative = True )
        else:
            hR , biR , _ = ax[0].hist( RGB_raw[:,:,0].ravel() , bins , color = 'r' , alpha = 0.5 , density = True , cumulative = False )
            hG , biG , _ = ax[1].hist( RGB_raw[:,:,1].ravel() , bins , color = 'g' , alpha = 0.5 , density = True , cumulative = False )
            hB , biB , _ = ax[2].hist( RGB_raw[:,:,2].ravel() , bins , color = 'b' , alpha = 0.5 , density = True , cumulative = False )

        # Setting histogram graphs limits:
        ax[0].set_xlim(biR[np.argmin(biR)+1] , np.max(biR))
        ax[1].set_xlim(biG[np.argmin(biG)+1] , np.max(biG))
        ax[2].set_xlim(biB[np.argmin(biB)+1] , np.max(biB))
        ax[0].set_ylim(0 , np.max(np.sort(hR)[0:len(hR)-1]))
        ax[1].set_ylim(0 , np.max(np.sort(hG)[0:len(hG)-1]))
        ax[2].set_ylim(0 , np.max(np.sort(hB)[0:len(hB)-1]))

        # Definition of global variables
        final_min_R , final_min_G , final_min_B = None , None , None
        final_max_R , final_max_G , final_max_B = None , None , None
        RGB_final = None

        im = ax[3].imshow(RGB_browse_norm)

        im.set_data(RGB_browse_norm)

        # Plot of the vertical lines on the histograms and sacing their data
        vr = ax[0].axvline( init_R[0] , color = 'r' , linestyle = '--' , linewidth = 0.6 )
        vR = ax[0].axvline( init_R[1] , color = 'r' , linestyle = '--' , linewidth = 0.6 )
        vg = ax[1].axvline( init_G[0] , color = 'g' , linestyle = '--' , linewidth = 0.6 )
        vG = ax[1].axvline( init_G[1] , color = 'g' , linestyle = '--' , linewidth = 0.6 )
        vb = ax[2].axvline( init_B[0] , color = 'b' , linestyle = '--' , linewidth = 0.6 )
        vB = ax[2].axvline( init_B[1] , color = 'b' , linestyle = '--' , linewidth = 0.6 )

        vr.set_xdata([init_R[0]])
        vR.set_xdata([init_R[1]])
        vg.set_xdata([init_G[0]])
        vG.set_xdata([init_G[1]])
        vb.set_xdata([init_B[0]])
        vB.set_xdata([init_B[1]])

        # Percentile calculations
        areaR , areaG , areaB = self.area(hR , biR , vr , vR) , self.area(hG , biG , vg , vG) , self.area(hB , biB , vb , vB)

        aRmin , aRmax = areaR[0] , areaR[1]
        aGmin , aGmax = areaG[0] , areaG[1]
        aBmin , aBmax = areaB[0] , areaB[1]

        # Writing percentiles and saving the value
        textR = ax[0].set_title(f"{aRmin:.2f}%" + " - " + f"{aRmax:.2f}%" , color = 'r' , fontsize = 8)
        textG = ax[1].set_title(f"{aGmin:.2f}%" + " - " + f"{aGmax:.2f}%" , color = 'g' , fontsize = 8)
        textB = ax[2].set_title(f"{aGmin:.2f}%" + " - " + f"{aRmax:.2f}%" , color = 'b' , fontsize = 8)

        textR.set_text(f"Perc.: {aRmax:.2f}%")
        textG.set_text(f"Perc.: {aGmax:.2f}%")
        textB.set_text(f"Perc.: {aBmax:.2f}%")

        # Red channel sliders
        ax_min_R = plt.axes([0.05, 0.9, slider_width, slider_height])
        ax_max_R = plt.axes([0.05, 0.9 + slider_spacing, slider_width, slider_height])

        min_R_slider = Slider(ax_min_R, label = 'min R',
                              valmin = R_min_in[0], valmax = R_min_in[1],
                              valinit = init_min_R , valstep = slider_step , color = 'r')

        max_R_slider = Slider(ax_max_R, label = 'max R',
                              valmin = R_max_in[0], valmax = R_max_in[1],
                              valinit = init_max_R , valstep = slider_step , color = 'r')

        # Green channel sliders
        ax_min_G = plt.axes([0.37, 0.9, slider_width, slider_height])
        ax_max_G = plt.axes([0.37, 0.9 + slider_spacing, slider_width, slider_height])

        min_G_slider = Slider(ax_min_G, label = 'min G',
                              valmin = G_min_in[0], valmax = G_min_in[1],
                              valinit = init_min_G , valstep = slider_step , color = 'g')

        max_G_slider = Slider(ax_max_G, label = 'max G',
                              valmin = G_max_in[0], valmax = G_max_in[1],
                              valinit = init_max_G , valstep = slider_step , color = 'g')

        # Blue channel sliders
        ax_min_B = plt.axes([0.69, 0.9, slider_width, slider_height])
        ax_max_B = plt.axes([0.69, 0.9 + slider_spacing, slider_width, slider_height])

        min_B_slider = Slider(ax_min_B, label = 'min B',
                              valmin = B_min_in[0] , valmax = B_min_in[1],
                              valinit = init_min_B , valstep = slider_step , color = 'b')

        max_B_slider = Slider(ax_max_B, label = 'max B',
                              valmin = B_max_in[0], valmax = B_max_in[1],
                              valinit = init_max_B , valstep = slider_step , color = 'b')

        def on_button_clicked(event):
            nonlocal is_toggled
            is_toggled = not is_toggled

        # Define a function to update the image displayed in the plot
        is_toggled = True # Varibale to see the state of the button
        def update_image(event):
            nonlocal is_toggled, RGB_final, final_min_R, final_min_G, final_min_B, final_max_R, final_max_G, final_max_B
            if not is_toggled:
                # Change the image to a new state
                img_new = RGB_true_colors # define the new image here
                im.set_data(img_new)
                fig.canvas.draw_idle()
            else:
                # Retrieve current slider values
                min_R, max_R = min_R_slider.val, max_R_slider.val
                min_G, max_G = min_G_slider.val, max_G_slider.val
                min_B, max_B = min_B_slider.val, max_B_slider.val

                # Update final slider values
                final_min_R, final_min_G, final_min_B = min_R, min_G, min_B
                final_max_R, final_max_G, final_max_B = max_R, max_G, max_B

                # Calculate updated RGB image
                RGB_browse_norm = self.f(RGB_raw, min_R, min_G, min_B, max_R, max_G, max_B, clip)
                im.set_data(RGB_browse_norm)

                RGB_final = RGB_browse_norm

                # Calculate updated vertical lines on histograms
                vr.set_xdata([min_R])
                vR.set_xdata([max_R])
                vg.set_xdata([min_G])
                vG.set_xdata([max_G])
                vb.set_xdata([min_B])
                vB.set_xdata([max_B])

                # Calculate updated percentiles
                areaR , areaG , areaB = self.area(hR , biR , vr , vR) , self.area(hG , biG , vg , vG) , self.area(hB , biB , vb , vB)

                aRmin , aRmax = areaR[0] , areaR[1]
                aGmin , aGmax = areaG[0] , areaG[1]
                aBmin , aBmax = areaB[0] , areaB[1]

                textR.set_text(f"{aRmin:.2f}%" + " - " + f"{aRmax:.2f}%")
                textG.set_text(f"{aGmin:.2f}%" + " - " + f"{aGmax:.2f}%")
                textB.set_text(f"{aBmin:.2f}%" + " - " + f"{aBmax:.2f}%")

                # Change the image back to the original state
                fig.canvas.draw_idle()

        # Create button
        button_ax = plt.axes([0.8, 0.82, 0.1, 0.07])
        button = Button(button_ax, 'Change image')
        button.on_clicked(on_button_clicked)
        button.on_clicked(update_image)

        # Set up update function to be called when sliders are changed
        for slider in [min_R_slider, max_R_slider, min_G_slider, max_G_slider, min_B_slider, max_B_slider]:
            slider.on_changed(update_image)

        # Set up title
        if self.preset != None:
            ax[3].set_title(self.preset)
            if self.preset != 'TRU' and self.preset != 'VNA' and self.preset != 'FEM' and self.preset != 'FM2':
                
                patches = []
                labe = []

                L = self.Labels()[self.preset]

                for i in range(len(L)):

                    lab , c1 , c2 = L[i][0] , L[i][1] , L[i][2]

                    labe.append(str(' ') + lab)

                    m1, = plt.plot([], [], c=c1 , marker='s', markersize=20,
                                  fillstyle='left', linestyle='none')

                    m2, = plt.plot([], [], c=c2 , marker='s', markersize=20,
                                  fillstyle='right', linestyle='none')

                    patches.append((m1,m2))

                plt.legend(( patches ), (labe), numpoints = 1, labelspacing = 2, ncol = 2 , frameon=False ,
                          handletextpad = 1, handlelength = 1.5 , columnspacing = 5 ,
                          loc = 'lower right', fontsize = 10 , bbox_to_anchor = (1,-11) )

            
            else:
                print('No legend')
        else:
            ax[3].set_title('Custom map')
        ax[3].axis('off')

        # Creating mask to set NaN values outside the images to be shown in white instead of black
        MASK = np.isnan(RGB_browse_norm[:,:,0])
        ALPHA = np.zeros((RGB_browse_norm.shape[0] , RGB_browse_norm.shape[1]))
        ALPHA[MASK] = 1
        ax[3].imshow(ALPHA , alpha = ALPHA , cmap = 'Greys_r')

        plt.show()
        
        self.RGB_final = RGB_final

        # Return the final map
        return self.RGB_final , [final_min_R , final_min_G , final_min_B , final_max_R , final_max_G , final_max_B]

    def f_BW(self, BW, min_W, max_W, clip=False):
        """
        This function uploads the RGB map during the customization. This function is only used inside BWmapmake.
        
        Parameters
        ----------
        RGB : 3-dim array
            The RGB image to be updated.
        min_W : float
            Minimum value for the channel.
        max_W : float
            Maximum value for the channel.
        clip : bool
            If True it clips the negative values. Default if False.
            
        Returns
        -------
        BW_raw : 3-dim array
            Updated BW map
        """
        stretches_min = [min_W]
        stretches_max = [max_W]

        stretch = np.array([ [min_W , max_W] ])

        BWs = np.where(BW < stretch[:,0], stretches_min, BW) 
        BWs = np.where(BW > stretch[:,1], stretches_max, BW) 

        BW_raw = (BWs - stretch[:,0]) / (stretch[:,1] - stretch[:,0])

        if clip == True:

            BW_raw = np.clip(BW_raw , 0. , 1.)

        return BW_raw

    def area_BW(self, hist, bins, line1, line2):
        """
        Function to calulate the percentile area of a histogram between two lines. Only used inside BWmapmake.
        
        Parameters
        ----------
        hist : array
            Histogram of the value of the RGB channel.
        bins : int
            Number of bins of the histogram.
        line1 : matplotlib.axes.Axes
            The line object of the left percentiles.
        line2 : matplotlib.axes.Axes
            The line object of the right percentiles.
            
        Returns
        -------
        Areas : list
            List containing the area of the histogram on the left of the first line and on the right of the second line.
        """
        area1 , area2 , total_area = 0 , 0 , 0
        for S in range(len(bins)-1):

            F = S + 1

            total_area += hist[S]*(bins[F] - bins[S])

            if bins[S] <= line1.get_xdata():
                area1 += hist[S]*(bins[F] - bins[S])
            if bins[S] <= line2.get_xdata():
                area2 += hist[S]*(bins[F] - bins[S])

        area_inferior = area1*100/total_area
        area_superior = area2*100/total_area

        return [area_inferior , area_superior]

    def BWmapmake(self, bins, sp_param, FALSE=None, clip=True, cumhist=False,
                  preset_true_colors='TRU', use_false_color=False,
                  W_min_in=[0, 1], W_max_in=[0, 1],
                  init_W=[0, 1],
                  slider_step=0.005,
                  slider_height=0.02, slider_width=0.25, slider_spacing=0.05):
        """
        Function to perform the customization of a BW map by moving sliders to enhance
        contrast of a selected spectral parameter.
    
        Normal behavior:
        - shows the stretched BW map.
    
        Hold RGB button behavior:
        - while the button is pressed, pixels that are empty in the stretched BW map
          are replaced by the corresponding pixels from the RGB reference map.
    
        Empty pixels are defined as:
        - pixels where BW_norm == 0 after stretching
        - or pixels where BW_norm is NaN
    
        Returns
        -------
        BW_final : 2-dim array
            Final BW stretched map.
        stretches : list
            [final_min_W, final_max_W]
        """
    
        fig, ax = plt.subplots(
            1,
            2,
            figsize=[10, 5],
            gridspec_kw={'width_ratios': [2, 3]}
        )
    
        plt.subplots_adjust(top=0.82)
    
        # setting initial stretch to float
        init_min_W = float(init_W[0])
        init_max_W = float(init_W[1])
    
        sr = np.array([
            'R770', 'RBR', 'BD530_2', 'SH600_2', 'SH770', 'BD640_2',
            'BD860_2', 'BD920_2', 'RPEAK1', 'BDI1000VIS', 'R440',
            'IRR1', 'BDI1000IR', 'OLINDEX3', 'R1330', 'BD1300',
            'LCPINDEX2', 'HCPINDEX2', 'VAR', 'ISLOPE1', 'BD1400',
            'BD1435', 'BD1500_2', 'ICER1_2', 'BD1750_2', 'BD1900_2',
            'BD1900R2', 'BDI2000', 'BD2100_2', 'BD2165', 'BD2190',
            'MIN2200', 'BD2210_2', 'D2200', 'BD2230', 'BD2250',
            'MIN2250', 'BD2265', 'BD2290', 'D2300', 'BD2355',
            'SINDEX2', 'ICER2_2', 'MIN2295_2480', 'MIN2345_2537',
            'BD2500_2', 'BD3000', 'BD3100', 'BD3200', 'BD3400_2',
            'CINDEX2', 'BD2600', 'IRR2', 'IRR3', 'R530', 'R600',
            'R1080', 'R1506', 'R2529', 'R3920'
        ], dtype=str)
    
        # Find selected spectral parameter
        matches = np.where(sr == sp_param)[0]
    
        if matches.size == 0:
            raise ValueError(
                f"Spectral parameter '{sp_param}' not found. "
                f"Available parameters are: {list(sr)}"
            )
    
        ch = int(matches[0])
        print(ch)
    
        # Read selected spectral parameter band
        BW_raw = np.array(self.img_sr[:, :, [ch]]).astype(float)
    
        # Make sure BW_raw is 2D
        if BW_raw.ndim == 3 and BW_raw.shape[2] == 1:
            BW_raw = BW_raw[:, :, 0]
    
        ax[0].set_xlabel(sr[ch], color='black', fontsize=15)
    
        # Create RGB reference map
        if use_false_color is False:
    
            if preset_true_colors is None:
                raise ValueError(
                    "preset_true_colors cannot be None when use_false_color=False."
                )
    
            true_channels, true_descr, true_pars = self.RGB_Viviano_Beck_2014()#preset_true_colors)
    
            RGB_true_colors = np.array(self.img_sr[:, :, true_channels].squeeze()).astype(float)
    
            RGB_true_colors[RGB_true_colors > 1.] = np.nan
    
        else:
    
            if FALSE is None:
                raise ValueError(
                    "FALSE cannot be None when use_false_color=True."
                )
    
            RGB_true_colors = np.array(FALSE).astype(float)
    
        if RGB_true_colors.ndim != 3 or RGB_true_colors.shape[2] != 3:
            raise ValueError("RGB reference map must have shape (rows, cols, 3).")
    
        if BW_raw.shape != RGB_true_colors.shape[:2]:
            raise ValueError(
                f"Shape mismatch: BW map has shape {BW_raw.shape}, "
                f"but RGB reference has shape {RGB_true_colors.shape[:2]}."
            )
    
        # Remove invalid high values
        BW_raw[BW_raw > 1.] = np.nan
    
        # Initial BW stretch
        BW_browse_norm = self.f_BW(BW_raw,init_W[0],init_W[1],clip=clip)
    
        # Helper: convert BW to displayable RGB grayscale
        def bw_to_rgb(BW_norm):
            BW_norm = np.asarray(BW_norm).astype(float)
    
            if BW_norm.ndim == 3 and BW_norm.shape[2] == 1:
                BW_norm = BW_norm[:, :, 0]
    
            BW_display = BW_norm.copy()
    
            # Normal display: NaN pixels are white
            BW_display[np.isnan(BW_display)] = 1.0
    
            BW_display = np.clip(BW_display, 0.0, 1.0)
    
            return np.dstack([BW_display,BW_display,BW_display])
    
        # Helper: show RGB only in empty BW pixels
        def make_empty_pixel_rgb_overlay(BW_norm, RGB_reference):
            BW_norm = np.asarray(BW_norm).astype(float)
            RGB_reference = np.asarray(RGB_reference).astype(float)
    
            if BW_norm.ndim == 3 and BW_norm.shape[2] == 1:
                BW_norm = BW_norm[:, :, 0]
    
            composite = bw_to_rgb(BW_norm)
    
            empty_mask = np.isnan(BW_norm) | np.isclose(BW_norm, 0.0)
    
            composite[empty_mask, :] = RGB_reference[empty_mask, :]
    
            return composite
    
        # Histogram only on finite values
        BW_hist_values = BW_raw[np.isfinite(BW_raw)]
    
        if BW_hist_values.size == 0:
            raise ValueError(
                f"No finite values found for spectral parameter '{sp_param}'."
            )
    
        if cumhist is True:
            hW, biW, _ = ax[0].hist(BW_hist_values.ravel(),bins,color='k',alpha=0.5,density=True,cumulative=True)
        else:
            hW, biW, _ = ax[0].hist(BW_hist_values.ravel(),bins,color='k',alpha=0.5,density=True,cumulative=False)
    
        # Histogram limits
        ax[0].set_xlim(biW[0], biW[-1])
    
        finite_h = hW[np.isfinite(hW)]
    
        if finite_h.size > 1:
            ymax = np.max(np.sort(finite_h)[:-1])
        elif finite_h.size == 1:
            ymax = finite_h[0]
        else:
            ymax = 1
    
        if ymax <= 0:
            ymax = 1
    
        ax[0].set_ylim(0, ymax * 1.05)
    
        # Initial final values
        final_min_W = init_W[0]
        final_max_W = init_W[1]
        BW_final = BW_browse_norm.copy()
        current_BW = BW_browse_norm.copy()
    
        # Show initial BW map
        im = ax[1].imshow(bw_to_rgb(current_BW))
        im.set_data(bw_to_rgb(current_BW))
    
        # Vertical lines
        vw = ax[0].axvline(init_W[0],color='grey',linestyle='--',linewidth=0.6)
    
        vW = ax[0].axvline(init_W[1],color='r',linestyle='--',linewidth=0.6)
    
        vw.set_xdata([init_W[0]])
        vW.set_xdata([init_W[1]])
    
        # Local percentile calculation
        def area_BW_local(hist, bin_edges, min_val, max_val):
            total_area = 0.0
            area_left = 0.0
            area_right = 0.0
    
            for i in range(len(bin_edges) - 1):
                dx = bin_edges[i + 1] - bin_edges[i]
                area = hist[i] * dx
    
                total_area += area
    
                if bin_edges[i] <= min_val:
                    area_left += area
    
                if bin_edges[i] <= max_val:
                    area_right += area
    
            if total_area == 0:
                return [0.0, 0.0]
    
            area_inferior = area_left * 100.0 / total_area
            area_superior = area_right * 100.0 / total_area
    
            return [area_inferior, area_superior]
    
        areaW = area_BW_local(hW,biW,init_W[0],init_W[1])
    
        aWmin, aWmax = areaW[0], areaW[1]
    
        textW = ax[0].set_title(
            f"{aWmin:.2f}%" + " - " + f"{aWmax:.2f}%",
            color='k',
            fontsize=8
        )
    
        # Sliders
        ax_min_W = plt.axes([0.05,0.90,slider_width,slider_height])
    
        ax_max_W = plt.axes([0.05,0.90 + slider_spacing,slider_width,slider_height])
    
        min_W_slider = Slider(ax_min_W,label='min W',valmin=W_min_in[0],valmax=W_min_in[1],valinit=init_min_W,valstep=slider_step,color='k')
    
        max_W_slider = Slider(ax_max_W,label='max W',valmin=W_max_in[0],valmax=W_max_in[1],valinit=init_max_W,valstep=slider_step,color='k')
    
        # Button: press and hold to reveal RGB in empty pixels
        button_ax = plt.axes([0.78,0.88,0.14, 0.07])
    
        button = Button(button_ax,'Hold RGB')
    
        is_holding_rgb = False
    
        def show_bw():
            im.set_data(bw_to_rgb(current_BW))
            fig.canvas.draw_idle()
    
        def show_rgb_on_empty_pixels():
            composite = make_empty_pixel_rgb_overlay( current_BW, RGB_true_colors )
    
            im.set_data(composite)
            fig.canvas.draw_idle()
    
        def on_button_press(event):
            nonlocal is_holding_rgb
    
            if event.inaxes == button_ax:
                is_holding_rgb = True
                show_rgb_on_empty_pixels()
    
        def on_button_release(event):
            nonlocal is_holding_rgb
    
            if is_holding_rgb:
                is_holding_rgb = False
                show_bw()
    
        fig.canvas.mpl_connect('button_press_event',on_button_press)
    
        fig.canvas.mpl_connect('button_release_event', on_button_release )
    
        # Update image when sliders move
        def update_image(event):
            nonlocal BW_final, final_min_W, final_max_W, current_BW
    
            min_W = min_W_slider.val
            max_W = max_W_slider.val
    
            final_min_W = min_W
            final_max_W = max_W
    
            BW_browse_norm = self.f_BW(BW_raw,min_W, max_W,clip=clip)
    
            current_BW = BW_browse_norm.copy()
            BW_final = BW_browse_norm.copy()
    
            if is_holding_rgb:
                im.set_data(make_empty_pixel_rgb_overlay(current_BW,RGB_true_colors))
            else:
                im.set_data(bw_to_rgb(current_BW))
    
            vw.set_xdata([min_W])
            vW.set_xdata([max_W])
    
            areaW = area_BW_local(hW,biW,min_W,max_W)
    
            aWmin, aWmax = areaW[0], areaW[1]
    
            textW.set_text(f"{aWmin:.2f}%" + " - " + f"{aWmax:.2f}%")
    
            fig.canvas.draw_idle()
    
        for slider in [min_W_slider, max_W_slider]:
            slider.on_changed(update_image)
    
        # Plot title
        ax[1].set_title(f"{sp_param} BW map")
        ax[1].axis('off')
    
        plt.show()
    
        self.BW_final = BW_final
    
        return self.BW_final, [final_min_W, final_max_W]

    def savemap(self, name, folder=None, array=None, map_type='rgb', save_txt=True, save_tif=True, save_png=False, show=False):
        """
        Save either an RGB or BW map.
    
        For RGB maps:
            - saves 3 txt files: name_R.txt, name_G.txt, name_B.txt
            - saves a georeferenced RGBA GeoTIFF
            - optionally saves an RGB PNG
    
        For BW maps:
            - saves 1 txt file: name.txt
            - saves a georeferenced single-band float32 GeoTIFF
            - optionally saves a black-and-white PNG
    
        Parameters
        ----------
        name : str
            Output base name.
    
        folder : str or pathlib.Path, optional
            Output folder. If None, saves in current directory. Default is None.
    
        array : np.ndarray, optional
            Array to save. If None, uses self.RGB_final or self.BW_final depending on map_type.
    
        map_type : str
            'rgb', 'bw'. Default is rgb
    
        save_txt : bool
            If True, save txt file(s). Default is True.
    
        save_tif : bool
            If True, save georeferenced GeoTIFF. Default is True.
    
        save_png : bool
            If True, save PNG visualization. Default is False.
    
        show : bool
            If True, show PNG preview. Default is False,
    
        Returns
        -------
        None
        """
    
        output_folder = Path(folder) if folder is not None else Path(".")
        output_folder.mkdir(parents=True, exist_ok=True)
    
        name_path = Path(name)
        base_name = name_path.stem
    
        # ------------------------------------------------------------
        # Select array and map type
        # ------------------------------------------------------------
        '''
        map_type = str(map_type).upper()
    
        if array is not None:
    
            data = np.asarray(array).astype(float)
    
            if map_type == 'AUTO':
                if data.ndim == 3 and data.shape[2] == 3:
                    map_type = 'RGB'
                elif data.ndim == 2 or (data.ndim == 3 and data.shape[2] == 1):
                    map_type = 'BW'
                else:
                    raise ValueError("Cannot infer map_type from array shape.")
    
        else:
    
            if map_type == 'rgb':
    
                if not hasattr(self, "RGB_final") or self.RGB_final is None:
                    raise ValueError("RGB_final does not exist. Create an RGB map first or pass array=...")
    
                data = np.asarray(self.RGB_final).astype(float)
    
            elif map_type == 'bw':
    
                if not hasattr(self, "BW_final") or self.BW_final is None:
                    raise ValueError("BW_final does not exist. Create a BW map first or pass array=...")
    
                data = np.asarray(self.BW_final).astype(float)
    
            else:
                raise ValueError("map_type must be 'rgb', 'bw'")
        '''   
        # ------------------------------------------------------------
        # Validate RGB / BW shape
        # ------------------------------------------------------------
        if map_type == 'rgb':
    
            RGB = self.RGB_final#data
    
            if RGB.ndim != 3 or RGB.shape[2] != 3:
                raise ValueError("RGB map must have shape (rows, cols, 3).")
    
        elif map_type == 'bw':
    
            BW = self.BW_final#data
    
            if BW.ndim == 3 and BW.shape[2] == 1:
                BW = BW[:, :, 0]
    
            if BW.ndim != 2:
                raise ValueError("BW map must have shape (rows, cols).")
    
        else:
            raise ValueError("map_type must be 'RGB', 'BW', or 'auto'.")
    
        # ------------------------------------------------------------
        # Helper: get georeference from ENVI metadata
        # ------------------------------------------------------------
        def get_georeference():
    
            metadata = self.img_sr.metadata
    
            if "map info" not in metadata:
                raise ValueError("No 'map info' found in self.img_sr.metadata. Cannot georeference GeoTIFF.")
    
            if "coordinate system string" not in metadata:
                raise ValueError("No 'coordinate system string' found in self.img_sr.metadata. Cannot assign CRS to GeoTIFF.")
    
            map_info = metadata["map info"]
            wkt = metadata["coordinate system string"]
    
            if isinstance(map_info, str):
                map_info = map_info.strip("{}").split(",")
                map_info = [v.strip() for v in map_info]
    
            if isinstance(wkt, list):
                wkt = ",".join(wkt)
    
            wkt = str(wkt).strip("{}")
    
            ref_pixel_x = float(map_info[1])
            ref_pixel_y = float(map_info[2])
            map_x = float(map_info[3])
            map_y = float(map_info[4])
            pixel_size_x = float(map_info[5])
            pixel_size_y = float(map_info[6])
    
            x_origin = map_x - (ref_pixel_x - 0.5) * pixel_size_x
            y_origin = map_y + (ref_pixel_y - 0.5) * pixel_size_y
    
            transform = from_origin(x_origin, y_origin, pixel_size_x, pixel_size_y)
            crs = CRS.from_wkt(wkt)
    
            return transform, crs
    
        # ------------------------------------------------------------
        # Save RGB map
        # ------------------------------------------------------------
        if map_type == 'rgb':
    
            R = RGB[:, :, 0]
            G = RGB[:, :, 1]
            B = RGB[:, :, 2]
    
            if save_txt is True:
                np.savetxt(output_folder / f"{base_name}_R.txt", R)
                np.savetxt(output_folder / f"{base_name}_G.txt", G)
                np.savetxt(output_folder / f"{base_name}_B.txt", B)
    
            if save_tif is True:
    
                transform, crs = get_georeference()
    
                valid_mask = np.all(np.isfinite(RGB), axis=2)
    
                RGB_clean = RGB.copy()
                RGB_clean[~np.isfinite(RGB_clean)] = 0.0
                RGB_clean = np.clip(RGB_clean, 0.0, 1.0)
    
                RGB_uint8 = (RGB_clean * 255).round().astype("uint8")
                alpha = np.where(valid_mask, 255, 0).astype("uint8")
    
                tif_path = output_folder / f"{base_name}.tif"
    
                with rasterio.open(
                    tif_path,
                    "w",
                    driver="GTiff",
                    height=RGB_uint8.shape[0],
                    width=RGB_uint8.shape[1],
                    count=4,
                    dtype="uint8",
                    crs=crs,
                    transform=transform
                ) as dst:
    
                    dst.write(RGB_uint8[:, :, 0], 1)
                    dst.write(RGB_uint8[:, :, 1], 2)
                    dst.write(RGB_uint8[:, :, 2], 3)
                    dst.write(alpha, 4)
    
                    dst.colorinterp = (
                        ColorInterp.red,
                        ColorInterp.green,
                        ColorInterp.blue,
                        ColorInterp.alpha
                    )
    
            if save_png is True:
    
                RGB_png = RGB.copy()
                RGB_png[~np.isfinite(RGB_png)] = 0.0
                RGB_png = np.clip(RGB_png, 0.0, 1.0)
    
                png_path = output_folder / f"{base_name}.png"
    
                plt.figure()
                plt.imshow(RGB_png)
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(png_path, bbox_inches="tight", pad_inches=0, transparent=True)
    
                if show is True:
                    plt.show()
                else:
                    plt.close()
    
            return
    
        # ------------------------------------------------------------
        # Save BW map
        # ------------------------------------------------------------
        if map_type == 'bw':
    
            if save_txt is True:
                np.savetxt(output_folder / f"{base_name}.txt", BW)
    
            if save_tif is True:
    
                transform, crs = get_georeference()
    
                BW_out = BW.astype("float32")
    
                tif_path = output_folder / f"{base_name}.tif"
    
                with rasterio.open(
                    tif_path,
                    "w",
                    driver="GTiff",
                    height=BW_out.shape[0],
                    width=BW_out.shape[1],
                    count=1,
                    dtype="float32",
                    crs=crs,
                    transform=transform,
                    nodata=np.nan
                ) as dst:
    
                    dst.write(BW_out, 1)
    
            if save_png is True:
    
                BW_png = BW.copy()
                BW_png[~np.isfinite(BW_png)] = np.nan
    
                png_path = output_folder / f"{base_name}.png"
    
                plt.figure()
                plt.imshow(BW_png, cmap="gray")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(png_path, bbox_inches="tight", pad_inches=0, transparent=True)
    
                if show is True:
                    plt.show()
                else:
                    plt.close()
    
            return

    def _nearest_band_index(self, wavelength_value):
        """
        Find the index of the spectral band closest to the requested wavelength.
    
        Parameters
        ----------
        wavelength_value : float
            Requested wavelength.
    
        Returns
        -------
        idx : int
            Index of the closest wavelength band.
        used_wavelength : float
            Actual wavelength used.
        """
    
        wavelengths = np.asarray(self.w).astype(float)
    
        idx = int(np.nanargmin(np.abs(wavelengths - float(wavelength_value))))
    
        return idx, float(wavelengths[idx])
    
    
    def _read_reflectance_band(self, band_index, invalid_reflectance_threshold=1.0):
        """
        Read one reflectance band from self.img.
    
        Parameters
        ----------
        band_index : int
            Spectral band index.
    
        invalid_reflectance_threshold : float or None
            If not None, values greater than this threshold are set to NaN.
            For CRISM reflectance, values > 1 are usually invalid/border/fill values.
    
        Returns
        -------
        band : 2D np.ndarray
            Reflectance band.
        """
    
        band = np.array(self.img[:, :, [band_index]]).astype(float)
    
        if band.ndim == 3 and band.shape[2] == 1:
            band = band[:, :, 0]
    
        if invalid_reflectance_threshold is not None:
            band[band > invalid_reflectance_threshold] = np.nan
    
        return band
    
    
    def _compute_band_ratio(self, numerator_wavelength, denominator_wavelength,
                            invalid_reflectance_threshold=1.0):
        """
        Compute one band ratio:
    
            R(numerator_wavelength) / R(denominator_wavelength)
    
        The closest available bands are used.
    
        Parameters
        ----------
        numerator_wavelength : float
            Numerator wavelength.
    
        denominator_wavelength : float
            Denominator wavelength.
    
        invalid_reflectance_threshold : float or None
            If not None, reflectance values greater than this threshold are set to NaN
            before computing the ratio.
    
        Returns
        -------
        ratio : 2D np.ndarray
            Band-ratio map.
    
        info : dict
            Information about requested and used wavelengths and band indices.
        """
    
        numerator_index, used_numerator = self._nearest_band_index(
            numerator_wavelength
        )
    
        denominator_index, used_denominator = self._nearest_band_index(
            denominator_wavelength
        )
    
        numerator = self._read_reflectance_band(
            numerator_index,
            invalid_reflectance_threshold=invalid_reflectance_threshold
        )
    
        denominator = self._read_reflectance_band(
            denominator_index,
            invalid_reflectance_threshold=invalid_reflectance_threshold
        )
    
        denominator[np.isclose(denominator, 0.0)] = np.nan
    
        ratio = numerator / denominator
    
        ratio[~np.isfinite(ratio)] = np.nan
    
        info = {
            "requested_numerator": numerator_wavelength,
            "requested_denominator": denominator_wavelength,
            "used_numerator": used_numerator,
            "used_denominator": used_denominator,
            "numerator_index": numerator_index,
            "denominator_index": denominator_index,
        }
    
        return ratio, info
    
    
    def _finite_values(self, array, name="array"):
        """
        Return finite values from an array.
    
        Parameters
        ----------
        array : np.ndarray
            Input array.
    
        name : str
            Name used in the error message.
    
        Returns
        -------
        values : 1D np.ndarray
            Finite values.
        """
    
        array = np.asarray(array).astype(float)
    
        values = array[np.isfinite(array)]
    
        if values.size == 0:
            raise ValueError(f"No finite values found in {name}.")
    
        return values
    
    
    def _percentile_range(self, array, percentiles):
        """
        Compute a percentile range on finite values.
    
        Parameters
        ----------
        array : np.ndarray
            Input array.
    
        percentiles : list or tuple
            Two percentiles, e.g. (2, 98).
    
        Returns
        -------
        range_values : list
            [lower_percentile_value, upper_percentile_value]
        """
    
        values = self._finite_values(array)
    
        return [
            float(np.nanpercentile(values, percentiles[0])),
            float(np.nanpercentile(values, percentiles[1]))
        ]
    
    
    def _area_from_hist(self, hist, bin_edges, min_val, max_val):
        """
        Compute approximate percentage area below min_val and max_val
        from an histogram.
    
        Parameters
        ----------
        hist : np.ndarray
            Histogram values.
    
        bin_edges : np.ndarray
            Histogram bin edges.
    
        min_val : float
            Lower stretch value.
    
        max_val : float
            Upper stretch value.
    
        Returns
        -------
        areas : list
            [percentage below min_val, percentage below max_val]
        """
    
        total_area = 0.0
        area_left = 0.0
        area_right = 0.0
    
        for i in range(len(bin_edges) - 1):
    
            dx = bin_edges[i + 1] - bin_edges[i]
            area = hist[i] * dx
    
            total_area += area
    
            if bin_edges[i] <= min_val:
                area_left += area
    
            if bin_edges[i] <= max_val:
                area_right += area
    
        if total_area == 0:
            return [0.0, 0.0]
    
        return [
            area_left * 100.0 / total_area,
            area_right * 100.0 / total_area
        ]
    
    
    def _bw_to_rgb(self, BW):
        """
        Convert a 2D BW map to a 3-channel grayscale RGB image for display.
    
        NaN pixels are shown as white.
    
        Parameters
        ----------
        BW : 2D np.ndarray
            BW image.
    
        Returns
        -------
        RGB : 3D np.ndarray
            RGB grayscale image.
        """
    
        BW = np.asarray(BW).astype(float)
    
        if BW.ndim == 3 and BW.shape[2] == 1:
            BW = BW[:, :, 0]
    
        BW_display = BW.copy()
        BW_display[np.isnan(BW_display)] = 1.0
        BW_display = np.clip(BW_display, 0.0, 1.0)
    
        return np.dstack([
            BW_display,
            BW_display,
            BW_display
        ])

    def _nearest_minimum_colormap(self, colors=("yellow", "blue", "red")):
        """
        Colormap for nearest-minimum maps.
    
        Parameters
        ----------
        colors : tuple/list of 3 matplotlib-compatible colors
            Colors in this order:
    
            colors[0] -> lower wavelengths than target_wavelength
            colors[1] -> target_wavelength
            colors[2] -> higher wavelengths than target_wavelength
    
            Examples:
            ("yellow", "blue", "red")
            ("cyan", "black", "magenta")
            ("#ffff00", "#0000ff", "#ff0000")
        """
    
        if colors is None:
            colors = ("yellow", "blue", "red")
    
        if len(colors) != 3:
            raise ValueError(
                "colors must contain exactly three colors: "
                "(low_wavelength_color, target_color, high_wavelength_color)."
            )
    
        color_low, color_center, color_high = colors
    
        return LinearSegmentedColormap.from_list(
            "nearest_minimum_custom_colormap",
            [
                (0.0, color_low),
                (0.5, color_center),
                (1.0, color_high)
            ]
        )

    def BR_BWmapmake(self, numerator_wavelength, denominator_wavelength,
                     bins=500, stretch=True, clip=True, cumhist=False,
                     stretch_percentiles=(2, 98),
                     slider_percentiles=(0.5, 99.5),
                     W_min_in=None, W_max_in=None, init_W=None,
                     invalid_reflectance_threshold=1.0,
                     slider_step=0.005,
                     slider_height=0.02, slider_width=0.25,
                     slider_spacing=0.05):
        """
        Create a BW map from a single band ratio:
    
            R(numerator_wavelength) / R(denominator_wavelength)
    
        Parameters
        ----------
        numerator_wavelength : float
            Numerator wavelength.
    
        denominator_wavelength : float
            Denominator wavelength.
    
        bins : int
            Number of histogram bins.
    
        stretch : bool
            If True, interactive stretching sliders are shown.
            If False, the raw band-ratio map is returned.
    
        clip : bool
            If True, stretched values are clipped to [0, 1].
    
        cumhist : bool
            If True, use cumulative histogram.
    
        stretch_percentiles : tuple
            Percentiles used to initialize the stretch if init_W is None.
    
        slider_percentiles : tuple
            Percentiles used to define default slider ranges if W_min_in or
            W_max_in are None.
    
        W_min_in : list or None
            Slider range for minimum stretch value.
    
        W_max_in : list or None
            Slider range for maximum stretch value.
    
        init_W : list or None
            Initial stretch values.
    
        invalid_reflectance_threshold : float or None
            If not None, reflectance values greater than this threshold are set to NaN.
    
        Returns
        -------
        BR_final : 2D np.ndarray
            Final BW band-ratio map.
    
        stretches : list or None
            If stretch=True: [final_min_W, final_max_W].
            If stretch=False: None.
        """
    
        BR_raw, info = self._compute_band_ratio(
            numerator_wavelength,
            denominator_wavelength,
            invalid_reflectance_threshold=invalid_reflectance_threshold
        )
    
        title = (
            f"R{info['used_numerator']:.0f} / "
            f"R{info['used_denominator']:.0f}"
        )
    
        print("Requested numerator wavelength:  ", info["requested_numerator"])
        print("Used numerator wavelength:       ", info["used_numerator"])
        print("Requested denominator wavelength:", info["requested_denominator"])
        print("Used denominator wavelength:     ", info["used_denominator"])
        print("Numerator band index:            ", info["numerator_index"])
        print("Denominator band index:          ", info["denominator_index"])
    
        self.BR_raw = BR_raw.copy()
        self.BR_info = info
    
        # ------------------------------------------------------------
        # No stretching
        # ------------------------------------------------------------
        if stretch is False:
    
            fig, ax = plt.subplots(
                1,
                2,
                figsize=[10, 5],
                gridspec_kw={'width_ratios': [2, 3]}
            )
    
            values = self._finite_values(BR_raw, name=title)
    
            ax[0].hist(
                values.ravel(),
                bins,
                color='k',
                alpha=0.5,
                density=True,
                cumulative=cumhist
            )
    
            ax[0].set_xlabel(title, color='black', fontsize=12)
    
            im = ax[1].imshow(BR_raw, cmap='gray')
            ax[1].set_title(title + " raw")
            ax[1].axis("off")
    
            plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    
            plt.show()
    
            self.BW_final = BR_raw.copy()
            self.BR_final = BR_raw.copy()
    
            return self.BR_final, None
    
        # ------------------------------------------------------------
        # Stretching mode
        # ------------------------------------------------------------
        fig, ax = plt.subplots(
            1,
            2,
            figsize=[10, 5],
            gridspec_kw={'width_ratios': [2, 3]}
        )
    
        plt.subplots_adjust(top=0.82)
    
        if init_W is None:
            init_W = self._percentile_range(BR_raw, stretch_percentiles)
    
        if W_min_in is None:
            W_min_in = self._percentile_range(BR_raw, slider_percentiles)
    
        if W_max_in is None:
            W_max_in = self._percentile_range(BR_raw, slider_percentiles)
    
        init_min_W = float(init_W[0])
        init_max_W = float(init_W[1])
    
        W_min_in = [
            min(float(W_min_in[0]), init_min_W),
            max(float(W_min_in[1]), init_min_W)
        ]
    
        W_max_in = [
            min(float(W_max_in[0]), init_max_W),
            max(float(W_max_in[1]), init_max_W)
        ]
    
        BR_final = self.f_BW(
            BR_raw,
            init_min_W,
            init_max_W,
            clip=clip
        )
    
        values = self._finite_values(BR_raw, name=title)
    
        hW, biW, _ = ax[0].hist(
            values.ravel(),
            bins,
            color='k',
            alpha=0.5,
            density=True,
            cumulative=cumhist
        )
    
        ax[0].set_xlim(biW[0], biW[-1])
        ax[0].set_xlabel(title, color='black', fontsize=12)
    
        finite_h = hW[np.isfinite(hW)]
    
        if finite_h.size > 1:
            ymax = np.max(np.sort(finite_h)[:-1])
        elif finite_h.size == 1:
            ymax = finite_h[0]
        else:
            ymax = 1
    
        if ymax <= 0:
            ymax = 1
    
        ax[0].set_ylim(0, ymax * 1.05)
    
        final_min_W = init_min_W
        final_max_W = init_max_W
    
        im = ax[1].imshow(self._bw_to_rgb(BR_final))
        ax[1].set_title(title + " stretched")
        ax[1].axis("off")
    
        vw = ax[0].axvline(
            init_min_W,
            color='grey',
            linestyle='--',
            linewidth=0.6
        )
    
        vW = ax[0].axvline(
            init_max_W,
            color='r',
            linestyle='--',
            linewidth=0.6
        )
    
        areaW = self._area_from_hist(
            hW,
            biW,
            init_min_W,
            init_max_W
        )
    
        textW = ax[0].set_title(
            f"{areaW[0]:.2f}%" + " - " + f"{areaW[1]:.2f}%",
            color='k',
            fontsize=8
        )
    
        ax_min_W = plt.axes([
            0.05,
            0.90,
            slider_width,
            slider_height
        ])
    
        ax_max_W = plt.axes([
            0.05,
            0.90 + slider_spacing,
            slider_width,
            slider_height
        ])
    
        min_W_slider = Slider(
            ax_min_W,
            label='min W',
            valmin=W_min_in[0],
            valmax=W_min_in[1],
            valinit=init_min_W,
            valstep=slider_step,
            color='k'
        )
    
        max_W_slider = Slider(
            ax_max_W,
            label='max W',
            valmin=W_max_in[0],
            valmax=W_max_in[1],
            valinit=init_max_W,
            valstep=slider_step,
            color='k'
        )
    
        def update_BW(event):
            nonlocal BR_final, final_min_W, final_max_W
    
            min_W = min_W_slider.val
            max_W = max_W_slider.val
    
            final_min_W = min_W
            final_max_W = max_W
    
            BR_final = self.f_BW(
                BR_raw,
                min_W,
                max_W,
                clip=clip
            )
    
            im.set_data(self._bw_to_rgb(BR_final))
    
            vw.set_xdata([min_W])
            vW.set_xdata([max_W])
    
            areaW = self._area_from_hist(
                hW,
                biW,
                min_W,
                max_W
            )
    
            textW.set_text(
                f"{areaW[0]:.2f}%" + " - " + f"{areaW[1]:.2f}%"
            )
    
            fig.canvas.draw_idle()
    
        for slider in [min_W_slider, max_W_slider]:
            slider.on_changed(update_BW)
    
        plt.show()
    
        self.BW_final = BR_final.copy()
        self.BR_final = BR_final.copy()
    
        return self.BR_final, [final_min_W, final_max_W]

    def BR_RGBmapmake(self, numerator_wavelengths, denominator_wavelengths,
                      bins=500, stretch=True, clip=True, cumhist=False,
                      stretch_percentiles=(0.1, 99.9),
                      slider_percentiles=(0.1, 99.9),
                      R_min_in=None, R_max_in=None,
                      G_min_in=None, G_max_in=None,
                      B_min_in=None, B_max_in=None,
                      init_R=None, init_G=None, init_B=None,
                      invalid_reflectance_threshold=1.0,
                      slider_step=0.0005,
                      slider_height=0.02, slider_width=0.25,
                      slider_spacing=0.05):
        """
        Create an RGB map from three band ratios.
    
        Red channel:
            R(numerator_wavelengths[0]) / R(denominator_wavelengths[0])
    
        Green channel:
            R(numerator_wavelengths[1]) / R(denominator_wavelengths[1])
    
        Blue channel:
            R(numerator_wavelengths[2]) / R(denominator_wavelengths[2])
    
        Parameters
        ----------
        numerator_wavelengths : list of 3 floats
            Numerator wavelengths for R, G, B.
    
        denominator_wavelengths : list of 3 floats
            Denominator wavelengths for R, G, B.
    
        bins : int
            Number of histogram bins.
    
        stretch : bool
            If True, interactive stretching sliders are shown.
            If False, raw band ratios are returned. For display only, the RGB image
            is clipped to [0, 1].
    
        clip : bool
            If True, stretched values are clipped to [0, 1].
    
        cumhist : bool
            If True, use cumulative histograms.
    
        stretch_percentiles : tuple
            Percentiles used to initialize the stretch if init_R, init_G, init_B
            are None.
    
        slider_percentiles : tuple
            Percentiles used to define default slider ranges.
    
        invalid_reflectance_threshold : float or None
            If not None, reflectance values greater than this threshold are set to NaN.
    
        Returns
        -------
        RGB_final : 3D np.ndarray
            Final RGB band-ratio map.
    
        stretches : list or None
            If stretch=True:
                [final_min_R, final_min_G, final_min_B,
                 final_max_R, final_max_G, final_max_B]
    
            If stretch=False:
                None.
        """
    
        if np.isscalar(numerator_wavelengths) or np.isscalar(denominator_wavelengths):
            raise ValueError(
                "BR_RGBmapmake requires two lists of three wavelengths."
            )
    
        numerator_wavelengths = list(numerator_wavelengths)
        denominator_wavelengths = list(denominator_wavelengths)
    
        if len(numerator_wavelengths) != 3 or len(denominator_wavelengths) != 3:
            raise ValueError(
                "numerator_wavelengths and denominator_wavelengths must both "
                "contain exactly three values."
            )
    
        ratio_channels = []
        info_channels = []
    
        for num_w, den_w in zip(numerator_wavelengths, denominator_wavelengths):
    
            ratio, info = self._compute_band_ratio(
                num_w,
                den_w,
                invalid_reflectance_threshold=invalid_reflectance_threshold
            )
    
            ratio_channels.append(ratio)
            info_channels.append(info)
    
        BR_raw = np.dstack(ratio_channels)
    
        labels = [
            f"R{info_channels[0]['used_numerator']:.0f}/R{info_channels[0]['used_denominator']:.0f}",
            f"R{info_channels[1]['used_numerator']:.0f}/R{info_channels[1]['used_denominator']:.0f}",
            f"R{info_channels[2]['used_numerator']:.0f}/R{info_channels[2]['used_denominator']:.0f}",
        ]
    
        print("Band ratio RGB channels:")
        print("Red:   ", labels[0], info_channels[0])
        print("Green: ", labels[1], info_channels[1])
        print("Blue:  ", labels[2], info_channels[2])
    
        self.BR_raw = BR_raw.copy()
        self.BR_info = info_channels
    
        colors = ['r', 'g', 'b']
    
        # ------------------------------------------------------------
        # No stretching
        # ------------------------------------------------------------
        if stretch is False:
    
            fig, ax = plt.subplots(
                1,
                4,
                figsize=[12, 5],
                gridspec_kw={'width_ratios': [1, 1, 1, 3]}
            )
    
            for c in range(3):
    
                values = self._finite_values(
                    BR_raw[:, :, c],
                    name=labels[c]
                )
    
                ax[c].hist(
                    values.ravel(),
                    bins,
                    color=colors[c],
                    alpha=0.5,
                    density=True,
                    cumulative=cumhist
                )
    
                ax[c].set_xlabel(
                    labels[c],
                    color=colors[c],
                    fontsize=9
                )
    
            RGB_display = BR_raw.copy()
            RGB_display[~np.isfinite(RGB_display)] = 0.0
            RGB_display = np.clip(RGB_display, 0.0, 1.0)
    
            ax[3].imshow(RGB_display)
            ax[3].set_title("Band ratio RGB raw, clipped for display")
            ax[3].axis("off")
    
            plt.show()
    
            self.RGB_final = RGB_display.copy()
            self.BR_final = BR_raw.copy()
    
            return self.BR_final, None
    
        # ------------------------------------------------------------
        # Stretching mode
        # ------------------------------------------------------------
        fig, ax = plt.subplots(
            1,
            4,
            figsize=[12, 5],
            gridspec_kw={'width_ratios': [1, 1, 1, 3]}
        )
    
        plt.subplots_adjust(top=0.78)
    
        if init_R is None:
            init_R = self._percentile_range(
                BR_raw[:, :, 0],
                stretch_percentiles
            )
    
        if init_G is None:
            init_G = self._percentile_range(
                BR_raw[:, :, 1],
                stretch_percentiles
            )
    
        if init_B is None:
            init_B = self._percentile_range(
                BR_raw[:, :, 2],
                stretch_percentiles
            )
    
        if R_min_in is None:
            R_min_in = self._percentile_range(
                BR_raw[:, :, 0],
                slider_percentiles
            )
    
        if R_max_in is None:
            R_max_in = self._percentile_range(
                BR_raw[:, :, 0],
                slider_percentiles
            )
    
        if G_min_in is None:
            G_min_in = self._percentile_range(
                BR_raw[:, :, 1],
                slider_percentiles
            )
    
        if G_max_in is None:
            G_max_in = self._percentile_range(
                BR_raw[:, :, 1],
                slider_percentiles
            )
    
        if B_min_in is None:
            B_min_in = self._percentile_range(
                BR_raw[:, :, 2],
                slider_percentiles
            )
    
        if B_max_in is None:
            B_max_in = self._percentile_range(
                BR_raw[:, :, 2],
                slider_percentiles
            )
    
        init_min_R, init_max_R = float(init_R[0]), float(init_R[1])
        init_min_G, init_max_G = float(init_G[0]), float(init_G[1])
        init_min_B, init_max_B = float(init_B[0]), float(init_B[1])
    
        R_min_in = [
            min(float(R_min_in[0]), init_min_R),
            max(float(R_min_in[1]), init_min_R)
        ]
    
        R_max_in = [
            min(float(R_max_in[0]), init_max_R),
            max(float(R_max_in[1]), init_max_R)
        ]
    
        G_min_in = [
            min(float(G_min_in[0]), init_min_G),
            max(float(G_min_in[1]), init_min_G)
        ]
    
        G_max_in = [
            min(float(G_max_in[0]), init_max_G),
            max(float(G_max_in[1]), init_max_G)
        ]
    
        B_min_in = [
            min(float(B_min_in[0]), init_min_B),
            max(float(B_min_in[1]), init_min_B)
        ]
    
        B_max_in = [
            min(float(B_max_in[0]), init_max_B),
            max(float(B_max_in[1]), init_max_B)
        ]
    
        hists = []
        bins_list = []
    
        for c in range(3):
    
            values = self._finite_values(
                BR_raw[:, :, c],
                name=labels[c]
            )
    
            h, bi, _ = ax[c].hist(
                values.ravel(),
                bins,
                color=colors[c],
                alpha=0.5,
                density=True,
                cumulative=cumhist
            )
    
            hists.append(h)
            bins_list.append(bi)
    
            ax[c].set_xlabel(
                labels[c],
                color=colors[c],
                fontsize=9
            )
    
            ax[c].set_xlim(bi[0], bi[-1])
    
            finite_h = h[np.isfinite(h)]
    
            if finite_h.size > 1:
                ymax = np.max(np.sort(finite_h)[:-1])
            elif finite_h.size == 1:
                ymax = finite_h[0]
            else:
                ymax = 1
    
            if ymax <= 0:
                ymax = 1
    
            ax[c].set_ylim(0, ymax * 1.05)
    
        RGB_final = self.f(
            BR_raw,
            init_min_R,
            init_min_G,
            init_min_B,
            init_max_R,
            init_max_G,
            init_max_B,
            clip=clip
        )
    
        final_min_R = init_min_R
        final_min_G = init_min_G
        final_min_B = init_min_B
    
        final_max_R = init_max_R
        final_max_G = init_max_G
        final_max_B = init_max_B
    
        im = ax[3].imshow(RGB_final)
        ax[3].set_title("Band ratio RGB stretched")
        ax[3].axis("off")
    
        vr = ax[0].axvline(
            init_min_R,
            color='grey',
            linestyle='--',
            linewidth=0.6
        )
    
        vR = ax[0].axvline(
            init_max_R,
            color='r',
            linestyle='--',
            linewidth=0.6
        )
    
        vg = ax[1].axvline(
            init_min_G,
            color='grey',
            linestyle='--',
            linewidth=0.6
        )
    
        vG = ax[1].axvline(
            init_max_G,
            color='g',
            linestyle='--',
            linewidth=0.6
        )
    
        vb = ax[2].axvline(
            init_min_B,
            color='grey',
            linestyle='--',
            linewidth=0.6
        )
    
        vB = ax[2].axvline(
            init_max_B,
            color='b',
            linestyle='--',
            linewidth=0.6
        )
    
        areaR = self._area_from_hist(
            hists[0],
            bins_list[0],
            init_min_R,
            init_max_R
        )
    
        areaG = self._area_from_hist(
            hists[1],
            bins_list[1],
            init_min_G,
            init_max_G
        )
    
        areaB = self._area_from_hist(
            hists[2],
            bins_list[2],
            init_min_B,
            init_max_B
        )
    
        textR = ax[0].set_title(
            f"{areaR[0]:.2f}%" + " - " + f"{areaR[1]:.2f}%",
            color='r',
            fontsize=8
        )
    
        textG = ax[1].set_title(
            f"{areaG[0]:.2f}%" + " - " + f"{areaG[1]:.2f}%",
            color='g',
            fontsize=8
        )
    
        textB = ax[2].set_title(
            f"{areaB[0]:.2f}%" + " - " + f"{areaB[1]:.2f}%",
            color='b',
            fontsize=8
        )
    
        # Red sliders
        ax_min_R = plt.axes([
            0.05,
            0.90,
            slider_width,
            slider_height
        ])
    
        ax_max_R = plt.axes([
            0.05,
            0.90 + slider_spacing,
            slider_width,
            slider_height
        ])
    
        min_R_slider = Slider(
            ax_min_R,
            label='min R',
            valmin=R_min_in[0],
            valmax=R_min_in[1],
            valinit=init_min_R,
            valstep=slider_step,
            color='r'
        )
    
        max_R_slider = Slider(
            ax_max_R,
            label='max R',
            valmin=R_max_in[0],
            valmax=R_max_in[1],
            valinit=init_max_R,
            valstep=slider_step,
            color='r'
        )
    
        # Green sliders
        ax_min_G = plt.axes([
            0.37,
            0.90,
            slider_width,
            slider_height
        ])
    
        ax_max_G = plt.axes([
            0.37,
            0.90 + slider_spacing,
            slider_width,
            slider_height
        ])
    
        min_G_slider = Slider(
            ax_min_G,
            label='min G',
            valmin=G_min_in[0],
            valmax=G_min_in[1],
            valinit=init_min_G,
            valstep=slider_step,
            color='g'
        )
    
        max_G_slider = Slider(
            ax_max_G,
            label='max G',
            valmin=G_max_in[0],
            valmax=G_max_in[1],
            valinit=init_max_G,
            valstep=slider_step,
            color='g'
        )
    
        # Blue sliders
        ax_min_B = plt.axes([
            0.69,
            0.90,
            slider_width,
            slider_height
        ])
    
        ax_max_B = plt.axes([
            0.69,
            0.90 + slider_spacing,
            slider_width,
            slider_height
        ])
    
        min_B_slider = Slider(
            ax_min_B,
            label='min B',
            valmin=B_min_in[0],
            valmax=B_min_in[1],
            valinit=init_min_B,
            valstep=slider_step,
            color='b'
        )
    
        max_B_slider = Slider(
            ax_max_B,
            label='max B',
            valmin=B_max_in[0],
            valmax=B_max_in[1],
            valinit=init_max_B,
            valstep=slider_step,
            color='b'
        )
    
        def update_RGB(event):
            nonlocal RGB_final
            nonlocal final_min_R, final_min_G, final_min_B
            nonlocal final_max_R, final_max_G, final_max_B
    
            min_R = min_R_slider.val
            max_R = max_R_slider.val
    
            min_G = min_G_slider.val
            max_G = max_G_slider.val
    
            min_B = min_B_slider.val
            max_B = max_B_slider.val
    
            final_min_R = min_R
            final_min_G = min_G
            final_min_B = min_B
    
            final_max_R = max_R
            final_max_G = max_G
            final_max_B = max_B
    
            RGB_final = self.f(
                BR_raw,
                min_R,
                min_G,
                min_B,
                max_R,
                max_G,
                max_B,
                clip=clip
            )
    
            im.set_data(RGB_final)
    
            vr.set_xdata([min_R])
            vR.set_xdata([max_R])
    
            vg.set_xdata([min_G])
            vG.set_xdata([max_G])
    
            vb.set_xdata([min_B])
            vB.set_xdata([max_B])
    
            areaR = self._area_from_hist(
                hists[0],
                bins_list[0],
                min_R,
                max_R
            )
    
            areaG = self._area_from_hist(
                hists[1],
                bins_list[1],
                min_G,
                max_G
            )
    
            areaB = self._area_from_hist(
                hists[2],
                bins_list[2],
                min_B,
                max_B
            )
    
            textR.set_text(
                f"{areaR[0]:.2f}%" + " - " + f"{areaR[1]:.2f}%"
            )
    
            textG.set_text(
                f"{areaG[0]:.2f}%" + " - " + f"{areaG[1]:.2f}%"
            )
    
            textB.set_text(
                f"{areaB[0]:.2f}%" + " - " + f"{areaB[1]:.2f}%"
            )
    
            fig.canvas.draw_idle()
    
        for slider in [
            min_R_slider, max_R_slider,
            min_G_slider, max_G_slider,
            min_B_slider, max_B_slider
        ]:
            slider.on_changed(update_RGB)
    
        plt.show()
    
        self.RGB_final = RGB_final.copy()
        self.BR_final = RGB_final.copy()
    
        return self.BR_final, [
            final_min_R,
            final_min_G,
            final_min_B,
            final_max_R,
            final_max_G,
            final_max_B
        ]

    def _smooth_cube_spectra(self, cube, smooth=False, t='savgol', w=5, o=2):
        """
        Smooth spectra along the spectral axis.
    
        Parameters
        ----------
        cube : np.ndarray
            Array with shape (rows, cols, bands).
    
        smooth : bool
            If False, returns cube unchanged.
    
        t : str
            Smoothing type:
            - 'savgol'
            - 'movmean'
    
        w : int
            Window size.
    
        o : int
            Polynomial order for Savitzky-Golay.
    
        Returns
        -------
        cube_smooth : np.ndarray
            Smoothed cube.
        """
    
        cube = np.asarray(cube).astype(float)
    
        if smooth is False:
            return cube.copy()
    
        if w < 1:
            raise ValueError("Smoothing window w must be at least 1.")
    
        if t == 'movmean':
    
            window_size = int(w)
            half_window = window_size // 2
    
            padded = np.pad(
                cube,
                pad_width=((0, 0), (0, 0), (half_window, half_window)),
                mode='constant',
                constant_values=np.nan
            )
    
            cube_sum = np.zeros_like(cube, dtype=float)
            cube_count = np.zeros_like(cube, dtype=float)
    
            for k in range(window_size):
    
                part = padded[:, :, k:k + cube.shape[2]]
                valid = np.isfinite(part)
    
                cube_sum += np.where(valid, part, 0.0)
                cube_count += valid.astype(float)
    
            cube_smooth = cube_sum / cube_count
            cube_smooth[cube_count == 0] = np.nan
    
            return cube_smooth
    
        elif t == 'savgol':
    
            if w % 2 == 0:
                w = w + 1
                print("Savitzky-Golay window must be odd. Using w =", w)
    
            if w > cube.shape[2]:
                w = cube.shape[2]
    
                if w % 2 == 0:
                    w = w - 1
    
                print("Savitzky-Golay window reduced to w =", w)
    
            if w <= o:
                raise ValueError(
                    "For Savitzky-Golay smoothing, window size w must be "
                    "greater than polynomial order o."
                )
    
            if w < 3:
                raise ValueError(
                    "Savitzky-Golay smoothing requires a window of at least 3."
                )
    
            # Savitzky-Golay does not handle NaNs well.
            # So we interpolate NaNs along the spectral axis before filtering.
            rows, cols, bands = cube.shape
            flat = cube.reshape(-1, bands)
            flat_filled = flat.copy()
    
            x = np.arange(bands)
    
            for i in range(flat.shape[0]):
    
                y = flat[i, :]
                valid = np.isfinite(y)
    
                if np.sum(valid) == 0:
                    continue
    
                if np.sum(valid) == 1:
                    flat_filled[i, :] = y[valid][0]
                    continue
    
                if not np.all(valid):
                    flat_filled[i, :] = np.interp(
                        x,
                        x[valid],
                        y[valid]
                    )
    
            filled_cube = flat_filled.reshape(rows, cols, bands)
    
            cube_smooth = savgol_filter(
                filled_cube,
                window_length=w,
                polyorder=o,
                axis=2,
                mode='interp'
            )
    
            # Restore pixels that were fully invalid
            all_invalid = ~np.any(np.isfinite(cube), axis=2)
            cube_smooth[all_invalid, :] = np.nan
    
            return cube_smooth
    
        else:
    
            raise ValueError("t must be either 'savgol' or 'movmean'.")
    
    
    def _compute_nearest_minimum_map(self, target_wavelength,
                                     search_window_nm=100,
                                     smooth=False,
                                     t='savgol',
                                     w=5,
                                     o=2,
                                     local_minimum=True,
                                     local_order=1,
                                     invalid_reflectance_threshold=1.0):
        """
        Compute a nearest-minimum wavelength map.
    
        For each pixel, this function searches around target_wavelength and returns
        the wavelength of the closest local minimum.
    
        If local_minimum=False, it returns the absolute minimum inside the search
        window.
    
        Returns
        -------
        min_wavelength_map : 2D np.ndarray
            Each pixel contains the wavelength position of the nearest minimum.
    
        min_value_map : 2D np.ndarray
            Each pixel contains the reflectance value at that minimum.
    
        info : dict
            Information about the search.
        """
    
        wavelengths = np.asarray(self.w).astype(float)
    
        lower = float(target_wavelength) - float(search_window_nm)
        upper = float(target_wavelength) + float(search_window_nm)
    
        band_mask = (wavelengths >= lower) & (wavelengths <= upper)
    
        if np.sum(band_mask) < 3:
            raise ValueError(
                "The selected wavelength window contains fewer than 3 bands. "
                "Increase search_window_nm."
            )
    
        band_indices = np.where(band_mask)[0].tolist()
        w_sub = wavelengths[band_indices]
        
        cube = np.array(self.img[:, :, band_indices]).astype(float)
    
        if invalid_reflectance_threshold is not None:
            cube[cube > invalid_reflectance_threshold] = np.nan
    
        cube[~np.isfinite(cube)] = np.nan
    
        cube_smooth = self._smooth_cube_spectra(
            cube,
            smooth=smooth,
            t=t,
            w=w,
            o=o
        )
    
        valid_any = np.any(np.isfinite(cube_smooth), axis=2)
    
        cube_for_min = np.where(
            np.isfinite(cube_smooth),
            cube_smooth,
            np.inf
        )
    
        absolute_min_index = np.argmin(cube_for_min, axis=2)
    
        if local_minimum is True and cube_smooth.shape[2] >= (2 * local_order + 1):
    
            local_mask = np.isfinite(cube_smooth)
    
            for shift in range(1, local_order + 1):
    
                left = np.full_like(cube_smooth, np.nan)
                right = np.full_like(cube_smooth, np.nan)
    
                left[:, :, shift:] = cube_smooth[:, :, :-shift]
                right[:, :, :-shift] = cube_smooth[:, :, shift:]
    
                local_mask &= cube_smooth <= left
                local_mask &= cube_smooth <= right
    
            local_mask[:, :, :local_order] = False
            local_mask[:, :, -local_order:] = False
    
            distance_from_target = np.abs(w_sub - float(target_wavelength))
    
            local_score = np.where(
                local_mask,
                distance_from_target[np.newaxis, np.newaxis, :],
                np.inf
            )
    
            has_local_minimum = np.any(np.isfinite(local_score), axis=2)
    
            local_min_index = np.argmin(local_score, axis=2)
    
            final_index = np.where(
                has_local_minimum,
                local_min_index,
                absolute_min_index
            )
    
        else:
    
            final_index = absolute_min_index
    
        min_wavelength_map = w_sub[final_index].astype(float)
    
        min_value_map = np.take_along_axis(
            cube_smooth,
            final_index[:, :, np.newaxis],
            axis=2
        )[:, :, 0].astype(float)
    
        min_wavelength_map[~valid_any] = np.nan
        min_value_map[~valid_any] = np.nan
    
        info = {
            "target_wavelength": target_wavelength,
            "search_window_nm": search_window_nm,
            "search_lower_nm": lower,
            "search_upper_nm": upper,
            "used_wavelengths": w_sub,
            "used_band_indices": band_indices,
            "smooth": smooth,
            "smooth_type": t,
            "smooth_window": w,
            "savgol_order": o,
            "local_minimum": local_minimum,
            "local_order": local_order
        }
    
        return min_wavelength_map, min_value_map, info

    def NM_BWmapmake(self, target_wavelength,
                     search_window_nm=100,
                     bins=500,
                     cumhist=False,
                     smooth=False,
                     t='savgol',
                     w=5,
                     o=2,
                     local_minimum=True,
                     local_order=1,
                     invalid_reflectance_threshold=1.0,
                     show=True,
                     colors=("yellow", "blue", "red")):
        """
        Create a BW nearest-minimum wavelength map.
    
        Each pixel contains the wavelength, in nm, of the minimum closest to
        target_wavelength.
    
        No stretching is applied: the output is always the raw wavelength-position
        map in nm.
    
        Color convention
        ----------------
        - lower wavelengths than target_wavelength are shown toward yellow
        - target_wavelength is shown as blue
        - higher wavelengths than target_wavelength are shown toward red
    
        Parameters
        ----------
        target_wavelength : float
            Expected minimum position, e.g. 1900.
    
        search_window_nm : float
            Half-width of the search interval in nm.
            Example: target_wavelength=1900 and search_window_nm=120 searches
            between 1780 and 2020 nm.
    
        bins : int
            Number of histogram bins.
    
        cumhist : bool
            If True, use cumulative histogram.
    
        smooth : bool
            If True, smooth spectra before minimum search.
    
        t : str
            Smoothing type: 'savgol' or 'movmean'.
    
        w : int
            Smoothing window.
    
        o : int
            Savitzky-Golay polynomial order.
    
        local_minimum : bool
            If True, search local minima first and choose the one closest to
            target_wavelength. If no local minimum is found, use the absolute
            minimum in the search window.
    
        local_order : int
            Number of neighboring bands on each side used to define a local minimum.
    
        invalid_reflectance_threshold : float or None
            If not None, reflectance values greater than this threshold are set to NaN.
    
        show : bool
            If True, show the map and histogram.
    
        Returns
        -------
        NM_final : 2D np.ndarray
            Raw nearest-minimum wavelength map in nm.
    
        NM_value : 2D np.ndarray
            Reflectance value at the selected minimum.
        """
    
        NM_raw, NM_value, info = self._compute_nearest_minimum_map(
            target_wavelength=target_wavelength,
            search_window_nm=search_window_nm,
            smooth=smooth,
            t=t,
            w=w,
            o=o,
            local_minimum=local_minimum,
            local_order=local_order,
            invalid_reflectance_threshold=invalid_reflectance_threshold
        )
    
        title = f"Nearest minimum around {target_wavelength:.0f} nm"
    
        print("Target wavelength:", info["target_wavelength"])
        print("Search interval:", info["search_lower_nm"], "-", info["search_upper_nm"], "nm")
        print("Number of bands used:", len(info["used_band_indices"]))
        print("First used wavelength:", info["used_wavelengths"][0])
        print("Last used wavelength:", info["used_wavelengths"][-1])
    
        self.NM_raw = NM_raw.copy()
        self.NM_value = NM_value.copy()
        self.NM_info = info
    
        # Now NM_final is always the scientific raw wavelength map in nm.
        self.NM_final = NM_raw.copy()
        self.BW_final = NM_raw.copy()
    
        values = self._finite_values(NM_raw, name=title)
    
        if show is True:
    
            fig, ax = plt.subplots(
                1,
                2,
                figsize=[10, 5],
                gridspec_kw={'width_ratios': [2, 3]}
            )
    
            hW, biW, _ = ax[0].hist(
                values.ravel(),
                bins,
                color='k',
                alpha=0.5,
                density=True,
                cumulative=cumhist
            )
    
            ax[0].set_xlabel("Minimum wavelength [nm]", color='black', fontsize=12)
            ax[0].set_xlim(biW[0], biW[-1])
    
            finite_h = hW[np.isfinite(hW)]
    
            if finite_h.size > 1:
                ymax = np.max(np.sort(finite_h)[:-1])
            elif finite_h.size == 1:
                ymax = finite_h[0]
            else:
                ymax = 1
    
            if ymax <= 0:
                ymax = 1
    
            ax[0].set_ylim(0, ymax * 1.05)
    
            lower = float(info["search_lower_nm"])
            upper = float(info["search_upper_nm"])
            center = float(target_wavelength)
    
            nm_cmap = self._nearest_minimum_colormap(colors=colors)
    
            norm = TwoSlopeNorm(
                vmin=lower,
                vcenter=center,
                vmax=upper
            )
    
            im = ax[1].imshow(
                NM_raw,
                cmap=nm_cmap,
                norm=norm
            )
    
            ax[1].set_title(title)
            ax[1].axis("off")
    
            cbar = plt.colorbar(
                im,
                ax=ax[1],
                fraction=0.046,
                pad=0.04
            )
    
            cbar.set_label("Minimum wavelength [nm]")
    
            cbar.set_ticks([
                lower,
                center,
                upper
            ])
    
            cbar.set_ticklabels([
                f"{lower:.0f}",
                f"{center:.0f}",
                f"{upper:.0f}"
            ])
    
            plt.show()
    
        return self.NM_final, self.NM_value

    def save_NMmap(self, name, folder=None, array=None, colors=("cyan", "black", "orange"), vmin=None, vcenter=None, vmax=None, save_txt=True, save_raw=True, save_colored=True, save_png=False, show=False):
        """
        Save a nearest-minimum wavelength map.
    
        This function is intended for maps produced by NM_BWmapmake().
    
        It can save:
            - raw wavelength map as txt
            - raw wavelength map as single-band float32 GeoTIFF
            - colored wavelength map as RGB/RGBA GeoTIFF
            - optional colored PNG
    
        Parameters
        ----------
        name : str
            Output base name.
    
        folder : str or None
            Output folder. Default is None.
    
        array : np.ndarray or None
            Nearest-minimum wavelength map in nm.
            If None, uses self.NM_raw. DEfault is None
    
        colors : tuple of 3 colors
            Colors in the order:
                lower wavelengths, center wavelength, higher wavelengths. Default is cyan, black, orange.
    
        vmin : float or None
            Lower wavelength for color scaling.
            If None, uses self.NM_info['search_lower_nm'] if available. Default is None.
    
        vcenter : float or None
            Center wavelength for color scaling.
            If None, uses self.NM_info['target_wavelength'] if available. Default is None.
    
        vmax : float or None
            Upper wavelength for color scaling.
            If None, uses self.NM_info['search_upper_nm'] if available. Default is None.
    
        save_txt : bool
            Save raw wavelength map as txt. Default is True.
    
        save_raw : bool
            Save raw wavelength map as single-band float32 GeoTIFF. Default is True.
    
        save_colored : bool
            Save colored map as RGB/RGBA GeoTIFF. Default is True.
    
        save_png : bool
            Save colored PNG. Default is False.
    
        show : bool
            Show PNG preview. Default is True.
    
        Returns
        -------
        None
        """
    
        import os
    
        if array is None:
            if not hasattr(self, "NM_raw") or self.NM_raw is None:
                raise ValueError("No NM_raw found. Run NM_BWmapmake() first or pass array=...")
            NM = np.asarray(self.NM_raw).astype(float)
        else:
            NM = np.asarray(array).astype(float)
    
        if NM.ndim == 3 and NM.shape[2] == 1:
            NM = NM[:, :, 0]
    
        if NM.ndim != 2:
            raise ValueError("Nearest-minimum map must have shape (rows, cols).")
    
        output_folder = folder if folder is not None else "."
        os.makedirs(output_folder, exist_ok=True)
    
        base_name = os.path.splitext(os.path.basename(name))[0]
    
        # ------------------------------------------------------------
        # Color limits
        # ------------------------------------------------------------
        if hasattr(self, "NM_info"):
    
            if vmin is None and "search_lower_nm" in self.NM_info:
                vmin = float(self.NM_info["search_lower_nm"])
    
            if vcenter is None and "target_wavelength" in self.NM_info:
                vcenter = float(self.NM_info["target_wavelength"])
    
            if vmax is None and "search_upper_nm" in self.NM_info:
                vmax = float(self.NM_info["search_upper_nm"])
    
        values = NM[np.isfinite(NM)]
    
        if values.size == 0:
            raise ValueError("No finite values found in nearest-minimum map.")
    
        if vmin is None:
            vmin = float(np.nanmin(values))
    
        if vmax is None:
            vmax = float(np.nanmax(values))
    
        if vcenter is None:
            vcenter = 0.5 * (vmin + vmax)
    
        if not (vmin < vcenter < vmax):
            raise ValueError("Color limits must satisfy vmin < vcenter < vmax.")
    
        if len(colors) != 3:
            raise ValueError("colors must contain exactly three colors: low, center, high.")
    
        # ------------------------------------------------------------
        # Georeference
        # ------------------------------------------------------------
        metadata = self.img_sr.metadata
    
        if "map info" not in metadata:
            raise ValueError("No 'map info' found in self.img_sr.metadata. Cannot georeference GeoTIFF.")
    
        if "coordinate system string" not in metadata:
            raise ValueError("No 'coordinate system string' found in self.img_sr.metadata. Cannot assign CRS to GeoTIFF.")
    
        map_info = metadata["map info"]
        wkt = metadata["coordinate system string"]
    
        if isinstance(map_info, str):
            map_info = map_info.strip("{}").split(",")
            map_info = [v.strip() for v in map_info]
    
        if isinstance(wkt, list):
            wkt = ",".join(wkt)
    
        wkt = str(wkt).strip("{}")
    
        ref_pixel_x = float(map_info[1])
        ref_pixel_y = float(map_info[2])
        map_x = float(map_info[3])
        map_y = float(map_info[4])
        pixel_size_x = float(map_info[5])
        pixel_size_y = float(map_info[6])
    
        x_origin = map_x - (ref_pixel_x - 0.5) * pixel_size_x
        y_origin = map_y + (ref_pixel_y - 0.5) * pixel_size_y
    
        transform = from_origin(x_origin, y_origin, pixel_size_x, pixel_size_y)
        crs = CRS.from_wkt(wkt)
    
        # ------------------------------------------------------------
        # Save txt
        # ------------------------------------------------------------
        if save_txt is True:
            np.savetxt(os.path.join(output_folder, base_name + "_nm.txt"), NM)
    
        # ------------------------------------------------------------
        # Save raw scientific GeoTIFF
        # ------------------------------------------------------------
        if save_raw is True:
    
            raw_path = os.path.join(output_folder, base_name + "_raw.tif")
    
            with rasterio.open(
                raw_path,
                "w",
                driver="GTiff",
                height=NM.shape[0],
                width=NM.shape[1],
                count=1,
                dtype="float32",
                crs=crs,
                transform=transform,
                nodata=np.nan
            ) as dst:
                dst.write(NM.astype("float32"), 1)
    
        # ------------------------------------------------------------
        # Prepare colored map
        # ------------------------------------------------------------
        cmap = LinearSegmentedColormap.from_list(
            "nearest_minimum_colormap",
            [
                (0.0, colors[0]),
                (0.5, colors[1]),
                (1.0, colors[2])
            ]
        )
    
        norm = TwoSlopeNorm(
            vmin=vmin,
            vcenter=vcenter,
            vmax=vmax
        )
    
        valid_mask = np.isfinite(NM)
    
        rgba = cmap(norm(NM))
        rgb = rgba[:, :, :3]
    
        rgb[~valid_mask, :] = 0.0
    
        rgb_uint8 = (np.clip(rgb, 0.0, 1.0) * 255).round().astype("uint8")
        alpha = np.where(valid_mask, 255, 0).astype("uint8")
    
        # ------------------------------------------------------------
        # Save colored GeoTIFF
        # ------------------------------------------------------------
        if save_colored is True:
    
            colored_path = os.path.join(output_folder, base_name + "_colored.tif")
    
            with rasterio.open(
                colored_path,
                "w",
                driver="GTiff",
                height=rgb_uint8.shape[0],
                width=rgb_uint8.shape[1],
                count=4,
                dtype="uint8",
                crs=crs,
                transform=transform
            ) as dst:
    
                dst.write(rgb_uint8[:, :, 0], 1)
                dst.write(rgb_uint8[:, :, 1], 2)
                dst.write(rgb_uint8[:, :, 2], 3)
                dst.write(alpha, 4)
    
                dst.colorinterp = (
                    ColorInterp.red,
                    ColorInterp.green,
                    ColorInterp.blue,
                    ColorInterp.alpha
                )
    
        # ------------------------------------------------------------
        # Save colored PNG
        # ------------------------------------------------------------
        if save_png is True:
    
            png_path = os.path.join(output_folder, base_name + "_colored.png")
    
            plt.figure()
            plt.imshow(NM, cmap=cmap, norm=norm)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(png_path, bbox_inches="tight", pad_inches=0, transparent=True)
    
            if show is True:
                plt.show()
            else:
                plt.close()
    
        return
