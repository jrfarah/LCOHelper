"""
====================================

* **Filename**:         spectral.py 
* **Author**:              Joseph Farah 
* **Description**:       Module for manipulating, plotting, fitting spectra.

====================================

**Notes**
*  
"""

#------------- imports -------------#
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import simps, trapz

## matplotlib settings ##
import smplotlib
import matplotlib.pyplot as plt
smplotlib.set_style(fontweight='normal', usetex=False, fontsize=15, figsize=(6, 6), dpi=120)



#------------- classes -------------#
class LCOSpectrum(object):

    def __init__(self, spectrum):
        if type(spectrum) == str:
            self.spectrum = pd.read_csv(spectrum, header=None, names=['wl', 'flux', 'err'], delimiter=' ')
        else:
            self.spectrum = spectrum


    def _moving_average2(self, x, y, n=250):
        ret = np.cumsum(y, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return x[n//2:-n//2 +1], ret[n - 1:] / n



    def plot(self, average=True, showboth=True, binsize=50):

        if showboth:
            plt.plot(self.spectrum.wl, self.spectrum.flux, c='blue', alpha=0.2)
            wl, flux = self._moving_average2(list(self.spectrum.wl), list(self.spectrum.flux), n=binsize)
            plt.plot(wl, flux, c='red')

        if not showboth:
            if average:
                wl, flux = self._moving_average2(list(self.spectrum.wl), list(self.spectrum.flux), n=binsize)
                plt.plot(wl, flux, c='red')
            else:
                plt.plot(self.spectrum.wl, self.spectrum.flux, c='blue', alpha=0.2)

        plt.xlabel("Wavelength (angstroms)")
        plt.ylabel("Intensity")

        plt.show()


    def fit_line_gaussian(self, min_wl=None, max_wl=None, show=True):   

        if min_wl is None and max_wl is None:
            input("We will now show the spectrum. Please consult and choose the wavelength range you wish to fit.")
            self.plot()
            min_wl = float(input("Minimum wavelength range: \n >>>"))
            max_wl = float(input("Maximum wavelength range: \n >>>"))

        _, min_wl_index = self.__find_nearest(self.spectrum.wl, min_wl)
        _, max_wl_index = self.__find_nearest(self.spectrum.wl, max_wl)


        gauss_popt, _ = self.fit_gaussian_with_linear_shift(self.spectrum.wl[min_wl_index:max_wl_index], self.spectrum.flux[min_wl_index:max_wl_index]) 

        if show:
            plt.plot(self.spectrum.wl[min_wl_index:max_wl_index], self.spectrum.flux[min_wl_index:max_wl_index], c='blue', alpha=0.2)
            plt.plot(self.spectrum.wl[min_wl_index:max_wl_index], self.gaussian_with_linear_shift(self.spectrum.wl[min_wl_index:max_wl_index], *gauss_popt))
            plt.show()


        class ReturnObject(object):
            def __init__(self, popt, wl, flux):
                self.amp = popt[0]
                self.mean = popt[1]
                self.stddev = popt[2]
                self.slope = popt[3]
                self.intercept = popt[4]
                self.wl = wl
                self.flux = flux
                self.popt = popt

        return ReturnObject(gauss_popt, self.spectrum.wl[min_wl_index:max_wl_index], self.spectrum.flux[min_wl_index:max_wl_index])


    def equivalent_width(self, min_wl=None, max_wl=None, show=True):

        gauss_popt = self.fit_line_gaussian(min_wl=min_wl, max_wl=max_wl, show=show)
        continuum = np.array([gauss_popt.intercept + gauss_popt.slope * c for c in gauss_popt.wl])
        continuum_intensity_level = np.mean(continuum)

        gaussian_approx = np.array([self.gaussian_with_linear_shift(c, *gauss_popt.popt) for c in gauss_popt.wl])
        continuum_integral = simps(continuum, gauss_popt.wl)
        gauss_integral = simps(gaussian_approx, gauss_popt.wl)
        area = continuum_integral - gauss_integral

        if show:
            print(f"Equivalent width: {area / continuum_intensity_level} angstroms")

        return area / continuum_intensity_level




    def gaussian_with_linear_shift(self, x, amplitude, mean, stddev, slope, intercept):

        gaussian = amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
        linear_shift = slope * x + intercept
        return gaussian + linear_shift

    def fit_gaussian_with_linear_shift(self, x, y):

        amplitude_guess = np.min(y) if np.min(y) < 0 else np.max(y)
        mean_guess = np.mean(x)
        stddev_guess = np.std(x) / 2
        slope_guess = 0
        intercept_guess = np.mean(y)
        initial_guess = [amplitude_guess, mean_guess, stddev_guess, slope_guess, intercept_guess]
        
        popt, pcov = curve_fit(self.gaussian_with_linear_shift, x, y, p0=initial_guess)
        return popt, pcov




    
    def __find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx 




#------------- functions -------------#
def main():
    print("unit tests")
    spec = LCOSpectrum('./sn2022hnt_20220425_redblu_060846.551.ascii')
    spec.plot()
    # spec.fit_line_gaussian(min_wl=7934, max_wl=8796)
    spec.equivalent_width(min_wl=7934, max_wl=8796)


#------------- switchboard -------------#
if __name__ == '__main__':
    main()      