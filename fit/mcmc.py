"""
====================================
Filename:         mcmc.py 
Author:              Joseph Farah 
Description:       Wrapper for simple and quick mcmc fits.
====================================
Notes
     
"""

#------------- imports -------------#
import copy
import emcee
import scipy
import corner
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

## matplotlib settings ##
import matplotlib.pyplot as plt
plt.style.use('classic')
import matplotlib as mpl
import matplotlib.font_manager as font_manager
mpl.rcParams['font.family']='serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif']=cmfont.get_name()
mpl.rcParams['mathtext.fontset']='cm'
mpl.rcParams['axes.unicode_minus']=False
colors = ['green', 'orange', 'cyan', 'darkred']
mpl.rcParams['figure.facecolor'] = '1.0'
plt.rcParams.update({'font.size': 14}) 




class LCOFitObject_MCMC(object):

    def __init__(self, xdata=None, ydata=None, sigma=None, labels_dict=None):

        """
            Initialize MCMC fit.
        
            Args:
                xdata,ydata,sigma (list/np.ndarray): data and errors
                labels_dict (dict): dictionary of labels and units; xdata denoted with key 'x' and ydata denoted with key 'y'; title denoted with 't'
        
            Returns:
                none (none): none
        
        """ 
        
        ## initialize data ##
        self.xdata = xdata
        self.ydata = ydata
        self.sigma = sigma
        self.merged_data = [self.xdata, self.ydata, self.sigma]

        ## initialize labels ##
        self.labels_dict = labels_dict
        if labels_dict is None:
            self.labels_dict = {}
            self.labels_dict['x'] = 'time'
            self.labels_dict['y'] = 'energy'
            self.labels_dict['t'] = 'Energy vs. time'
        else:
            self.labels_dict = labels_dict

        ## initialize fit function and priors ##
        self.model = None
        self.priors = None

        ## initial guess ##
        self.initial_guess = []
        self.dim = len(self.initial_guess)

        ## initialize MCMC settings ##
        self.p0 = None
        self.mcmc_settings = self.update_mcmc_settings(0, 0, restore_default=True)

        ## initialize results ##
        self.results = {'sampler':None, 'pos':None, 'prob':None, 'state':None }

    def plot_data(self, show=True, process_data=None):

        """
            Plots data.
        
            Args:
                none (none): none
        
            Returns:
                none (none): none
        
        """

        if process_data is not None:
            xdata, ydata = process_data(self.xdata, self.ydata)
        else:
            xdata = self.xdata
            ydata = self.ydata
        plt.errorbar(xdata, ydata, yerr=self.sigma, fmt='.')
        plt.xlabel(self.labels_dict['x'])
        plt.ylabel(self.labels_dict['y'])
        plt.title(self.labels_dict['t'])

        if show: plt.show()

    def add_fit_function(self, func):
        """
            Add a fit function to the object, which will be used in the MCMC.
        
            Args:
                func (function): fit function that will produce expected values. Function should take list of parameters as first arg, and xdata as optional argument 'x'; i.e., def func(theta, x=[]):
        
            Returns:
                none (none): none
        
        """

        self.model = func


    def add_priors(self, prior_dict):
        """
            Adds prior constraints assuming uniform priors
        
            Args:
                prior_dict (dict): prior dictionary with keys-> variable labels and values->[min_value, max_value]
        
            Returns:
                none (none): none
        
        """

        ## check each prior has min and max ##
        for key in list(prior_dict.keys()):
            if len(prior_dict[key]) == 2:
                continue
            else:
                print(f"Please make sure every prior has a min and max. (error thrown on key={key})")

        self.priors = prior_dict
        self.initial_guess = [1 for i in list(self.priors.keys())]
        self.dim = len(self.initial_guess)

    

    def update_mcmc_settings(self, setting, new_value, restore_default=False):
        """
             Update mcmc settings
         
             Args:
                 setting (str): setting name in dictionary
                 new_value (obj): new value you want to change to
                 restore_default (bool): restore all settings to default
         
             Returns:
                 none (none): none
         
         """

        if restore_default:

            self.mcmc_settings = {
                'NWALKERS': 500,
                'N_ITER': 200
            }
            self.p0 = [np.array(self.initial_guess) + 1e-7 * np.random.randn(len(self.initial_guess)) for i in range(self.mcmc_settings['NWALKERS'])]


            return self.mcmc_settings

        else:

            self.mcmc_settings[setting] = new_value
            self.p0 = [np.array(self.initial_guess) + 1e-7 * np.random.randn(len(self.initial_guess)) for i in range(self.mcmc_settings['NWALKERS'])]

            return self.mcmc_settings



    def add_initial_guess(self, guess):
        """
            Replaces default guess with custom guess.
        
            Args:
                guess (list): list of initial parameters
        
            Returns:
                none (none): none
        
        """
        self.initial_guess = guess
        self.p0 = [np.array(self.initial_guess) + 1e-7 * np.random.randn(len(self.initial_guess)) for i in range(self.mcmc_settings['NWALKERS'])]


    def generate_initial_guess(self): 
        """
            Generates initial guess using gradient descent fit
            TODO: use other LCOhelper function to do this
        
            Args:
                none (none): none
        
            Returns:
                none (none): none
        
        """
        initial = scipy.optimize.minimize(self._helper_chi2, self.initial_guess)

        print(f"Initial guess: {initial.x}")
        self.initial_guess = initial.x
        self.p0 = [np.array(self.initial_guess) + 1e-7 * np.random.randn(len(self.initial_guess)) for i in range(self.mcmc_settings['NWALKERS'])]
          
    

    def _log_likelihood(self, theta, x_d, y_d, sigma_y):
        """
            log likelihood helper function
        
            Args:
                theta (list): list of parameters   
        
            Returns:
                log_like (float): log likelihood
        
        """

        K = np.sum(25*np.log(1./(sigma_y*np.sqrt(2*np.pi))))
        chi2 = 0
        for idx, yy in enumerate(y_d):
            if len(np.shape(x_d)) > 1:
                xx = [i[idx] for i in x_d]
            else:
                xx = x_d[idx]
            try:
                chi2 += (1./(2*sigma_y[idx]**2.))*(yy - self.model(theta, x=xx))**2.
            except TypeError:
                chi2 += (1./(2*sigma_y**2.))*(yy - self.model(theta, x=xx))**2.
        log_like = K - chi2
        return log_like



    def _helper_chi2(self, theta):
        """
            chi2 helper function
        
            Args:
                theta (list): list of parameters

            Returns:
                chi2 (float): chi2
        
        """

        # return  -self._log_likelihood(theta, self.xdata, self.ydata, self.sigma)

        y_d = self.ydata
        x_d = self.xdata
        sigma_y = self.sigma

        if type(sigma_y) in [float, int]:
            sigma_y = [sigma_y for s in y_d]

        chi2 = 0
        for idx, yy in enumerate(y_d):
            if len(np.shape(x_d)) > 1:
                xx = [i[idx] for i in x_d]
            else:
                xx = x_d[idx]
            try:
                chi2 += (1./(2*sigma_y[idx]**2.))*(yy - self.model(theta, x=xx))**2.
            except TypeError:
                chi2 += (1./(2*sigma_y**2.))*(yy - self.model(theta, x=xx))**2.

        return np.sum(chi2)/len(x_d)



        
    def _prior(self, theta):
        """
            Function to ensure all variables are within prior
        
            Args:
                theta (list): list of parameters
        
            Returns:
                float (float): -np.inf if not in prior (to disfavor solution), 0 otherwise
        
        """
         
        

        within_priors = True
        for p, prior in enumerate(self.priors.keys()):
            param = theta[p]
            param_min = self.priors[prior]['min']
            param_max = self.priors[prior]['max']

            if param > param_max or param < param_min:
                within_priors = False

        if within_priors == False:
            return -np.inf

        else:
            return 0.0

    def _conditional_likelihood(self, theta, x_d, y_d, sigma_y):

        return self._prior(theta) + self._log_likelihood(theta, x_d, y_d, sigma_y)


    def run_MCMC(self):
        """
            Run MCMC function with previous settings and data, save to self.results
        
        """
        sampler, pos, prob, state = self._mcmc(self.p0, self.mcmc_settings['NWALKERS'], self.mcmc_settings['N_ITER'], self.dim, self._conditional_likelihood, self.merged_data)

        self.results['sampler'] = sampler
        self.results['pos'] = pos
        self.results['prob'] = prob
        self.results['state'] = state
         
        

    def _mcmc(self, p0, nwalkers,niter,ndim,lnprob,data):
        """
            MCMC function
        
            Args:
                arg1 (type): argument description
        
            Returns:
                return1 (type): return description
        
        """
         
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=self.merged_data, pool=pool)

            print("Running burn-in...")
            p0 = sampler.run_mcmc(p0, 100, progress=True)
            sampler.reset()

            print("Running production...")
            result = sampler.run_mcmc(p0, niter, progress=True)
        

        return sampler, result.coords, result.log_prob, result.random_state

    def _sort_by_indexes(self, lst, indexes, reverse=False):
        return [val for (_, val) in sorted(zip(indexes, lst), key=lambda x: \
              x[0], reverse=reverse)]

    def plot_results(self, process_data=None):
        if process_data is not None:
            xdata, ydata = process_data(self.xdata, self.ydata)
        else:
            xdata = self.xdata
            ydata = self.ydata
        sigma = self.sigma

        ## sort so plot will work ##
        xdata_copy = copy.copy(xdata)
        ydata = self._sort_by_indexes(ydata, xdata_copy)
        if type(self.sigma) == list:
            sigma = self._sort_by_indexes(sigma, xdata_copy)

        xdata = self._sort_by_indexes(xdata, xdata_copy)
        ## plot data
        plt.errorbar(xdata, ydata, yerr=sigma, fmt='.', markeredgecolor='black', ecolor='black')

        ## plot samples
        samples = self.results['sampler'].flatchain
        plt.plot([], [], color="r", alpha=1, label='posterior samples')
        for theta in samples[np.random.randint(len(samples), size=100)]:
            plt.plot(xdata, self.model(theta, self.xdata), color="r", alpha=0.05)

        ## plot best sample
        theta_max  = samples[np.argmax(self.results['sampler'].flatlnprobability)]
        best_fit_model = self.model(theta_max, self.xdata)
        plt.plot(xdata, best_fit_model,label='highest likelihood model', c='green', lw=2)
       
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.xlabel(self.labels_dict['x'])
        plt.ylabel(self.labels_dict['y'])
        plt.title(self.labels_dict['t'])
        plt.legend(loc='lower right', facecolor='white', frameon=False)
        plt.show()


    def retrieve_best_theta(self):

        samples = self.results['sampler'].flatchain
        return samples[np.argmax(self.results['sampler'].flatlnprobability)]

    def plot_corner(self):
        samples = self.results['sampler'].flatchain
        labels = [key for key in list(self.priors.keys())]

        fig = corner.corner(samples,
                    show_titles=True,
                    labels=labels,
                    plot_datapoints=True,
                    quantiles=[0.16, 0.5, 0.84], 
                    color='b', 
                    bins=15,
                    plot_contours=True, 
                    use_math_text=True,
                    plot_density=False,
                    no_fill_contours=True,
                    contour_kwargs={"colors":'black'}
        )

        plt.show()

    def plot_data_vs_model(self, theta, process_data=None, scatter=False):

        if process_data is not None:
            xdata, ydata = process_data(self.xdata, self.ydata)
        else:
            xdata = self.xdata
            ydata = self.ydata

        plt.errorbar(xdata, ydata, yerr=self.sigma, fmt='.')
        plt.xlabel(self.labels_dict['x'])
        plt.ylabel(self.labels_dict['y'])
        plt.title(self.labels_dict['t'])

        if scatter:
            plt.scatter(xdata, self.model(theta, x=self.xdata), c='red')
        else:
            plt.plot(xdata, self.model(theta, x=self.xdata), c='red')
        plt.show()







        
#------------- unit tests -------------#
def linear_test():
    ## generate fake data ##
    sigma = 3
    x = np.linspace(0, 10, 100)
    y = [xx + random.gauss(0, sigma) for xx in x]

    ## create object ##
    fit_o = LCOFitObject_MCMC(xdata=x, ydata=y, sigma=sigma)

    ## view data ##
    fit_o.plot_data(show=True)

    ## define priors ##
    PRIORS = {
        "alpha_0":{'min':0, 'max':10},
        "alpha_1":{'min':0, 'max':10},
    }   

    ## add priors ##
    fit_o.add_priors(PRIORS)

    ## define model ##
    def model_linearFunction(theta, x=None):
        alpha_0, alpha_1 = theta 

        return alpha_0 + np.array(x)*alpha_1

    ## add model ##
    fit_o.add_fit_function(model_linearFunction)

    ## generate initial guess ##
    fit_o.generate_initial_guess()

    ## perform fit ##
    fit_o.run_MCMC()

    ## plot samples ##
    fit_o.plot_results()

    ## plot corner ##
    fit_o.plot_corner()


def quad_test():
    ## generate fake data ##
    sigma = 15
    x = np.linspace(0, 10, 100)
    y = [xx**2. + random.gauss(0, sigma) for xx in x]

    ## create object ##
    fit_o = LCOFitObject_MCMC(xdata=x, ydata=y, sigma=sigma)

    ## view data ##
    fit_o.plot_data(show=True)

    ## define priors ##
    PRIORS = {
        "alpha_0":{'min':0, 'max':10},
        "alpha_1":{'min':0, 'max':10},
        "alpha_2":{'min':0, 'max':10},
    }   

    ## add priors ##
    fit_o.add_priors(PRIORS)

    ## define model ##
    def model_linearFunction(theta, x=None):
        alpha_0, alpha_1, alpha_2 = theta 

        return alpha_0 + np.array(x)*alpha_1 + (np.array(x)**2.)*alpha_2

    ## add model ##
    fit_o.add_fit_function(model_linearFunction)

    ## generate initial guess ##
    fit_o.generate_initial_guess()

    ## perform fit ##
    fit_o.run_MCMC()

    ## plot samples ##
    fit_o.plot_results()

    ## plot corner ##
    fit_o.plot_corner()


## define model ##
def model_sineFunction(theta, x=None):
    alpha_0, alpha_1, alpha_2, alpha_3 = theta 

    return alpha_0 * np.sin(alpha_1 * x + alpha_2) + alpha_3

def sine_test():

    ## generate fake data ##
    sigma = 0.25
    x = np.linspace(0, 10, 100)
    y = [np.sin(xx) + random.gauss(0, sigma) for xx in x]

    ## create object ##
    fit_o = LCOFitObject_MCMC(xdata=x, ydata=y, sigma=sigma)

    ## view data ##
    fit_o.plot_data(show=True)

    ## define priors ##
    PRIORS = {
        "alpha_0":{'min':-10, 'max':10},
        "alpha_1":{'min':-10, 'max':10},
        "alpha_2":{'min':-10, 'max':10},
        "alpha_3":{'min':-10, 'max':10},
    }   

    ## add priors ##
    fit_o.add_priors(PRIORS)



    ## add model ##
    fit_o.add_fit_function(model_sineFunction)

    ## generate initial guess ##
    fit_o.generate_initial_guess()
    print(fit_o._helper_chi2(fit_o.initial_guess))
    fit_o.plot_data_vs_model(fit_o.initial_guess)

    ## perform fit ##
    fit_o.run_MCMC()

    ## plot samples ##
    fit_o.plot_results()

    ## plot corner ##
    fit_o.plot_corner()

    print(fit_o._helper_chi2(fit_o.retrieve_best_theta()))


def twodim_x_test():
    ## generate fake data ##
    sigma = 0.25
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    z = [np.sin(np.sqrt(xx**2. + y[ix]**2.)) + random.gauss(0, sigma) for ix, xx in enumerate(x)]

    ## create object ##
    fit_o = LCOFitObject_MCMC(xdata=(x, y), ydata=z, sigma=sigma)

    ## view data ##
    def process_data(xdata, ydata):
        return xdata[0], ydata
    fit_o.plot_data(show=True, process_data=process_data)

    ## define priors ##
    PRIORS = {
        "alpha_0":{'min':-10, 'max':10},
        "alpha_1":{'min':-10, 'max':10},
        "alpha_2":{'min':-10, 'max':10},
        "alpha_3":{'min':-10, 'max':10},
    }   

    fit_o.add_priors(PRIORS)

    ## define model ##
    def model_sineFunction(theta, x=None):
        alpha_0, alpha_1, alpha_2, alpha_3 = theta 

        return alpha_0 * np.sin(alpha_1 * (np.sqrt(x[0]**2. + x[1]**2.)) + alpha_2) + alpha_3

    ## add model ##
    fit_o.add_fit_function(model_sineFunction)

    ## generate initial guess ##
    fit_o.generate_initial_guess()
    print(fit_o._helper_chi2(fit_o.initial_guess))
    fit_o.plot_data_vs_model(fit_o.initial_guess, process_data=process_data, scatter=True)

    ## perform fit ##
    fit_o.run_MCMC()

    ## plot samples ##
    fit_o.plot_results(process_data=process_data)

    ## plot corner ##
    fit_o.plot_corner()

    print(fit_o._helper_chi2(fit_o.retrieve_best_theta()))

if __name__ == '__main__':
    
    # linear_test()
    # quad_test()
    sine_test()
    # twodim_x_test()