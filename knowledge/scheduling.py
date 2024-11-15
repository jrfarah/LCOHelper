"""
====================================
Filename:         scheduling.py 
Author:              Joseph Farah 
Description:       Helpful tools and info for scheduling.
====================================
Notes
     
"""

#------------- imports -------------#
import sys


#------------- classes -------------#
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


    @staticmethod
    def warning(message):
        print (bcolors.WARNING + "[" + message + "]" + bcolors.ENDC)

    @staticmethod
    def success(message):
        print (bcolors.OKGREEN + "[" + message + "]" + bcolors.ENDC)

    @staticmethod
    def failure(message):
        print (bcolors.FAIL + "[" + message + "]" + bcolors.ENDC)

    @staticmethod
    def with_color(message, color):
        print(color + message + bcolors.ENDC)


class Observation(object):

    def __init__(self, object_name, observation_type, last_mag):

        """
            Initialize observation object
        
            Args:
                object_name (str): name of object for recording
                observation_type (str): 'spec' or 'phot'
                last_mag (int/float): most recent apparent magnitude
        
        """
         
        

        self.object_name = object_name
        self.observation_type = observation_type
        self.last_mag = last_mag

    def recommend_exposures(self, output='pretty'):

        """
            Recommend exposures based on observation type and last mag.
        
            Args:
                output (str): options: 'pretty' for readability and 'utility' for future SNEx integration.
        
        """

        if output == 'pretty':
            self.__print_pretty__(self.observation_type, self.last_mag)
        else:
            self.__print_utility__(self.observation_type, self.last_mag)

    def __print_pretty__(self, typ, mag):

        rel_dict = MAG_DICTIONARY[typ][min(list(MAG_DICTIONARY[typ].keys()), key=lambda x:abs(x-mag))]

        if typ == 'phot':
            for key in list(rel_dict.keys()):
                bcolors.with_color(f"{key} band: {rel_dict[key]} seconds", COLORS_DICT[key])

        elif typ == 'spec':
            print(f"Exposure required for spectroscopy at {mag} mag: {bcolors.OKGREEN} {rel_dict} seconds {bcolors.ENDC}")


    def __print_utility__(typ, mag):

        rel_dict = MAG_DICTIONARY[typ][min(list(MAG_DICTIONARY[typ].keys()), key=lambda x:abs(x-mag))]

        if typ == 'phot':
            return typ+";"+';'.join([f"{key}:{rel_dict[key]}" for key in list(rel_dict.keys())])
             
            



#------------- knowledge -------------#

COLORS_DICT = {
    'U':bcolors.BOLD,
    'B':bcolors.OKBLUE,
    'g':bcolors.OKGREEN,
    'V':bcolors.ENDC,
    'r':bcolors.FAIL,
    'i':bcolors.WARNING
}

MAG_DICTIONARY = {
    'phot':{
        (20.5+19.5)/2.: {'U':-1, "B":400, "g":400, "V":300,"r":300,"i":300},
        (19.5+18.5)/2.: {'U':400, "B":300, "g":300, "V":200,"r":200,"i":200},
        (18.5+17.5)/2.: {'U':300, "B":200, "g":200, "V":120,"r":120,"i":120},
        (17.5+16.5)/2.: {'U':200, "B":120, "g":120, "V":90,"r":90,"i":90},
        (16.5+15.5)/2.: {'U':120, "B":90, "g":90, "V":60,"r":60,"i":60},
        (15.5+14.5)/2.: {'U':90, "B":60, "g":60, "V":40,"r":40,"i":40},
        (14.5+13.5)/2.: {'U':60, "B":40, "g":40, "V":20,"r":20,"i":20},
        (13.5+12.5)/2.: {'U':40, "B":20, "g":20, "V":10,"r":10,"i":10}
    },
    'spec':{
        (18.5+17.5)/2.: 3600 ,
        (17.5+16.5)/2.: 2700 ,
        (16.5+15.5)/2.: 1800 ,
        (15.5+14.5)/2.: 900 ,
        (14.5+13.5)/2.: 600 ,
        (13.5+12.5)/2.: 400 
    }
}


if __name__ == '__main__':
    
    #------------- unit tests -------------#
    ObservationTest = Observation('SN 1993J', 'spec', 18)
    ObservationTest.recommend_exposures()