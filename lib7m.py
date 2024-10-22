# A module to deal with the data format of the 7m data
import csv

# The numpy module for various data operations
import numpy as np

# This function returns a 7m scan loaded from a file named filename
def load_scan(filename):
    offsetlist = []
    fluxdensitylist = []
    print("Opening file %s" %(filename))
    input_file = open(filename,'r+')        #Open the input file
    input_reader = csv.reader(input_file, delimiter=',', quotechar='"')   #Initialise the reader for the CSV file format
    rownr = 0
    for input_row in input_reader:     #Loop over all lines (spectral bins) in the csv file
        offset = input_row[1]             #Second column (first column is index [0]) is the offset (in degrees)
        offsetlist.append(float(offset))
        #print("Offset=%f deg"%(float(offset)))
        fluxdensity = input_row[2]          #Third column is the corresponding flux density (uncalibrated)
        fluxdensitylist.append(float(fluxdensity))
        #print("Flux density=%f"%(float(fluxdensity)))
        rownr += 1
    input_file.close()
    print("  Done with reading in %d scan points" %(rownr))
    # Return the data as a list of two numpy arrays, corresponding to the offset (in degrees), and the flux density
    return [ np.array(offsetlist) , np.array(fluxdensitylist) ]

# This function returns a 7m spectrum loaded from a file named filename
def load_spectrum(filename):
    freqlist = []
    velocitylist = []
    temperaturelist = []
    print("Opening file %s" %(filename))
    input_file = open(filename,'r+')        #Open the input file
    input_reader = csv.reader(input_file, delimiter=',', quotechar='"')   #Initialise the reader for the CSV file format
    rownr = 0
    for input_row in input_reader:     #Loop over all lines (spectral bins) in the csv file
        freq = input_row[1]             #Second column (first column is index [0]) is the frequency (in MHz)
        freqlist.append(float(freq))
        #print("Freq=%f MHz"%(float(freq)))
        velocity = input_row[2]          #Third column is the corresponding velocity in km/s
        velocitylist.append(float(velocity))
        #print("Vel=%f km/s"%(float(velocity)))
        temperature = input_row[3]       #Fourth column is the measured (probably uncalibrated) temperature in K
        temperaturelist.append(float(temperature))
        #print("T=%f K"%(float(temperature)))
        rownr += 1
    input_file.close()
    print("  Done with reading in %d spectral points" %(rownr))
    # Return the data as a list of three numpy arrays, corresponding to the frequency (in MHz), velocity (in km/s) and the temperature (in K)
    return [ np.array(freqlist) , np.array(velocitylist) , np.array(temperaturelist) ]


# Given a spectrum, remove velocities between vellow ... velhigh. A new spectrum is returned, so the input spectrum is preserved.
def cut_velocity_range(spectrum, vellow, velhigh):
    freqlist = []
    velocitylist = []
    temperaturelist = []

    #Loop over all defined spectral channels in the input spectrum
    nrbinscut = 0
    for binnr in range(0, len(spectrum[0])):
        if spectrum[1][binnr] < vellow or spectrum[1][binnr] > velhigh:
            #print ("%d %f" %(binnr, spectrum[1][binnr]))
            freqlist.append(spectrum[0][binnr])
            velocitylist.append(spectrum[1][binnr])
            temperaturelist.append(spectrum[2][binnr])
        else:
            nrbinscut += 1
    print ("Cutting %d of the %d spectral channels" %(nrbinscut, len(spectrum[0])))
    return [ np.array(freqlist) , np.array(velocitylist) , np.array(temperaturelist) ]



#Given a spectrum, integrate the velocities between vellow ... velhigh. The temperature integral is returned.
#The uncut spectrum is expected as input, as otherwise the normalisation of the integral might go wrong (the velocity resolution might be misinterpreted otherwise).
def velocity_integral(spectrum, vellow, velhigh, rms):
    #Loop over all defined spectral channels in the input spectrum
    nrbins = 0
    tempint = 0.0
    for binnr in range(0, len(spectrum[0])):
        if spectrum[1][binnr] >= vellow and spectrum[1][binnr] <= velhigh \
            and np.abs( spectrum[2][binnr] ) > 5*rms:

            tempint += spectrum[2][binnr]
            nrbins += 1
    if nrbins == 0:
        print ("No spectral channels found in the range %f ... %f km/s" %(vellow, velhigh))
        return 0.0, 0.0
    dv = spectrum[1][1] - spectrum[1][0]  # Velocity resolution
    tempint *= dv
    error = rms*np.sqrt(nrbins)
    error *= dv
    print ("Temperature integral is (%f +- %f) K km/s" %(tempint, error))
    return tempint, error;

