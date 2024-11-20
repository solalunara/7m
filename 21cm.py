import numpy as np;
from matplotlib import pyplot as plt;
from os import walk;
import lib7m;
import scipy.optimize as sp;
import scipy.odr as odr;

JODRELL_DIR = '/home/fvp/Downloads/jodrell/';
S7_STRENGTH = 4600.0;
DO_M33 = True;

M33_DIST = 0.869; #Mpc
M33_DIST_ERR = 0.018; #Mpc
M33_DIAMETER = 18.75; #kpc
BOLTZMANN = 1.38064852e-23; #m^2 kg s^-2 K^-1
TELESCOPE_AREA = np.pi * ( 6.4 / 2 )**2;
def TEMP_TO_JY( temp ):
    return temp * 1e26 / TELESCOPE_AREA * BOLTZMANN;

def chi_squared( function, data, *params ):
    """
    Calculates the chi-squared value for an arbitrary function

    Args:
        function (lambda): predictive function f(x, ...) where ... are params
        data (array): 2d array, rows are x/y/sigma and cols are each data point
        params: various function parameters
    """
    predicted = function( data[ 0 ], *params );
    observed = data[ 1 ];
    error = data[ 2 ];
    
    return np.sum((np.abs( predicted - observed ) / error)**2)

def cut_and_fit( spectrum, *cuts ):
    """cuts a spectrum, fits a polynomial to the cut part, and subtracts the fit from the spectrum

    Args:
        spectrum (list[NDArray]): the spectrum as given by lib7m.load_spectrum
        cuts (list[(float, float)]): the velocity ranges to cut

    Returns:
        list[NDArray]: a flattened spectrum
        float: the root mean square of the spectrum minus the cuts
        lambda: the polynomial fit to the cuts
        list[NDArray]: the spectrum after the final cut
    """
    degree = 6;
    try:
        degree = int( cuts[ -1 ] );
        cuts = cuts[ :-1 ];
    except:
        pass;
    
    spectrum_adjusted = np.copy( spectrum );
    spectrum_cut = np.copy( spectrum );
    for cut in cuts:
        spectrum_cut = lib7m.cut_velocity_range( spectrum_cut, cut[ 0 ], cut[ 1 ] );
    coeff = np.polyfit( spectrum_cut[ 1 ], spectrum_cut[ 2 ], degree );
    fitfunction = np.poly1d( coeff );
    spectrum_adjusted[ 2 ] -= fitfunction( spectrum[ 1 ] );
    spectrum_cut_adjusted = np.copy( spectrum_cut );
    spectrum_cut_adjusted[ 2 ] -= fitfunction( spectrum_cut[ 1 ] );
    rms = np.sqrt( np.mean( ( spectrum_cut_adjusted[ 2 ] )**2 ) )
    return spectrum_adjusted, rms, fitfunction, spectrum_cut;

def integral_error_s7_adjustment( rawvalue, rawerror, s7error, s7rawint ):
    """calculates the error of the integral of the spectrum

    Args:
        rawvalue (float): the integral value of the spectrum
        rawerror (float): the error of the integral value of the spectrum
        s7error (float): the error of the s7 integral
        s7rawvalue (float): the integral value of the s7 spectrum before adjustment

    Returns:
        float: the error of the integral value of the spectrum
    """
    if ( rawvalue == 0.0 ): return 0.0, 0.0;
    
    adjusted_value = rawvalue * S7_STRENGTH/s7rawint;
    adjusted_error = np.abs( adjusted_value ) * np.sqrt( ( rawerror/rawvalue )**2 + ( s7error/s7rawint )**2 );
    return adjusted_value, adjusted_error;

H1_30 = [];
ROT_CURVE = [];
ROT_CURVE_PROC = [];

paths_thickness = [
    JODRELL_DIR+'2024-10-01/',
    JODRELL_DIR+'2024-10-02/',
    JODRELL_DIR+'2024-11-05/',
];

paths_rotcurve = [
    JODRELL_DIR+'2024-10-15/',
    JODRELL_DIR+'2024-10-16/',
    JODRELL_DIR+'2024-10-16-pt2/',
];

m33_path = JODRELL_DIR+"2024-11-10/"

scans_path = [
    JODRELL_DIR+'2024-10-15/'
]

for path7m in paths_thickness:
    filelist = []
    for (dirpath, dirnames, filenames) in walk(path7m):
        filelist.extend(filenames)
        break
    
    date = path7m.split( '/' )[ -2 ];
    s7path = JODRELL_DIR+'/s7/'+date+'.csv';
    
    s7spectrum = lib7m.load_spectrum(s7path)
    s7_range = (-120.0, 30.0);
    s7_reflaction_range = (-350.0, -187.0);
    
    if ( date != '2024-10-01' ):
        s7_adjusted, s7rms, _, _ = cut_and_fit( s7spectrum, s7_range, s7_reflaction_range );
    else: s7_adjusted, s7rms, _, _ = cut_and_fit( s7spectrum, s7_range ); #no reflection for first day
    
    # scale the s7 spectrum to 4600 K
    s7rawint, s7error = lib7m.velocity_integral( s7_adjusted, s7_range[ 0 ], s7_range[ 1 ], s7rms );
    s7_adjusted[ 2 ] *= S7_STRENGTH/s7rawint;
    
    #plt.figure();
    #plt.plot( s7_adjusted[ 1 ], s7_adjusted[ 2 ] );
    #plt.xlabel( "Velocity [km/s]" );
    #plt.ylabel( "Temperature [K]" );
    #plt.title( "s7 " + date );
    #plt.show()
    
    
    for filename in filelist:
        if filename.endswith(".csv"):
            #files have format: l<longitude>b<latitude>.csv
            longitude = float( filename.split( 'l' )[ 1 ].split( 'b' )[ 0 ] );
            latitude = float( filename.split( 'b' )[ 1 ].split( '.csv' )[ 0 ] );
            
            spectrum_raw = lib7m.load_spectrum(path7m+filename)
            H1_reflection_range = (-245.0, -180.0);
            H1_range = (-20, 40);
            lowbound_cut = ( -float( 'inf' ), -100 );
            highbound_cut = ( 100, float( 'inf' ) );
            spectrum, rms, fitfunction, spectrum_cut = cut_and_fit( spectrum_raw, H1_range, H1_reflection_range, lowbound_cut, highbound_cut );
            
            rawint, rawerror = lib7m.velocity_integral( spectrum, H1_range[ 0 ], H1_range[ 1 ], rms );
            
            integral, error = integral_error_s7_adjustment( rawint, rawerror, s7error, s7rawint );
            if error == 0:
                error = 10;
            print( "integral = %f +/- %f K km/s" %(integral, error) )
            if ( longitude == 30.0 and latitude > 0.1 ):
                H1_30.append( ( latitude, TEMP_TO_JY( integral ), TEMP_TO_JY( error ) ) );

            #plt.figure()
            #plt.plot(spectrum_raw[1], spectrum_raw[2], spectrum_raw[1], fitfunction(spectrum_raw[1]));
            #plt.plot(spectrum_cut[1], spectrum_cut[2], '.')
            #plt.xlabel("Velocity [km/s]")
            #plt.ylabel("Temperature [K]")
            #plt.ylim(400, 800)
            #plt.title(filename)
            #plt.show()
H1_30 = np.array( sorted( H1_30, key = lambda x: x[ 0 ] ) );
def d( B, x ):
    return B[ 0 ] / ( np.sin( np.deg2rad( x ) ) );
#perform a nonlinear regression on the data using ODR
sx = [ 0.1 for _ in H1_30[ :, 0 ] ];
model = odr.Model( d );
data = odr.RealData( H1_30[ :, 0 ], H1_30[ :, 1 ], sx=sx, sy=H1_30[ :, 2 ] );
fit = odr.ODR( data, model, beta0=[ 1.0 ] );
fit.set_job( fit_type=0 );
output = fit.run();
output.pprint();

params = output.beta;
errors = output.sd_beta;

x = np.linspace( 5, 90, 100 );
y = d( params, x );

plt.figure();
plt.plot( x, y );
plt.errorbar( H1_30[ :, 0 ], H1_30[ :, 1 ], yerr = H1_30[ :, 2 ], fmt = 'o' );
plt.xlabel( "Latitude (Degrees)" );
plt.ylabel( "Flux Integral [Jy km/s]" );
plt.title( "H1 30" );
plt.show();


for path7m in paths_rotcurve:
    filelist = []
    for (dirpath, dirnames, filenames) in walk(path7m):
        filelist.extend(filenames)
        break
    
    date = path7m.split( '/' )[ -2 ];
    s7path = JODRELL_DIR+'/s7/'+date+'.csv';
    
    s7spectrum = lib7m.load_spectrum(s7path)
    s7_range = (-120.0, 30.0);
    s7_reflaction_range = (-350.0, -187.0);
    
    if ( date != '2024-10-01' ):
        s7_adjusted, s7rms, _, _ = cut_and_fit( s7spectrum, s7_range, s7_reflaction_range );
    else: s7_adjusted, s7rms, _, _ = cut_and_fit( s7spectrum, s7_range ); #no reflection for first day
    
    # scale the s7 spectrum to 4600 K
    s7rawint, s7error = lib7m.velocity_integral( s7_adjusted, s7_range[ 0 ], s7_range[ 1 ], s7rms );
    s7_adjusted[ 2 ] *= S7_STRENGTH/s7rawint;
    
    for filename in filelist:
        if filename.endswith(".csv"):
            #files have format: l<longitude>b<latitude>.csv
            longitude = float( filename.split( 'l' )[ 1 ].split( 'b' )[ 0 ] );
            latitude = float( filename.split( 'b' )[ 1 ].split( '.csv' )[ 0 ] );
            
            spectrum_raw = lib7m.load_spectrum(path7m+filename)
            H1_reflection_range = (-500.0, -250.0);
            H1_range = (-180, 150);
            lowbound_cut = ( -float( 'inf' ), -500 );
            highbound_cut = ( 400, float( 'inf' ) );
            spectrum, rms, fitfunction, spectrum_cut = cut_and_fit( spectrum_raw, H1_range, H1_reflection_range, lowbound_cut, highbound_cut );
            
            rawint, rawerror = lib7m.velocity_integral( spectrum, H1_range[ 0 ], H1_range[ 1 ], rms );
            integral, error = integral_error_s7_adjustment( rawint, rawerror, s7error, s7rawint );
            
            start_low = 0;
            start = 0;
            start_high = 0;
            for i in range( len( spectrum[ 2 ] ) ):
                if ( spectrum[ 1 ][ i ] > 250 or spectrum[ 1 ][ i ] < -250 ):
                    continue;
                
                if ( spectrum[ 2 ][ i ] > 1 * rms ):
                    start_low = spectrum[ 1 ][ i ];
                elif start_low != 0 and start_high != 0 and start != 0:
                    break;
                    
                if ( spectrum[ 2 ][ i ] > 10 * rms ):
                    start_high = spectrum[ 1 ][ i ];
                    
                if ( spectrum[ 2 ][ i ] > 5 * rms ):
                    start = spectrum[ 1 ][ i ];
                
            error = np.abs( start_high - start_low ) / 2;
            print( "start = %d" %start );
            print( "error = %d" %error );
            
            
            ROT_CURVE.append( ( longitude, start, error ) );
            
            
            R_0 = 8.5; #Kpc
            shift_factor = 220 * np.sin( np.radians( longitude ) ); #km/s
            velocity = ( start + shift_factor );
            assumption_factor = velocity / 220 - 1 - np.sin( np.radians( longitude ) );
            if ( longitude <= 90 ):
                radius = R_0 * np.sin( np.radians( longitude ) );
                if ( assumption_factor >= 0 ):
                    raise ValueError( "Assumption factor is negative - Velocity Curve does not obey assumptions" );
                ROT_CURVE_PROC.append( ( radius, velocity, error, assumption_factor ) );
            

            #plt.figure()
            #plt.plot(spectrum_raw[1], spectrum_raw[2], spectrum_raw[1], fitfunction(spectrum_raw[1]));
            #plt.plot(spectrum_cut[1], spectrum_cut[2], '.')
            #plt.axvline( x = start_low, color = 'r' );
            #plt.axvline( x = start, color = 'g' );
            #plt.axvline( x = start_high, color = 'r' );
            #plt.xlabel("Velocity [km/s]")
            #plt.ylabel("Temperature [K]")
            #plt.ylim(400, 800)
            #plt.title(filename)
            #plt.show()
            
ROT_CURVE = np.array( sorted( ROT_CURVE, key = lambda x: x[ 0 ] ) );
plt.figure();
plt.errorbar( ROT_CURVE[ :, 0 ], ROT_CURVE[ :, 1 ], yerr = ROT_CURVE[ :, 2 ], fmt = 'o' );
plt.xlabel( "Longitude (Degrees)" );
plt.ylabel( "Max Velocity [km/s]" );
plt.title( "Rotation Observations" );
plt.show();

ROT_CURVE_PROC = np.array( sorted( ROT_CURVE_PROC, key = lambda x: x[ 0 ] ) );
plt.figure();
plt.errorbar( ROT_CURVE_PROC[ :, 0 ], ROT_CURVE_PROC[ :, 1 ], yerr=ROT_CURVE_PROC[ :, 2 ], fmt='o' );
plt.xlabel( "Radius (km)" );
plt.ylabel( "Velocity (km/s)" );
plt.title( "Rotation Curve" );
plt.show();

plt.figure();
plt.plot( ROT_CURVE_PROC[ :, 0 ], ROT_CURVE_PROC[ :, 3 ], 'o' );
plt.xlabel( "Radius (km)" );
plt.ylabel( "Assumption Factor" );
plt.title( "Verification Graph" );
plt.show();


###### M33 PART ######
if ( DO_M33 ):
    filelist = []
    for (dirpath, dirnames, filenames) in walk(m33_path):
        filelist.extend(filenames)
        break

    date = m33_path.split( '/' )[ -2 ];
    s7path = JODRELL_DIR+'/s7/'+date+'.csv';

    s7spectrum = lib7m.load_spectrum(s7path)
    s7_range = (-120.0, 30.0);
    s7_reflaction_range = (-350.0, -187.0);

    if ( date != '2024-10-01' ):
        s7_adjusted, s7rms, _, _ = cut_and_fit( s7spectrum, s7_range, s7_reflaction_range );
    else: s7_adjusted, s7rms, _, _ = cut_and_fit( s7spectrum, s7_range ); #no reflection for first day

    # scale the s7 spectrum to 4600 K
    s7rawint, s7error = lib7m.velocity_integral( s7_adjusted, s7_range[ 0 ], s7_range[ 1 ], s7rms );
    s7_adjusted[ 2 ] *= S7_STRENGTH/s7rawint;

    num_entries = 0;
    m33_spectrum = None;
    spectrums = [];
    for filename in filelist:
        if filename.endswith(".csv"):
            spectrum_raw = lib7m.load_spectrum(m33_path+filename);
            spectrums.append( spectrum_raw );
            if ( m33_spectrum == None ):
                m33_spectrum = spectrum_raw;
            else:
                m33_spectrum[ 2 ] += spectrum_raw[ 2 ]; #add spectra but not velocities/frequencies
                
            num_entries += 1;
    m33_spectrum[ 2 ] /= num_entries;
    
    #calculate standard deviation between spectrums without modifying m33
    spectrum_stdev = np.sqrt( np.mean( ( np.array( [ spectrum[ 2 ] for spectrum in spectrums ] ) - m33_spectrum[ 2 ] )**2 ) );
    

    data_cut = ( -350, 50 );
    m33_range = ( -300, -65 );
    spectrum, rms, fitfunction, spectrum_cut = cut_and_fit( m33_spectrum, data_cut );
    
    rawint, rawerror = lib7m.velocity_integral( spectrum, m33_range[ 0 ], m33_range[ 1 ], rms );
    integral, error = integral_error_s7_adjustment( rawint, rawerror, s7error, s7rawint );
    
    
    integral = TEMP_TO_JY( integral );
    error = TEMP_TO_JY( error );
    scale_factor = integral / rawint;
    scale_factor_error = scale_factor * np.sqrt( ( rawerror/rawint )**2 + ( error/integral )**2 );
    
    spectrum[ 2 ] *= scale_factor;
    spectrum_errors = np.abs( spectrum[ 2 ] ) * np.sqrt( ( scale_factor_error / scale_factor )**2 + ( spectrum_stdev / m33_spectrum[ 2 ] )**2 );
    
    
    m = 5.72;
    M = m - 5 * np.log10( M33_DIST * 1e5 );
    luminosity = 10**( 0.4 * ( 4.83 - M ) );
    luminosity_error = luminosity * M33_DIST_ERR / M33_DIST;
    print( "Luminosity = %e +/- %e Solar Luminosities" %(luminosity, luminosity_error) );
    mass = luminosity;
    mass_error = luminosity_error;
    print( "Mass = %e +/- %e Solar Masses" %(mass, mass_error) );
    
    hydrogen_mass = 2.36e5 * integral * M33_DIST**2;
    hydrogen_mass_error = hydrogen_mass * np.sqrt( ( error / integral )**2 + ( M33_DIST_ERR / M33_DIST )**2 );
    print( "Hydrogen Mass = %e +/- %e Solar Masses" %(hydrogen_mass, hydrogen_mass_error) );
    
    helium_mass = 0.25 * hydrogen_mass;
    helium_mass_error = 0.25 * hydrogen_mass_error;
    
    plt.figure();
    plt.errorbar( m33_spectrum[ 1 ], spectrum[ 2 ], spectrum_errors );
    plt.xlabel( "Velocity [km/s]" );
    plt.ylabel( "Janskys [Jy]" );
    plt.title( "M33" );
    plt.show();
    
    central_velocity = m33_range[ 0 ] + ( m33_range[ 1 ] - m33_range[ 0 ] ) / 2;
    peak_velocity = m33_range[ 1 ] - central_velocity;
    
    #m33 diameter in AU
    diameter = M33_DIAMETER * 2.063e+8;
    peak_velocity = peak_velocity / 29.78;
    mass = diameter / 2 * peak_velocity;
    print( "Mass = %e Solar Masses" %mass );
    
    