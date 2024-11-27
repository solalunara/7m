import numpy as np;
from matplotlib import pyplot as plt;
from os import walk;
import lib7m;
import scipy.optimize as sp;
import scipy.odr as odr;

JODRELL_DIR = '/home/luna/prog/7m/data/jodrell/';
S7_STRENGTH = 4600.0;
DO_M33 = True;

M33_DIST = 0.869; #Mpc
M33_DIST_ERR = 0.018; #Mpc
M33_DIAMETER = 18.75; #kpc
BOLTZMANN = 1.38064852e-23; #m^2 kg s^-2 K^-1
TELESCOPE_DIAMETER = 6.4; #m^2
H1_FREQUENCY = 1420.40575177e6; #Hz
C = 299792.458; #km/s

def TEMP_TO_JY( temp ):
    return temp * 3520 / ( TELESCOPE_DIAMETER**2 * 0.55 );

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

def degrees_of_freedom( func, parameters, xdata, xerr, ydata, yerr, *beta0 ):
    """The definition d.o.f = N - P does not extend to nonlinear fits
        Rene Andrae, Tim Schulze-Hartung, and Peter Melchior. (2010). Dos and don'ts of reduced chi-squared. 
            (https://arxiv.org/pdf/1012.3754.pdf)
        using a generalized d.o.f. definition d.o.f. = N - P_eff
        calculate effective parameters P_eff = trace[ H ]
        where H = du_i / dy_i
        u = predicted, y = observed
            Ye, J. 1998, Journal of the American Statistical Association, 93, 120

        Args:
            xdata (NDArray): x values for each point
            ydata (Array): y values for each point
            sigma (Array): errors on the y values
"""
    h = .0001;
    hatmatrix_diag = [];
    odr_model = odr.Model( func );
    for y_i in range( 0, len( ydata ) ):
        ydata_plus = ydata.copy();
        ydata_plus[ y_i ] += h;
        odr_data_plus = odr.RealData( xdata, ydata_plus, xerr, yerr );
        odr_plus = odr.ODR( odr_data_plus, odr_model, beta0 );
        odr_plus.set_job( fit_type=0 );
        output_plus = odr_plus.run();
        
        yhat = func( parameters, xdata[ y_i ] );
        yhat_plus = func( output_plus.beta, xdata[ y_i ] );
        hatmatrix_diag.append( ( yhat_plus - yhat ) / h );
    hatmatrix_diag = np.array( hatmatrix_diag );
    effective_parameters = np.sum( hatmatrix_diag );
    return len( xdata ) - effective_parameters;


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

duplicate_arr_storage = np.empty( 0 );
lat_0_integrals = [];
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

            # we have a duplicate at latitude 80, just hard code it
            if ( np.abs( latitude - 80 ) <= 0.1 ):
                if ( not duplicate_arr_storage.any() ):
                    duplicate_arr_storage = np.copy( spectrum_raw );
                    continue;
                else:
                    spectrum_raw[ 2 ] = ( spectrum_raw[ 2 ] + duplicate_arr_storage[ 2 ] ) / 2

            H1_reflection_range = (-245.0, -180.0);
            H1_range = (-20, 40);
            if ( latitude == 0 or latitude == 5 ):
                H1_reflection_range = ( -290, -80 );
                H1_range = ( -60, 150 );
            lowbound_cut = ( -float( 'inf' ), -420 );
            highbound_cut = ( 640, float( 'inf' ) );
            spectrum, rms, fitfunction, spectrum_cut = cut_and_fit( spectrum_raw, H1_range, H1_reflection_range, lowbound_cut, highbound_cut );
            
            rawint, rawerror = lib7m.velocity_integral( spectrum, H1_range[ 0 ], H1_range[ 1 ], rms );
            
            integral, error = integral_error_s7_adjustment( rawint, rawerror, s7error, s7rawint );
            if error == 0:
                error = 10;
            print( "integral = %f +/- %f K km/s" %(integral, error) );
            if ( longitude == 30.0 and latitude > 0.1 and ( latitude != 80 or duplicate_arr_storage.any() ) ):
                H1_30.append( ( latitude, integral, error ) );
            elif ( latitude == 0 ):
                lat_0_integrals.append( (longitude, integral, error) );

            #plt.figure()
            #plt.plot(spectrum_raw[1], spectrum_raw[2], spectrum_raw[1], fitfunction(spectrum_raw[1]));
            #plt.plot(spectrum_cut[1], spectrum_cut[2], '.')
            #plt.xlabel("Velocity [km/s]")
            #plt.ylabel("Temperature [K]")
            #plt.ylim(400, 800)
            #plt.title(filename)
            #plt.show()

lat_0_integrals = np.array( lat_0_integrals );
longitudes = lat_0_integrals[ :, 0 ];
index = np.where( longitudes == 30 )[ 0 ][ 0 ];
l30b0 = lat_0_integrals[ index ];

H1_30 = np.array( sorted( H1_30, key = lambda x: x[ 0 ] ) );
#H1_30[ :, 1 ] *= scale_factor_integral_to_distance;
H1_30[ :, 1 ] += 0.0001; #add a small value to prevent 0 errors
#H1_30[ :, 2 ] = H1_30[ :, 1 ] * np.sqrt( ( scale_factor_integral_to_distance_error / scale_factor_integral_to_distance )**2 + ( H1_30[ :, 2 ] * scale_factor_integral_to_distance / H1_30[ :, 1 ] )**2 );

H1_30[ :, 2 ] += 100;

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

dof = degrees_of_freedom( d, output.beta, H1_30[ :, 0 ], sx, H1_30[ :, 1 ], H1_30[ :, 2 ], 1.0 );
chi_squared_red = output.sum_square / dof;
print( "Reduced Chi-Squared = %f" %chi_squared_red );

params = output.beta;
errors = output.sd_beta;

y_val = 2*params[ 0 ];
y_val_err = 2*errors[ 0 ];
t_val = l30b0[ 1 ];
t_val_err = l30b0[ 2 ];

ratio = t_val / y_val;
ratio_err = ratio * np.sqrt( ( y_val_err / y_val )**2 + ( t_val_err / t_val )**2 );


print( "Thickness ratio = %f +/- %f" %(ratio , ratio_err) );


x = np.linspace( 5, 90, 100 );
y = d( params, x );

plt.figure();
plt.plot( x, y );
plt.errorbar( H1_30[ :, 0 ], H1_30[ :, 1 ], yerr = H1_30[ :, 2 ], fmt = 'o' );
plt.xlabel( "Latitude (Degrees)" );
plt.ylabel( "Integrals (K km/s)" );
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
            if ( longitude <= 90 ):
                radius = R_0 * np.sin( np.radians( longitude ) );
                ROT_CURVE_PROC.append( ( radius, velocity, error ) );
            

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
plt.xlabel( "Radius (Kpc)" );
plt.ylabel( "Velocity (km/s)" );
plt.title( "Rotation Curve" );
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
    m33_bandwith = m33_range[ 1 ] - m33_range[ 0 ];
    m33_rotvel = m33_bandwith / 2 / np.cos( np.radians( 54 ) );

    m33_rotvel /= 29.78; #convert to solar units
    m33_rotvel_err = m33_rotvel * 0.05;
    m33_radius = M33_DIAMETER / 2 * 2.063e+8; #convert to AU

    m33_dyn_mass = m33_radius * m33_rotvel**2;
    m33_dyn_mass_err = m33_dyn_mass * ( m33_rotvel_err / m33_rotvel ) * np.sqrt( 2 );
    print( "M33 Dynamic Mass = %e +/- %e Solar Masses" %(m33_dyn_mass, m33_dyn_mass_err) );


    spectrum, rms, fitfunction, spectrum_cut = cut_and_fit( m33_spectrum, data_cut );
    
    rawint, rawerror = lib7m.velocity_integral( spectrum, m33_range[ 0 ], m33_range[ 1 ], rms );
    integral, error = integral_error_s7_adjustment( rawint, rawerror, s7error, s7rawint );

    #add an inherent uncertainty due to an overlap with a negative dip in our data
    error += integral * 0.01;
    
    
    scale_factor = integral / rawint;
    scale_factor_error = scale_factor * np.sqrt( ( rawerror/rawint )**2 + ( error/integral )**2 );

    integral = TEMP_TO_JY( integral );
    error = TEMP_TO_JY( error );
    # no need to convert km/s to frequency, factor already included in conversions
    
    spectrum[ 2 ] *= scale_factor;
    spectrum_errors = np.abs( spectrum[ 2 ] ) * np.sqrt( ( scale_factor_error / scale_factor )**2 + ( spectrum_stdev / m33_spectrum[ 2 ] )**2 );
    
    
    m = 5.72;
    M = m - 5 * np.log10( M33_DIST * 1e5 );
    luminosity = 10**( 0.4 * ( 4.83 - M ) );
    luminosity_error = luminosity * M33_DIST_ERR / M33_DIST;
    mass = luminosity;
    mass_error = luminosity_error;
    print( "Stellar Mass = %e +/- %e Solar Masses" %(mass, mass_error) );
    
    hydrogen_mass = 2.36e5 * integral * M33_DIST**2;
    hydrogen_mass_error = hydrogen_mass * np.sqrt( ( error / integral )**2 + ( M33_DIST_ERR / M33_DIST )**2 );
    print( "Hydrogen Mass = %e +/- %e Solar Masses" %(hydrogen_mass, hydrogen_mass_error) );
    
    helium_mass = 0.25 * hydrogen_mass;
    helium_mass_error = 0.25 * hydrogen_mass_error;
    print( "Helium Mass = %e +/- %e Solar Masses" %(helium_mass, helium_mass_error) );

    mass_tot = mass + hydrogen_mass + helium_mass;
    mass_tot_error = np.sqrt( mass_error**2 + hydrogen_mass_error**2 + helium_mass_error**2 );
    print( "Total Mass = %e +/- %e Solar Masses" %(mass_tot, mass_tot_error) );
    
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
    
