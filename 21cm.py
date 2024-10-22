import numpy as np;
from matplotlib import pyplot as plt;
from os import walk;
import lib7m;

JODRELL_DIR = '/home/fvp/Downloads/jodrell/';
S7_STRENGTH = 4600.0;

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
    spectrum_adjusted = np.copy( spectrum );
    spectrum_cut = np.copy( spectrum );
    for cut in cuts:
        spectrum_cut = lib7m.cut_velocity_range( spectrum_cut, cut[ 0 ], cut[ 1 ] );
    coeff = np.polyfit( spectrum_cut[ 1 ], spectrum_cut[ 2 ], 4 );
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
];
paths_rotcurve = [
    JODRELL_DIR+'2024-10-15/',
    JODRELL_DIR+'2024-10-16/',
    JODRELL_DIR+'2024-10-16-pt2/',
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
            factor = 0.9 * np.cos( np.radians( latitude ) ) + 0.1;
            #H1__reclection_range = (-280.0, -86.0) * ( 0.5 * np.sin( latitude ) + 0.5 );
            #H1_range = (-60.0, 140.0)  * ( 0.5 * np.sin( latitude ) + 0.5 );
            H1_reflection_range = (-245.0, -180.0);
            H1_range = (-20, 40);
            lowbound_cut = ( -float( 'inf' ), -100 );
            highbound_cut = ( 100, float( 'inf' ) );
            spectrum, rms, fitfunction, spectrum_cut = cut_and_fit( spectrum_raw, H1_range, H1_reflection_range, lowbound_cut, highbound_cut );
            
            rawint, rawerror = lib7m.velocity_integral( spectrum, H1_range[ 0 ], H1_range[ 1 ], rms );
            
            integral, error = integral_error_s7_adjustment( rawint, rawerror, s7error, s7rawint );
            print( "integral = %f +/- %f K km/s" %(integral, error) )
            if ( longitude == 30.0 and latitude > 0.1 ):
                H1_30.append( ( latitude, integral, error ) );

            #plt.figure()
            #plt.plot(spectrum_raw[1], spectrum_raw[2], spectrum_raw[1], fitfunction(spectrum_raw[1]));
            #plt.plot(spectrum_cut[1], spectrum_cut[2], '.')
            #plt.xlabel("Velocity [km/s]")
            #plt.ylabel("Temperature [K]")
            #plt.ylim(400, 800)
            #plt.title(filename)
            #plt.show()
H1_30 = np.array( sorted( H1_30, key = lambda x: x[ 0 ] ) );
plt.figure();
plt.errorbar( H1_30[ :, 0 ], H1_30[ :, 1 ], yerr = H1_30[ :, 2 ], fmt = 'o' );
plt.xlabel( "Latitude (Degrees)" );
plt.ylabel( "Magnitude [K km/s]" );
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
            factor = 0.9 * np.cos( np.radians( latitude ) ) + 0.1;
            #H1__reclection_range = (-280.0, -86.0) * ( 0.5 * np.sin( latitude ) + 0.5 );
            #H1_range = (-60.0, 140.0)  * ( 0.5 * np.sin( latitude ) + 0.5 );
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
                if ( start_low == 0 and spectrum[ 1 ][ i ] > -250 and np.abs( spectrum[ 2 ][ i ] ) > 10 * rms ):
                    start_low = spectrum[ 1 ][ i ];
                    
                if ( start_high == 0 and spectrum[ 1 ][ i ] > -250 and np.abs( spectrum[ 2 ][ i ] ) > 20 * rms ):
                    start_high = spectrum[ 1 ][ i ];
                    
                if ( start == 0 and spectrum[ 1 ][ i ] > -250 and np.abs( spectrum[ 2 ][ i ] ) > 15 * rms ):
                    start = spectrum[ 1 ][ i ];
                    
                if ( start_low != 0 and start_high != 0 and start != 0 ):
                    break;
                
            error = np.abs( start_high - start_low );
            print( "start = %d" %start );
            print( "error = %d" %error );
            
            
            ROT_CURVE.append( ( longitude, start, error ) );
            
            shift_factor = 220 * np.sin( np.radians( latitude ) ); #km/s
            R_0 = 2.6e17; #km
            radius = R_0 * np.sin( np.radians( longitude ) );
            velocity = -( start - shift_factor );
            ROT_CURVE_PROC.append( ( radius, velocity ) );
            

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
plt.ylabel( "Magnitude [K km/s]" );
plt.title( "ROT_CURVE" );
plt.show();

ROT_CURVE_PROC = np.array( sorted( ROT_CURVE_PROC, key = lambda x: x[ 0 ] ) );
plt.figure();
plt.plot( ROT_CURVE_PROC[ :, 0 ], ROT_CURVE_PROC[ :, 1 ], 'o' );
plt.xlabel( "Radius (km)" );
plt.ylabel( "Velocity (km/s)" );
plt.title( "ROT_CURVE_PROC" );
plt.show();