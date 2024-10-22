import numpy as np

import astropy.coordinates as coord

import astropy.units as u

import astropy.time as atime

import matplotlib.pyplot as plt

 

target = coord.Galactic(l=180*u.deg,b=0*u.deg)

day="2023-09-26"

t0 = atime.Time(day)

telescope=coord.EarthLocation(lat=53.2366243*u.deg,lon=-2.3091195*u.deg)

dt = atime.TimeDelta(np.arange(360)*240,format='sec')

t = t0+dt

 

altaz = target.transform_to(coord.AltAz(obstime=t,location=telescope))

x=24*(t.mjd-t0.mjd)

alt=altaz.alt.deg

plt.plot(x,alt)

plt.axhline(0,color='k')

plt.axhline(10,color='red')

 

plt.ylim(-10,90)

plt.title(f"{target}")

plt.xlabel(f"Hour since {day} at 00:00")

plt.ylabel("Angle above horizon (deg)")

plt.show()