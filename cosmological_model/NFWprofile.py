
import numpy as np
import scipy.integrate 
from astropy.cosmology import Planck13

""" Calculate properties of NFW mass profile """

""" Use Duffy+ 2008 concertration-mass relation"""
def c200(z,M_200):

	#M_200_sun = M_200/2e30 # M_sun

	c_200 = 9.23*(M_200/2e12)**(-0.090)*(1+z)**(-0.69) # no unit

	return c_200

def r200(z,M_200): ## unit: arcsec

	critical_density = Planck13.critical_density(z).si.value # unit: kg/m^3
	m2arcs = np.degrees(1.0/Planck13.angular_diameter_distance(z).si.value)*3600.
	critical_density  = critical_density*Planck13.H0/100./m2arcs**3/2e30 # unit: Msun/arcsec^3/h

	r200_cube = 200*critical_density/M_200/(4./3.*np.pi)
	r200 = r200_cube**(1./3.)

	return r200.value

def rs(z,M_200):
	c200_list = c200(z,M_200)
	r200_list = r200(z,M_200)

	return r200_list/c200_list

def rho_s(M_200,r200,c200):
	rho_list =M_200/(4.0*np.pi*r200**3)/(np.log(1+c200)-c200/(1+c200))

	return rho_list

""" NFW P(r)*4 pi r^2, PDF """

def pdf(r,rs):
	x = r/rs
	return (4.0*np.pi*r**2)*1.0/x/(1+x)**2

""" Discrete CDF, ready for interpolate """
def cdf_d(ri,rs):

	#ri = np.linspace(0,r_end,r_end*10000)

	pdf_d = pdf(ri,rs) # discrete pdf
	cdf_d = np.zeros(len(ri))

	i = 1
	while i < len(ri):
		cdf_d[i] = cdf_d[i-1]+pdf_d[i]*(ri[1]-ri[0])
		i = i+1

	# normalization
	cdf_d = cdf_d/max(cdf_d)

	return cdf_d

def cdf(ri,rs):

	profile = lambda r: (4.0*np.pi*r**2)*1.0/(r/rs)/(1.+r/rs)**2

	integrate = np.zeros(len(ri))

	for i in range(len(ri)):

		I = scipy.integrate.quad(profile,0,ri[i])
		integrate[i] = I[0]

	return integrate

""" Draw from inverse_cdf will get cloned distribution """
def inverse_cdf(r,rs,r_end):

	ri = np.linspace(0,r_end,100)
	#Ix = cdf_d(ri,rs)
	Ix = cdf(ri,rs)
	Ix = Ix/Ix[-1]
	Iy = ri

	return np.interp(r,Ix,Iy)

