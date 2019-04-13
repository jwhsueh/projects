import numpy as np
import NFWprofile as NFW
from astropy import cosmology
import scipy.integrate

#### --------------- substructure ---------------- ####

""" Substructure mass function"""
def sub_massf(mass,m_c):
    dn = mass**(-1.9)*(1+m_c/mass)**(-1.3)

    return dn


""" Substructure properties """
def sub_pdf(ml,mh,f_sub,ktot,m_c,light_cone,Re_arc,sigC):

	f_low,f_hi = light_cone[3],light_cone[4]
	area = np.pi*(f_hi*Re_arc-f_low*Re_arc)**2
	
	## total mass for substructures
	Mtot = f_sub*sigC*ktot # M_Sun

	m_sub = np.linspace(ml,mh,100)
	m_sub = 10**m_sub

	prob_sub = sub_massf(m_sub,m_c)
	prob_sub = prob_sub/np.sum(prob_sub)

	sub_np = lambda x: x**(-0.9)*(1+m_c/x)**(-1.3)
	A0 = Mtot/scipy.integrate.quad(sub_np,10**ml,10**mh)[0]
	#A0 = 1.0/scipy.integrate.quad(sub_np,10**ml,10**mh)[0]
	#print A0

	sub_mf = lambda x: x**(-1.9)*(1+m_c/x)**(-1.3)
	Nsub = A0*(scipy.integrate.quad(sub_mf,10**ml,10**mh)[0])

	if Nsub == 0:
		Nsub = 1

	return m_sub,prob_sub,Mtot,Nsub

""" Create substructure list """
def create_subs(sub_out,light_cone,Re_arc):

	m_sub,prob_sub,Mtot,Nsub = sub_out[0],sub_out[1],sub_out[2],sub_out[3]
	z_l,z_s = light_cone[0],light_cone[1]
	f_low,f_hi = light_cone[3],light_cone[4]

	Ntot = np.random.poisson(Nsub)
	if Ntot == 0:
		Ntot = 1

	n_inv = 1
	m_list = np.empty(0)

	while True:
		inv_list = np.random.choice(m_sub,n_inv,p=prob_sub)
		m_list = np.append(m_list,inv_list)

		if len(m_list)==Ntot:
			break

	x_list,y_list = np.empty(0),np.empty(0)
	i=0
	while i<len(m_list):

		x_temp = np.random.rand(1)*2.0*f_hi*Re_arc-f_hi*Re_arc
		y_temp = np.random.rand(1)*2.0*f_hi*Re_arc-f_hi*Re_arc
		dis2_temp = np.sqrt(x_temp**2+y_temp**2)

		if np.logical_and(dis2_temp>(f_low*Re_arc),dis2_temp<f_hi*Re_arc):

			x_list = np.append(x_list,x_temp)
			y_list = np.append(y_list,y_temp)
			i=i+1

	## calculate ks, rs
	r200_list = NFW.r200(z_l,m_list)
	c200_list = NFW.c200(z_l,m_list)

	return  m_list,x_list,y_list,r200_list,c200_list

