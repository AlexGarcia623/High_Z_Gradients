import sys
import os
import time
import h5py
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import illustris_python as il

from sfms import sfmscut, center, calc_rsfr_io, calc_incl, trans, calczgrad, calcrsfr, grad_valid
from matplotlib.colors import LogNorm
from tqdm import tqdm

from os import path, mkdir

xh     = 7.600E-01
zo     = 3.500E-01
mh     = 1.6726219E-24
kb     = 1.3806485E-16
mc     = 1.270E-02

snap2zTNG = {
    99:0,
    50:1,
    33:2,
    25:3,
    21:4,
    17:5,
    13:6,
    11:7,
    8:8
}

mpl.rcParams['text.usetex']        = True
mpl.rcParams['text.latex.unicode'] = True
mpl.rcParams['font.family']        = 'serif'
mpl.rcParams['font.size']          = 20

fs_og = 24
mpl.rcParams['font.size'] = fs_og
mpl.rcParams['axes.linewidth']  = 2
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.minor.visible'] = 'true'
mpl.rcParams['ytick.minor.visible'] = 'true'
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['xtick.minor.width'] = 1.0
mpl.rcParams['ytick.minor.width'] = 1.0
mpl.rcParams['xtick.major.size']  = 7.5
mpl.rcParams['ytick.major.size']  = 7.5
mpl.rcParams['xtick.minor.size']  = 3.5
mpl.rcParams['ytick.minor.size']  = 3.5
mpl.rcParams['xtick.top']   = True
mpl.rcParams['ytick.right'] = True

def save_data(snap, out_dir, m_star_min=9.0, m_star_max=np.inf, m_gas_min=9.0, res=2160,
              where_to_save=None):
    
    hdr  = il.groupcat.loadHeader(out_dir, snap)
    box_size = hdr['BoxSize']
    scf      = hdr['Time']
    h        = hdr['HubbleParam']
    print('Scale-factor: %s' %round(scf,1))
    z0       = (1.00E+00 / scf - 1.00E+00)
    print('Redshift: %s' %round(z0,1))
    fields = ['SubhaloGasMetallicity', 'SubhaloPos', 'SubhaloMass', 'SubhaloVel', 'SubhaloSFR',
              'SubhaloMassType','SubhaloGasMetallicitySfr','SubhaloHalfmassRadType']
    sub_cat = il.groupcat.loadSubhalos(out_dir, snap, fields = fields)
    subs    = il.groupcat.loadHalos(out_dir,snap,fields=['GroupFirstSub'])
    sub_cat['SubhaloMass']     *= 1.000E+10 / h
    sub_cat['SubhaloMassType'] *= 1.000E+10 / h
    
    sfms_idx = sfmscut(sub_cat['SubhaloMassType'][subs,4], sub_cat['SubhaloSFR'][subs])
    subs     = subs[(sub_cat['SubhaloMassType'][subs,4] > 1.000E+01**m_star_min) & 
                    (sub_cat['SubhaloMassType'][subs,4] < 1.000E+01**m_star_max) &
                    (sub_cat['SubhaloMassType'][subs,0] > 1.000E+01**m_gas_min) &
                    (sfms_idx) ]    
    
    print('Number of SF galaxies: %s' %len(subs))
    
    stellarHalfmassRad = sub_cat['SubhaloHalfmassRadType'][subs,4]
    
    stellarHalfmassRad *= (scf / h)
    
    for sub in tqdm(subs):
        get_profile(out_dir, snap, sub, sub_cat, box_size,
                    scf, h, res, sf_region=True, plot=False,
                    where_to_save=where_to_save)

def get_profile(out_dir, snap, sub, sub_cat, box_size, scf, h, res,
                sf_region=True, plot=False, where_to_save=None):
    
    rmax = 1.00E+02
    
    sub_pos = sub_cat['SubhaloPos'][sub]
    sub_met = sub_cat['SubhaloGasMetallicity'][sub]
    sub_vel = sub_cat['SubhaloVel'][sub]
    
    sub_SFR = sub_cat['SubhaloSFR'][sub]
    sub_SHM = sub_cat['SubhaloHalfmassRadType'][sub,4] * (scf / h)
    
    sub_stellar_mass = np.log10(sub_cat['SubhaloMassType'][sub,4])
    
    sub_redshift = snap2zTNG[snap]

    
    gas_pos   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Coordinates'      ])
    gas_vel   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Velocities'       ])
    gas_mass  = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Masses'           ])
    gas_sfr   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['StarFormationRate'])
    gas_rho   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Density'          ])
    gas_met   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metallicity'  ])
    ZO        = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metals'       ])[:,4]
    XH        = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metals'       ])[:,0]
    
    GFM_Metal = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metals'       ])
    
    gas_pos    = center(gas_pos, sub_pos, box_size)
    gas_pos   *= (scf / h)
    gas_vel   *= np.sqrt(scf)
    gas_vel   -= sub_vel
    gas_mass  *= (1.000E+10 / h)
    gas_rho   *= (1.000E+10 / h) / (scf / h )**3.00E+00
    gas_rho   *= (1.989E+33    ) / (3.086E+21**3.00E+00)
    gas_rho   *= XH / mh
    
    OH = ZO/XH * 1.00/16.00
    
    Zgas = np.log10(OH) + 12

    ri, ro = calc_rsfr_io(gas_pos, gas_sfr)
    ro2    = 2.000E+00 * ro
    
    riprime = ri + 0.25 * (ro - ri)
    
    sub_rsfr50 = calcrsfr(gas_pos, gas_sfr)

    # print('Rows without nans: %s' %len(gas_rho))
    sf_idx = gas_rho > 1.300E-01
    # print('SF gas particles in this halo: %s' %sum(sf_idx))
    incl   = calc_incl(gas_pos[sf_idx], gas_vel[sf_idx], gas_mass[sf_idx], ri, ro)

    gas_pos  = trans(gas_pos, incl)
    gas_vel  = trans(gas_vel, incl)

    r, rerr, oh, oherr, _rad_, _oh_ = calczgrad(gas_pos, gas_mass, gas_rho, GFM_Metal, rmax, res)
    
    if grad_valid(r, oh):
        
        rsmall, rbig = riprime, ro
        fit_mask     = ( r > rsmall ) & ( r < rbig ) & ~np.isnan(oh)
        try:
            gradient_SF, intercept_SF = np.polyfit( r[fit_mask], oh[fit_mask], 1 )
        except:
            gradient_SF = np.nan

        rsmall, rbig = 0.5 * sub_SHM, 2.0 * sub_SHM
        fit_mask     = ( r > rsmall ) & ( r < rbig ) & ~np.isnan(oh)
        try:
            gradient_OE, intercept_OE = np.polyfit( r[fit_mask], oh[fit_mask], 1 )
        except:
            gradient_OE = np.nan
        
        if plot:
            plt.clf()
            
            fig = plt.figure(figsize=(10,6))

            plt.hist2d( _rad_, _oh_, cmap=plt.cm.Greys, bins=(100,100) )
            plt.plot( r, oh, color='red' )
            
            plt.axvline( rsmall, color='k', linestyle='--' )
            plt.axvline( rbig  , color='k', linestyle='--' )
            
            
            _x_ = np.arange( 0, np.max(r), 0.1 )
            _y_ = gradient_SF * _x_ + intercept_SF
            plt.plot( _x_, _y_, color='blue' )
            
            plt.xlabel( r'${\rm Radius}~{\rm (kpc)}$' )
            plt.ylabel( r'$\log {\rm O/H} + 12~{\rm (dex)}$' )

            plt.text( 0.7, 0.9 , r'$z=%s$' %snap2zTNG[snap], transform=plt.gca().transAxes )
            plt.text( 0.7, 0.8 , r'$\log M_* = %s$' %round( sub_stellar_mass,2 ),transform=plt.gca().transAxes )
            
            plt.tight_layout()
            plt.savefig( './diagnostic_figs/' + str(sub) + '_TNG_profile.pdf', bbox_inches='tight' )
    else:
        gradient_OE = np.nan
        gradient_SF = np.nan
        
    if where_to_save:
        
        this_subhalo = where_to_save.create_group('Subhalo_%s' %sub)
        
        this_subhalo.create_dataset( 'StarFormingRegion' , data = gradient_SF      )
        this_subhalo.create_dataset( 'ObservationalEquiv', data = gradient_OE      )
        this_subhalo.create_dataset( 'StellarMass'       , data = sub_stellar_mass )
        this_subhalo.create_dataset( 'StarFormationRate' , data = sub_SFR          )
        this_subhalo.create_dataset( 'StellarHalfMassRad', data = sub_SHM          )
        this_subhalo.create_dataset( 'SFRHalfMassRad'    , data = sub_rsfr50       )
        this_subhalo.create_dataset( 'Redshift'          , data = snap2zTNG[snap]  )
    
if __name__ == "__main__":
    
    z_to_snap_TNG = {
        0:99,
        1:50,
        2:33,
        3:25,
        4:21,
        5:17,
        6:13,
        7:11,
        8:8
    }

    run  = 'L35n2160TNG'
    
    gradients_file = h5py.File( 'TNG_Gradients.hdf5', 'w' )
    
    gradients_file.create_dataset( 'run', data=run )
    
    for redshift in z_to_snap_TNG.keys():

        this_group = gradients_file.create_group( 'z=%s' %redshift )
        
        snap = z_to_snap_TNG[redshift]

        # Used for pointing to correct directory
        if snap > 25:
            person = 'zhemler'
        else:
            person = 'alexgarcia'

        DIR  = '/orange/paul.torrey/' + person + '/IllustrisTNG/' + run + '/output' + '/'

        save_data(snap, DIR, m_star_min=9.0, m_star_max=11.0, m_gas_min=9.0, where_to_save=this_group)
        
    gradients_file.close()