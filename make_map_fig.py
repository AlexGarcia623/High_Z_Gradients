import sys
import h5py
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

from sfms import sfmscut, center, calc_rsfr_io, calc_incl, trans, calczgrad, calcrsfr, grad_valid, calc_sfr_prof
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from os import path, mkdir

# Images
sys.path.insert( 1, '/home/alexgarcia/torreylabtoolsPy3' )
import visualization.contour_makepic as makepic
import util.calc_hsml as calc_hsml

# EAGLE
EAGLE_SQL_TOOLS = '/home/alexgarcia/github/eagleSqlTools'
sys.path.insert(1,EAGLE_SQL_TOOLS)
import eagleSqlTools as sql
con = sql.connect( "rbs016", password="yigERne4" )

from read_eagle import read_eagle_subhalo

# Illustris (TNG)
import illustris_python as il

# FIRE 
import gizmo_analysis as gizmo # https://bitbucket.org/awetzel/gizmo_analysis/src/master/
import halo_analysis as halo   # https://bitbucket.org/awetzel/halo_analysis/src/master/
import utilities as ut         # https://bitbucket.org/awetzel/utilities/src/master/

mpl.rcParams['text.usetex']        = True
# mpl.rcParams['text.latex.unicode'] = True
mpl.rcParams['font.family']        = 'serif'
mpl.rcParams['font.size']          = 20

fs_large = 22
fs_small = 18

mpl.rcParams['font.size'] = fs_large
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

xh     = 7.600E-01
zo     = 3.500E-01
mh     = 1.6726219E-24
kb     = 1.3806485E-16
mc     = 1.270E-02

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

z_to_snap_EAGLE = {
    0:28,
    1:19,
    2:15,
    3:12,
    4:10,
    5: 8,
    6: 6,
    7: 5,
    8: 4
}
snap_to_file_name = { # For EAGLE
    4 :'z008p075',
    5 :'z007p050',
    6 :'z005p971',
    8 :'z005p037',
    10:'z003p984',
    12:'z003p017',
    15:'z002p012',
    19:'z001p004',
    28:'z000p000'
}
# All atomic species tracked by EAGLE
ALL_EAGLE_SPECIES = ["Carbon", "Helium", "Hydrogen", "Iron", "Magnesium", "Neon", "Nitrogen", "Oxygen", "Silicon"]

z_to_snap_high_redshift = {
     5.0:67 ,
     5.5:59 ,
     6.0:52 ,
     6.5:46 ,
     7.0:41 ,
     7.5:37 ,
     8.0:33 ,
     8.5:29 ,
     9.0:26 ,
     9.5:23 ,
    10.0:20 
}

black_blue= mpl.colors.LinearSegmentedColormap.from_list("custom",["black","#161368","#1B228A","#33E3FF"])

def tng(snap, ID, ax_map, ax_map2, ax_prof, fig):
    
    run     = 'L35n2160TNG'
    person  = 'alexgarcia'
    # person  = 'zhemler'
    out_dir = '/orange/paul.torrey/' + person + '/IllustrisTNG/' + run + '/output' + '/'
    
    hdr  = il.groupcat.loadHeader(out_dir, snap)
    box_size = hdr['BoxSize']
    scf      = hdr['Time']
    h        = hdr['HubbleParam']
    z        = hdr['Redshift']
    
    fields  = ['SubhaloMassType','SubhaloPos','SubhaloVel']
    prt     = il.groupcat.loadSubhalos(out_dir, snap, fields = fields)
    
    stellar_mass = prt['SubhaloMassType'] * 1.00E+10 / h
    
    sub = int( ID )
    
    this_stellar_mass = stellar_mass[ sub, 4 ]
    this_pos          = prt['SubhaloPos'][sub] #* (scf / h)
    this_vel          = prt['SubhaloVel'][sub] #* np.sqrt( scf )
        
    gas_pos   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Coordinates'      ])
    gas_vel   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Velocities'       ])
    gas_mass  = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Masses'           ])
    gas_sfr   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['StarFormationRate'])
    gas_rho   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['Density'          ])
    gas_met   = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metallicity'  ])
    ZO        = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metals'       ])[:,4]
    XH        = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metals'       ])[:,0]
    
    GFM_Metal = il.snapshot.loadSubhalo(out_dir, snap, sub, 0, fields = ['GFM_Metals'       ])
    
    gas_pos    = center(gas_pos, this_pos, box_size)
    gas_pos   *= (scf / h)
    gas_vel   *= np.sqrt(scf)
    gas_vel   -= this_vel
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
    
    sf_idx = gas_rho > 1.300E-01

    incl   = calc_incl(gas_pos[sf_idx], gas_vel[sf_idx], gas_mass[sf_idx], ri, ro)

    gas_pos  = trans(gas_pos, incl)
    gas_vel  = trans(gas_vel, incl)
    
    rmax = 10.0
    res  = 2160
    r, rerr, oh, oherr, _rprof_, _ohprof_, _massmap_, _ohmap_ = make_map(gas_pos, gas_mass, gas_rho, GFM_Metal,
                                                                         rmax, res, which='TNG')
    
    mappable=ax_map.imshow( _massmap_, cmap=plt.cm.inferno, extent=[-10,10,-10,10], norm=LogNorm() )
    cbaxes = fig.add_axes([0.060, 0.875, 0.125, 0.02]) #Add position (left, bottom, width, height)
    cb     = plt.colorbar(mappable, cax = cbaxes, orientation='horizontal') 
    cb.ax.tick_params(labelsize=15)
    
    ymin, ymax = 7,9.5
    hist_mask = (_ohprof_ > ymin) & (_ohprof_ < ymax)
    
    ax_prof.hist2d( _rprof_[hist_mask], _ohprof_[hist_mask], cmap=plt.cm.Greys, bins=(100,100) )
    ax_prof.plot( r, oh, color='blue', lw=2.0 )
    
    ymin, ymax = 7,9
    
    mappable=ax_map2.imshow( _ohmap_, cmap=black_blue, extent=[-10,10,-10,10], vmin=ymin, vmax=ymax )
    cbaxes = fig.add_axes([0.215, 0.875, 0.125, 0.02]) #Add position (left, bottom, width, height)
    cb     = plt.colorbar(mappable, cax = cbaxes, orientation='horizontal') 
    cb.ax.tick_params(labelsize=15)
    
    fit_mask = ( r > riprime ) & ( r < ro ) & ~np.isnan(oh)
    gradient_SF, intercept_SF = np.polyfit( r[fit_mask], oh[fit_mask], 1 )
    
    ax_prof.plot( r, gradient_SF * r + intercept_SF, color='red' )
            
    ax_prof.axvline( riprime, color='k', linestyle='--' )
    ax_prof.axvline( ro     , color='k', linestyle='--' )
    
    ax_prof.set_xlabel( r'${\rm Radius~(kpc)}$' )
    ax_prof.set_ylabel( r'$\log{\rm O/H} + 12~({\rm dex})$' )
    
    ax_prof.text( 0.975, 0.9   , r'${\rm TNG}~z=5$', transform=ax_prof.transAxes, ha='right', fontsize=fs_small )
    ax_prof.text( 0.975, 0.825 , r'$\log(M_*/M_\odot) = %s~$' %round(np.log10(this_stellar_mass),2),
                 transform=ax_prof.transAxes, ha='right', fontsize=fs_small )
    ax_prof.text( 0.975, 0.75, r'$\nabla = %s\,({\rm dex/kpc})$' %round(gradient_SF,2), transform=ax_prof.transAxes,
                 ha='right', fontsize=fs_small )
    
    ax_prof.set_yticks([7,8,9])
    
    plot_circle( ax_map, riprime )
    plot_circle( ax_map, ro )
    
    plot_circle( ax_map2, riprime )
    plot_circle( ax_map2, ro )
    

def eagle(snap, galaxy, group_cat,ax_map, ax_map2, ax_prof, fig, file_ext):
    
    run = 'RefL0100N1504'
    DIR = '/orange/paul.torrey/alexgarcia/EAGLE/'
    EAGLE = DIR + run  + '/' + 'snapshot_%s_%s' %(str(snap).zfill(3),snap_to_file_name[snap]) + '/' 
    
    sub_vel   = np.column_stack( (group_cat[b'sub_vel_x'], group_cat[b'sub_vel_y'], group_cat[b'sub_vel_z']) )
    sub_pos   = np.column_stack( (group_cat[b'sub_pos_x'], group_cat[b'sub_pos_y'], group_cat[b'sub_pos_z']) )
    sub_grnr  = np.array(group_cat[b'Grnr'],dtype=int)
    sub_Zgas  = np.array(group_cat[b'Zgas'])
    sub_Mstar = np.array(group_cat[b'Stellar_Mass'])
    sub_Mgas  = np.array(group_cat[b'Gas_Mass'])
    sub_RSHM  = np.array(group_cat[b'RSHM'])
    sub_SFR   = np.array(group_cat[b'SFR'])
        
    # Get bulk info about this galaxy
    this_sub_vel =          sub_vel  [np.where(sub_grnr == galaxy)][0]
    this_sub_pos =          sub_pos  [np.where(sub_grnr == galaxy)][0]
    this_Zgas    =          sub_Zgas [np.where(sub_grnr == galaxy)][0]
    this_Mstar   = np.log10(sub_Mstar[np.where(sub_grnr == galaxy)][0])
    this_RSHM    =          sub_RSHM [np.where(sub_grnr == galaxy)][0]
    this_Mgas    =          sub_Mgas [np.where(sub_grnr == galaxy)][0]
    this_SFR     =          sub_SFR  [np.where(sub_grnr == galaxy)][0]

    # Read in the particle data
    gas_data  = read_eagle_subhalo(EAGLE, galaxy, snap, PartType=0, file_ext=file_ext,
                                   keys=['Coordinates','Mass','Metallicity',
                                         'Density','StarFormationRate','Velocity',
                                         'OnEquationOfState','ElementAbundance'], run=run)

    gas_pos   = np.array(gas_data['Coordinates'      ])
    gas_vel   = np.array(gas_data['Velocity'         ])
    gas_mass  = np.array(gas_data['Mass'             ])
    gas_sfr   = np.array(gas_data['StarFormationRate'])
    gas_rho   = np.array(gas_data['Density'          ])
    gas_met   = np.array(gas_data['Metallicity'      ])
    gas_sf    = np.array(gas_data['OnEquationOfState'])
    scf       = np.array(gas_data['scf'              ])
    h         = np.array(gas_data['h'                ])
    box_size  = np.array(gas_data['boxsize'          ])
    Hydrogen  = np.array(gas_data['Hydrogen'         ])
    Oxygen    = np.array(gas_data['Oxygen'           ])
    
    this_sub_pos *= (scf)   # Convert from cMpc to Mpc
    this_sub_pos *= 1.00E+3 # Convert from Mpc to kpc

    # This is an artifact of the way I'm reading in data
    # If you add more keys, you need to add them here
    rows_without_nan = ~np.isnan(gas_pos).any(axis=1)
    gas_pos  = gas_pos [rows_without_nan]
    gas_vel  = gas_vel [rows_without_nan]
    gas_mass = gas_mass[rows_without_nan]
    gas_sfr  = gas_sfr [rows_without_nan]
    gas_rho  = gas_rho [rows_without_nan]
    gas_met  = gas_met [rows_without_nan]
    gas_sf   = gas_sf  [rows_without_nan]
    Hydrogen = Hydrogen[rows_without_nan]
    Oxygen   = Oxygen  [rows_without_nan]
    
    # Note that read eagle subhalo already gets us in physical units #
    gas_pos  *= 3.2407792896664E-22 # Convert from cm to kpc -- VERY SENSITVE TO THE VALUE USED???
    gas_pos   = center(gas_pos, this_sub_pos)
    gas_vel  *= 1.00E-05 # Convert from cm/s to km/s
    gas_vel  -= this_sub_vel
    gas_mass *= 5.00E-34 # Convert from g to Msun
    gas_rho  *= Hydrogen / mh
    gas_sfr  *= 5.00E-34 # Convert from g to Msun
    gas_sfr  /= 3.17E-08 # Convert from s to yr
    
    ri, ro = calc_rsfr_io(gas_pos, gas_sfr)
    ro2    = 2.000E+00 * ro
        
    riprime = ri + 0.25 * (ro - ri)
        
    this_rsfr50 = calcrsfr(gas_pos, gas_sfr)
    
    sf_idx  = gas_sf > 0

    incl    = calc_incl(gas_pos[sf_idx], gas_vel[sf_idx], gas_mass[sf_idx], ri, ro)
    gas_pos = trans(gas_pos, incl)
    
    GFM_Metal = np.zeros( (len(Hydrogen),len(ALL_EAGLE_SPECIES)) )
    GFM_Metal[:,2] = Hydrogen
    GFM_Metal[:,7] = Oxygen
    
    rmax = 10
    res  = 1080
    r, rerr, oh, oherr, _rprof_, _ohprof_, _massmap_, _ohmap_ = make_map( gas_pos, gas_mass, gas_rho, GFM_Metal,
                                                                          rmax, res,O_index = 7, H_index = 2, 
                                                                          EAGLE_rho=True, rhocutidx=sf_idx,
                                                                          which='EAGLE' )
    
    mappable=ax_map.imshow( _massmap_, cmap=plt.cm.inferno, extent=[-10,10,-10,10], norm=LogNorm() )
    cbaxes = fig.add_axes([0.37, 0.875, 0.125, 0.02]) #Add position (left, bottom, width, height)
    cb     = plt.colorbar(mappable, cax = cbaxes, orientation='horizontal') 
    cb.ax.tick_params(labelsize=15)
    
    ymin, ymax = 7,9.25
    hist_mask = (_ohprof_ > ymin) & (_ohprof_ < ymax)
    
    ax_prof.hist2d( _rprof_[hist_mask], _ohprof_[hist_mask], cmap=plt.cm.Greys, bins=(100,100) )
    ax_prof.plot( r, oh, color='blue', lw=2.0 )
    
    ymin, ymax = 7,9#ax_prof.get_ylim()
    
    mappable=ax_map2.imshow( _ohmap_, cmap=black_blue, extent=[-10,10,-10,10], vmin=ymin, vmax=ymax )
    cbaxes = fig.add_axes([0.5275, 0.875, 0.125, 0.02]) #Add position (left, bottom, width, height)
    cb     = plt.colorbar(mappable, cax = cbaxes, orientation='horizontal') 
    cb.ax.tick_params(labelsize=15)
    
    fit_mask = ( r > riprime ) & ( r < ro ) & ~np.isnan(oh)
    gradient_SF, intercept_SF = np.polyfit( r[fit_mask], oh[fit_mask], 1 )
    
    ax_prof.plot( r, gradient_SF * r + intercept_SF, color='red' )
            
    ax_prof.axvline( riprime, color='k', linestyle='--' )
    ax_prof.axvline( ro     , color='k', linestyle='--' )
    
    ax_prof.set_xlabel( r'${\rm Radius~(kpc)}$' )
    
    ax_prof.text( 0.975 , 0.9  , r'${\rm EAGLE}~z=5$', transform=ax_prof.transAxes, ha='right', fontsize=fs_small )
    ax_prof.text( 0.975 , 0.825, r'$\log(M_*/M_\odot) = %s~$' %round(this_Mstar,2),
                 transform=ax_prof.transAxes, ha='right', fontsize=fs_small )
    ax_prof.text( 0.975 , 0.75 , r'$\nabla = %s\,({\rm dex/kpc})$' %round(gradient_SF,2), transform=ax_prof.transAxes,
                 ha='right', fontsize=fs_small )
    
    ax_prof.set_yticks([7,8,9])
    
    plot_circle( ax_map, riprime )
    plot_circle( ax_map, ro )
    
    plot_circle( ax_map2, riprime )
    plot_circle( ax_map2, ro )

def fire(location, redshift, snap, ptType, tag, ax_map, ax_map2, ax_prof, fig):
    # Load in group catalogs
    grp_cat = halo.io.IO.read_catalogs( 'redshift', redshift, location, species='star' )
    
    high_res_selection = np.where(grp_cat['star.mass'] > 0)[0]
    
    positions = grp_cat['star.position'] [high_res_selection]
    masses    = grp_cat['star.mass']     [high_res_selection]
    ids       = grp_cat['id']            [high_res_selection]
    RSHMs     = grp_cat['star.radius.50'][high_res_selection]
    
    arg_most_massive = np.argmax(grp_cat['mass'][high_res_selection])
    
    this_mass     = np.log10(masses[arg_most_massive])
    this_position = positions[arg_most_massive]
    this_id       = ids      [arg_most_massive]
    this_RSHM     = RSHMs    [arg_most_massive]
    
    print('%'*100)
    print( 'Stellar Mass at z=%s: %s' %(redshift, this_mass ))
    print('%'*100)
    
    h   = grp_cat.Cosmology['hubble']
    z   = grp_cat.snapshot['redshift']
    scf = grp_cat.snapshot['scalefactor']
    
    # Load in particle data
    
    part = gizmo.io.Read.read_snapshots('gas', 'redshift', redshift, location)
    
    r, rerr, oh, oherr, _rprof_,_ohprof_, _xym_, _xyoh_, riprime, ro = make_map_FIRE(part)
    
    mappable=ax_map.imshow( _xym_, cmap=plt.cm.inferno, norm=LogNorm(), extent=[-10,10,-10,10])
    cbaxes = fig.add_axes([0.680, 0.875, 0.125, 0.02]) #Add position (left, bottom, width, height)
    cb     = plt.colorbar(mappable, cax = cbaxes, orientation='horizontal') 
    cb.ax.tick_params(labelsize=15)
    
    ymin, ymax = 8,9
    hist_mask = (_ohprof_ > ymin) & (_ohprof_ < ymax) 
    
    ax_prof.hist2d( _rprof_[hist_mask], _ohprof_[hist_mask], cmap=plt.cm.Greys, bins=(100,100) )
    ax_prof.plot( r, oh, color='blue', lw=2.0 )
    
    ymin, ymax = 8,9.5
    
    mappable=ax_map2.imshow( _xyoh_, cmap=black_blue, extent=[-10,10,-10,10], vmin=ymin, vmax=ymax)
    cbaxes = fig.add_axes([0.835, 0.875, 0.125, 0.02]) #Add position (left, bottom, width, height)
    cb     = plt.colorbar(mappable, cax = cbaxes, orientation='horizontal') 
    cb.ax.tick_params(labelsize=15)
    
    fit_mask = ( r > riprime ) & ( r < ro ) & ~np.isnan(oh)
    gradient_SF, intercept_SF = np.polyfit( r[fit_mask], oh[fit_mask], 1 )
    
    ax_prof.plot( r, gradient_SF * r + intercept_SF, color='red' )
            
    ax_prof.axvline( riprime, color='k', linestyle='--' )
    ax_prof.axvline( ro     , color='k', linestyle='--' )
    
    ax_prof.set_xlabel( r'${\rm Radius~(kpc)}$' )
    
    ax_prof.text( 0.975, 0.9  , r'${\rm FIRE}~z=5$', transform=ax_prof.transAxes, ha='right', fontsize=fs_small )
    ax_prof.text( 0.975, 0.825, r'$\log(M_*/M_\odot) = %s~$' %round(this_mass,2),
                 transform=ax_prof.transAxes, ha='right', fontsize=fs_small )
    ax_prof.text( 0.975, 0.75 , r'$\nabla = %s\,({\rm dex/kpc})$' %round(gradient_SF,2), transform=ax_prof.transAxes,
                 ha='right', fontsize=fs_small )
    
    ax_prof.set_yticks([8,9])
    
    plot_circle( ax_map, riprime )
    plot_circle( ax_map, ro )
    
    plot_circle( ax_map2, riprime )
    plot_circle( ax_map2, ro )

def read_eagle_subhalo(file_path, group_index, snap, keys=['GroupNumber','SubGroupNumber'],
                       PartType=0, nfiles=256, sub_group_index=0, file_ext='z000p000',
                       run='RefL0100N1504',species=["Oxygen","Hydrogen"]):
    req_keys = ['GroupNumber','SubGroupNumber']
    for _k in req_keys:
        if _k not in keys:
            keys.append(_k)
    
    # Remove the `ElementAbundance` key and add the species
    if "ElementAbundance" in keys:
        keys.remove("ElementAbundance")
        for atom in species:
            keys.append( atom )
    
    return_dic = {}
    
    # Check where the galaxy particle data is saved
    galaxy_file_locator = np.load( './%s_SF_galaxies/snap_%s/file_lookup.npy' %(run,str(snap).zfill(3)) ,
                                  allow_pickle=True, encoding='bytes').item()
    
    files_to_look = galaxy_file_locator[group_index] 
    
    # Get the particle data
    for file_counter in files_to_look:
        fname = file_path + 'snap_%s_%s.%s.hdf5' % (str(snap).zfill(3), file_ext, file_counter)
        with h5py.File(fname, 'r') as f:
            a       = f[b'Header'].attrs.get('Time')
            BoxSize = f[b'Header'].attrs.get('BoxSize')
            z       = (1.00E+00 / a - 1.00E+00)
            h       = f[b'Header'].attrs.get('HubbleParam')
                        
            pt = 'PartType' + str(int(PartType))
            
            this_grnrs  = np.array(f[pt][b'GroupNumber'])
            this_subnrs = np.array(f[pt][b'SubGroupNumber'])
            
            subhalo_mask = ( (this_grnrs  == group_index) &
                             (this_subnrs == sub_group_index) )
    
            subhalo_indices   = np.where(subhalo_mask)[0]
            subhalo_mask_bool = np.zeros_like(subhalo_mask, dtype=bool)
            subhalo_mask_bool[subhalo_indices] = True

            data_dict = {key: mask_array(f[pt], subhalo_mask_bool, key) for key in keys}
            
            for key in keys:
                # Get conversion factor unless its an element
                if key not in ALL_EAGLE_SPECIES:
                    cgs = f[pt][key].attrs.get('CGSConversionFactor')
                    axp = f[pt][key].attrs.get('aexp-scale-exponent')
                    hxp = f[pt][key].attrs.get('h-scale-exponent')

                if key not in return_dic:
                    return_dic[key] = np.multiply(np.array(data_dict[key]), cgs * a**axp * h ** hxp )
                else:
                    return_dic[key] = np.concatenate((return_dic[key],
                                                      np.multiply(np.array(data_dict[key]), cgs * a**axp * h ** hxp )),
                                                      axis = 0
                                                    )            
        
    # Get header parameters
    with h5py.File(fname, 'r') as f:
        return_dic['scf']     = a      
        return_dic['boxsize'] = BoxSize
        return_dic['z']       = z      
        return_dic['h']       = h      
        
    return return_dic

def mask_array(particle_data, mask, key, chunk_size=5000):
    # Handles Element Abundance Group 
    if (key in ALL_EAGLE_SPECIES):
        data = particle_data["ElementAbundance"][key]
    else:
        data = particle_data[key]
    
    # Get only the data we want from the particle catalog
    if (data.ndim > 1):
        result = np.empty_like(data)
        for i in range(0, data.shape[0], chunk_size):
            chunk_data = data[i:i+chunk_size]
            result[i:i+chunk_size] = np.where(mask[i:i+chunk_size, None], chunk_data, np.nan)
    else:
        result = np.where(mask, data, np.nan)        
    return result

def make_map(pos, m, hrho, zm9, rmax, res, O_index=4, H_index=0,
             EAGLE_rho=False, rhocutidx=None, which='TNG',save=False):
    # Search area. First 0.05 kpc, then 0.125, 0.25, 0.5, and finally 1.0 kpc
    bpass  = [5.000E-02, 1.250E-01, 2.500E-01, 5.000E-01, 1.000E+00]
    
    # Different resolutions need difference pixel sizes
    if (res == 2160): # TNG50-1 equiv.
        pixl   =  1.000E-01
        nmin   =  16 #min particles needed to be observationally equivalent
    elif (res == 1080): # TNG50-2 equiv.
        pixl   =  2.500E-01
        nmin   =  8
    elif (res == 540): # TNG50-3 equiv.
        pixl   =  5.000E-01
        nmin   =  4
    elif (res == 270): # TNG50-4 equiv.
        pixl   =  1.000E+00
        nmin   =  2
        
    dr     =  1.00E-01 #kpc
    pixa   =  pixl**2.000E+00
    sigcut =  1.000E+00
    rhocut =  1.300E-01
    mcut   =  1.000E+01**sigcut * (pixa*1.000E+06)
    
    # Create map
    pixlims = np.arange(-rmax, rmax + pixl, pixl)
    pix   = len(pixlims) - 1
    pixcs = pixlims[:-1] + (pixl / 2.000E+00)
    rs    = np.full((pix, pix), np.nan, dtype = float)
    for r in range(0, pix):
        for c in range(0, pix):
            rs[r,c] = np.sqrt(pixcs[r]**2.000E+00 + 
                              pixcs[c]**2.000E+00 )
    
    if save:
        hsml = calc_hsml.get_particle_hsml( pos[:,0], pos[:,1], pos[:,2], DesNgb=32  )

        n_pixels = 720
        massmap,image = makepic.contour_makepic( pos[:,0], pos[:,1], pos[:,2], hsml, m,
            xlen = rmax,
            pixels = n_pixels, set_aspect_ratio = 1.0,
            set_maxden = 1.0e10, ## (gadget units, 10^10 msun/kpc^2 = 10^4 msun/pc^2)
            set_dynrng = 1.0e4  )

        np.save( '%s_massmap.npy' %which, massmap )
        
        Omassmap,image = makepic.contour_makepic( pos[:,0], pos[:,1], pos[:,2], hsml,
            np.multiply(m, zm9[:,O_index]),
            xlen = rmax,
            pixels = n_pixels, set_aspect_ratio = 1.0,
            set_maxden = 1.0e10, ## (gadget units, 10^10 msun/kpc^2 = 10^4 msun/pc^2)
            set_dynrng = 1.0e10  )
        
        Hmassmap,image = makepic.contour_makepic( pos[:,0], pos[:,1], pos[:,2], hsml,
            np.multiply(m, zm9[:,H_index]),
            xlen = rmax,
            pixels = n_pixels, set_aspect_ratio = 1.0,
            set_maxden = 1.0e10, ## (gadget units, 10^10 msun/kpc^2 = 10^4 msun/pc^2)
            set_dynrng = 1.0e10  )
        
        OH_map = (Omassmap / Hmassmap) * (1.00 / 16.00)
        OH_map = np.log10(OH_map) + 12
        
        np.save( '%s_metallicitymap.npy' %which, OH_map )
        
    else:
        massmap = np.load( '%s_massmap.npy' %which )
        OH_map  = np.load( '%s_metallicitymap.npy' %which )
    
    if EAGLE_rho:
        rhoidx = rhocutidx
    else:
        rhoidx = hrho > rhocut
    # Mass map
    xym, x, y = np.histogram2d(pos[:,0], pos[:,1], weights = m, bins = [pixlims, pixlims])
    # Oxygen map
    xyo, x, y = np.histogram2d(pos[rhoidx,0], pos[rhoidx,1], weights = np.multiply(m[rhoidx], zm9[rhoidx,O_index]), bins = [pixlims, pixlims])
    # Hydrogen map
    xyh, x, y = np.histogram2d(pos[rhoidx,0], pos[rhoidx,1], weights = np.multiply(m[rhoidx], zm9[rhoidx,H_index]), bins = [pixlims, pixlims])
    
    xym       = np.transpose(xym)
    xyo       = np.transpose(xyo)
    xyh       = np.transpose(xyh)
    rs        = np.ravel( rs)
    xym       = np.ravel(xym)
    xyo       = np.ravel(xyo)
    xyh       = np.ravel(xyh)
    xyh[xyh < 1.000E-12] = np.nan
    
    xyoh     = xyo / xyh
    cutidx   =(xym > mcut) & (~np.isnan(xyoh)) & (xyoh > 0)
    _xyoh_   = np.where( cutidx, xym, np.nan )
    rs       =   rs[cutidx]
    xyoh     = xyoh[cutidx]
    xyoh     = np.log10(xyoh * (1.000E+00 / 1.600E+01)) + 1.200E+01
    rsortidx = np.argsort(rs)
    rs       =   rs[rsortidx]
    xyoh     = xyoh[rsortidx]
    
    robs   = np.arange(0.000E+00, rs[-1], dr)
    lgrad  = len(robs)
    stdrs  = np.zeros(lgrad,         dtype = float)
    medohs =  np.full(lgrad, np.nan, dtype = float)
    stdohs =  np.full(lgrad, np.nan, dtype = float)
    for i in range(0, lgrad):
        goodflag = False
        for j in range(0, len(bpass)):
            idx = ((rs > robs[i] - bpass[j]) & 
                   (rs < robs[i] + bpass[j]))
            if (np.sum(idx) >= nmin):
                goodflag = True
                break
        if (goodflag):
            stdrs [i] =    np.std(  rs[idx])
            medohs[i] = np.median(xyoh[idx])
            stdohs[i] =    np.std(xyoh[idx])
        
    return robs, stdrs, medohs, stdohs, rs, xyoh, massmap, OH_map

def make_map_FIRE( part,species_name='gas',weight_name='mass',distance_max=10,distance_bin_width=0.1,
                   rotation=True,host_index=0,threshold=10,save=False ):

    ### For Romeo might need to check hosts??
    
    bpass = [5.000E-02, 1.250E-01, 2.500E-01, 5.000E-01, 1.000E+00]
    nmin  = 16
    
    # get distance limits for plot
    position_limits = [[-distance_max, distance_max] for _ in range(2)]
    position_limits = np.array(position_limits)

    # get array of particle indices
    part_indices = ut.array.get_arange(part[species_name]['position'].shape[0])

    # get positions relative to host galaxy center
    host_name = ut.catalog.get_host_name(host_index)
    positions = part[species_name].prop(f'{host_name}.distance', part_indices)

    gas_sfr  = part['gas'].prop('sfr', part_indices)
    
    SFR_10_kpc = np.sum(gas_sfr[ part['gas'].prop(f'{host_name}.distance.total', part_indices) < 10 ])

    gas_mass = part['gas'].prop('mass', part_indices)
    gas_pos  = part['gas'].prop(f'{host_name}.distance', part_indices)
    gas_vel  = part['gas'].prop(f'{host_name}.velocity', part_indices)
    
    ZO = part['gas'].prop( 'massfraction.o', part_indices)
    XH = part['gas'].prop( 'massfraction.h', part_indices)

    ri, ro  = calc_rsfr_io(gas_pos, gas_sfr)
    riprime = ri + 0.25 * (ro - ri)
    
    criteria1 = SFR_10_kpc > 0
    criteria2 = ( ro - riprime ) > 1
    
    robs, stdrs, medohs, stdohs, rs, xyoh = [],[],[],[],[],[]
    
    if criteria1 & criteria2:
    
        incl = calc_incl(gas_pos, gas_vel, gas_mass, ri, ro)

        positions = trans(positions, incl)

        # weight particles by some property?
        weights = None
        if weight_name:
            weights = part[species_name].prop(weight_name, part_indices)

        # keep only particles within distance limits along each dimension
        masks = positions[:, 0] <= distance_max
        for dimen_i in [0,1]: # I only care about x, y
            masks *= (positions[:, dimen_i] >= -distance_max) * (positions[:, dimen_i] <= distance_max)

        # keep only positions and weights within distance limits
        positions = positions[masks]
        if weights is not None:
            weights = weights[masks]

        # get number of bins (pixels) along each dimension
        position_bin_number = int(np.round(2 * np.max(distance_max) / distance_bin_width))

        if save:
            hsml = calc_hsml.get_particle_hsml( positions[:,0], positions[:,1], positions[:,2], DesNgb=32  )

            n_pixels = 720
            massmap,image = makepic.contour_makepic( positions[:,0], positions[:,1], positions[:,2], hsml, gas_mass,
                xlen = distance_max,
                pixels = n_pixels, set_aspect_ratio = 1.0,
                set_maxden = 1.0e10, ## (gadget units, 10^10 msun/kpc^2 = 10^4 msun/pc^2)
                set_dynrng = 1.0e4  )
            
            np.save( 'FIRE_massmap.npy', massmap )
            
            Omassmap,image = makepic.contour_makepic( positions[:,0], positions[:,1], positions[:,2], hsml,
                np.multiply( gas_mass, ZO ),
                xlen = distance_max,
                pixels = n_pixels, set_aspect_ratio = 1.0,
                set_maxden = 1.0e10, ## (gadget units, 10^10 msun/kpc^2 = 10^4 msun/pc^2)
                set_dynrng = 1.0e12  )
            
            Hmassmap,image = makepic.contour_makepic( positions[:,0], positions[:,1], positions[:,2], hsml,
                np.multiply( gas_mass, XH ),
                xlen = distance_max,
                pixels = n_pixels, set_aspect_ratio = 1.0,
                set_maxden = 1.0e10, ## (gadget units, 10^10 msun/kpc^2 = 10^4 msun/pc^2)
                set_dynrng = 1.0e12  )
            
            OH_map = Omassmap / Hmassmap
            OH_map = np.log10(OH_map) + 12
            
            np.save( 'FIRE_metallicitymap.npy', OH_map )
        else:
            massmap = np.load('FIRE_massmap.npy')
            OH_map  = np.load('FIRE_metallicitymap.npy')
        
        ## Create maps
        # mass map
        xym, hist_xs, hist_ys = np.histogram2d(
            positions[:, 0],
            positions[:, 1],
            position_bin_number,
            position_limits,
            weights=gas_mass[masks],
        )
        # position map
        pixlims = np.arange(-distance_max, distance_max + distance_bin_width, distance_bin_width)
        pix   = len(pixlims) - 1
        pixcs = pixlims[:-1] + (distance_bin_width / 2.000E+00)
        rs = np.full((pix,pix), np.nan, dtype = float)
        for r in range(0, pix):
            for c in range(0, pix):
                rs[r,c] = np.sqrt(pixcs[r]**2.000E+00 + 
                                  pixcs[c]**2.000E+00 )
        # mass weighted oxygen map
        xyo, x, y = np.histogram2d(
            positions[:, 0],
            positions[:, 1],
            position_bin_number,
            position_limits,
            weights=np.multiply( gas_mass[masks], ZO[masks] ),
        )
        # mass weighted hydrogen map
        xyh, x, y = np.histogram2d(
            positions[:, 0],
            positions[:, 1],
            position_bin_number,
            position_limits,
            weights=np.multiply( gas_mass[masks], XH[masks] ),
        )

        # convert to surface density (and in 1/pc^2)
        xym /= np.diff(hist_xs)[0] * np.diff(hist_ys)[0]
        xym *= 1e-6

        xym = np.transpose(xym)
        xyo = np.transpose(xyo)
        xyh = np.transpose(xyh)
        rs  = np.ravel( rs)
        xym = np.ravel(xym)
        xyo = np.ravel(xyo)
        xyh = np.ravel(xyh)
        xyh[xyh < 1.000E-12] = np.nan
        xyoh= xyo / xyh

        cutidx   = (xym > threshold) & (~np.isnan(xyoh))
        _xyoh_   = np.where( cutidx, xym, np.nan )
        rs       =   rs[cutidx]
        xyoh     = xyoh[cutidx]
        xyoh     = np.log10(xyoh * (1.000E+00 / 1.600E+01)) + 1.200E+01
        rsortidx = np.argsort(rs)
        rs       =   rs[rsortidx]
        xyoh     = xyoh[rsortidx]

        dr     = 1.00E-01
        robs   = np.arange(0.000E+00, rs[-1], dr)
        lgrad  = len(robs)
        stdrs  = np.zeros(lgrad,         dtype = float)
        medohs =  np.full(lgrad, np.nan, dtype = float)
        stdohs =  np.full(lgrad, np.nan, dtype = float)
        for i in range(0, lgrad):
            goodflag = False
            for j in range(0, len(bpass)):
                idx = ((rs > robs[i] - bpass[j]) & 
                       (rs < robs[i] + bpass[j]))
                if (np.sum(idx) >= nmin):
                    goodflag = True
                    break
            if (goodflag):
                stdrs [i] =    np.std(  rs[idx])
                medohs[i] = np.median(xyoh[idx])
                stdohs[i] =    np.std(xyoh[idx])
        
    return robs, stdrs, medohs, stdohs, rs, xyoh, massmap, OH_map, riprime, ro

def plot_circle( ax, r, color='white', lw=2.0 ):
    
    theta = np.linspace( 0, 2*np.pi, 100 )
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    ax.plot( x, y, color=color, lw=lw, alpha=0.5 )

if __name__ == "__main__":
    
    redshift = 5.0
    
    heights = [0.2,1,1]
    widths  = [1,1,1,1,1,1]
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    fig, axs = plt.subplots( 3, 6, figsize=(15, 8), gridspec_kw=gs_kw )
    
    TNG_galaxy_ID   = 6099
    EAGLE_galaxy_ID = 50
    FIRE_galaxy_ID  = 'm12c'
    
    #### TNG ####
    gs = axs[2, 0].get_gridspec()
    # remove the underlying axes
    for ax in axs[2:, 1]:
        ax.axis('off')
        ax.remove()
    axbig_TNG = fig.add_subplot(gs[2, 0:2])
    
    print('TNG')
    tng(z_to_snap_TNG[int(redshift)], TNG_galaxy_ID, axs[1,0], axs[1,1], axbig_TNG, fig)
    axbig_TNG.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(6))
    ############
    
    #### EAGLE ####
    gs = axs[2, 2].get_gridspec()
    # remove the underlying axes
    for ax in axs[2:,3]:
        ax.axis('off')
        ax.remove()
    axbig_EAGLE = fig.add_subplot(gs[2, 2:4])
    
    EAGLE_dir     = './RefL0100N1504_SF_galaxies/snap_%s/grp_cat.npy' %str(z_to_snap_EAGLE[redshift]).zfill(3)
    eagle_grp_cat = np.load(EAGLE_dir, allow_pickle=True, encoding='bytes' ).item()
    
    print('EAGLE')
    eagle( z_to_snap_EAGLE[redshift],EAGLE_galaxy_ID, eagle_grp_cat,
           axs[1,2], axs[1,3], axbig_EAGLE, fig, file_ext=snap_to_file_name[ z_to_snap_EAGLE[redshift] ] )
    axbig_EAGLE.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(6))
    ############
    
    #### FIRE ####
    gs = axs[2, 4].get_gridspec()
    # remove the underlying axes
    for ax in axs[2:, 5]:
        ax.axis('off')
        ax.remove()
    axbig_FIRE = fig.add_subplot(gs[2, 4:])
    
    which_suite = 'high_redshift'
    which_sim   = 'z5' + FIRE_galaxy_ID
    loc = '/orange/paul.torrey/FIRE/' + which_suite + '/' + which_sim 
    
    print('FIRE')
    fire(loc, redshift, z_to_snap_high_redshift[redshift], ptType=0, tag=which_sim,
         ax_map=axs[1,4], ax_map2=axs[1,5], ax_prof=axbig_FIRE, fig=fig)
    axbig_FIRE.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(6))
    ############
    
    axs[0,0].text( 0.5,0.4, r'$\Sigma_{\rm gas}~(M_\odot/{\rm kpc}^2)$', transform=axs[0,0].transAxes, ha='center',
                   fontsize=fs_large-2)
    axs[0,2].text( 0.5,0.4, r'$\Sigma_{\rm gas}~(M_\odot/{\rm kpc}^2)$', transform=axs[0,2].transAxes, ha='center',
                   fontsize=fs_large-2)
    axs[0,4].text( 0.5,0.4, r'$\Sigma_{\rm gas}~(M_\odot/{\rm kpc}^2)$', transform=axs[0,4].transAxes, ha='center',
                   fontsize=fs_large-2)
    
    axs[0,1].text( 0.5,0.4, r'$\log{\rm O/H} + 12~({\rm dex})$', transform=axs[0,1].transAxes, ha='center',
                   fontsize=fs_large-2)
    axs[0,3].text( 0.5,0.4, r'$\log{\rm O/H} + 12~({\rm dex})$', transform=axs[0,3].transAxes, ha='center',
                   fontsize=fs_large-2)
    axs[0,5].text( 0.5,0.4, r'$\log{\rm O/H} + 12~({\rm dex})$', transform=axs[0,5].transAxes, ha='center',
                   fontsize=fs_large-2)
    
    for ax in axs[0,:]:
        ax.axis('off')
        
    for ax in [axs[2,0],axs[2,2],axs[2,4]]:
        ax.axis('off')
        
    for ax in axs[1,:]:
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.tight_layout()
    
    axs[1,0].text( 0.7  , 0.875, r'${\rm TNG}$'  , transform=axs[1,0].transAxes,
                  color='white', ha='left', fontsize=fs_small+2 )
    axs[1,0].text( 0.975, 0.02 , r'${\rm Gas~Mass}$', transform=axs[1,0].transAxes,
                  color='white', fontsize=fs_small+2, ha='right' )
    axs[1,1].text( 0.7  , 0.875 , r'${\rm TNG}$'  , transform=axs[1,1].transAxes,
                  color='white', ha='left', fontsize=fs_small+2 )
    axs[1,1].text( 0.975, 0.02 , r'${\rm Metallicity}$', transform=axs[1,1].transAxes,
                  color='white', fontsize=fs_small+2, ha='right' )
    
    axs[1,2].text( 0.525, 0.875, r'${\rm EAGLE}$', transform=axs[1,2].transAxes,
                  color='white', ha='left', fontsize=fs_small+2 )
    axs[1,2].text( 0.975, 0.02 , r'${\rm Gas~Mass}$', transform=axs[1,2].transAxes,
                  color='white', fontsize=fs_small+2, ha='right' )
    axs[1,3].text( 0.525, 0.875, r'${\rm EAGLE}$', transform=axs[1,3].transAxes,
                  color='white', ha='left', fontsize=fs_small+2 )
    axs[1,3].text( 0.975, 0.02 , r'${\rm Metallicity}$', transform=axs[1,3].transAxes,
                  color='white', fontsize=fs_small+2, ha='right' )
    
    axs[1,4].text( 0.65 , 0.875, r'${\rm FIRE}$' , transform=axs[1,4].transAxes,
                  color='white', ha='left', fontsize=fs_small+2 )
    axs[1,4].text( 0.975, 0.02 , r'${\rm Gas~Mass}$', transform=axs[1,4].transAxes,
                  color='white', fontsize=fs_small+2, ha='right' )
    axs[1,5].text( 0.65 , 0.875, r'${\rm FIRE}$' , transform=axs[1,5].transAxes,
                  color='white', ha='left', fontsize=fs_small+2 )
    axs[1,5].text( 0.975, 0.02 , r'${\rm Metallicity}$' , transform=axs[1,5].transAxes,
                  color='white', fontsize=fs_small+2, ha='right' )
    
    plt.subplots_adjust( wspace=0.2, hspace=-0.2 )
    plt.savefig( './figures/' + 'all_maps.pdf',bbox_inches='tight' )