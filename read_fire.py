import sys
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import h5py

from sfms import calc_rsfr_io, calc_incl, trans, calcrsfr
from matplotlib.colors import LogNorm
from tqdm import tqdm

## FIRE reading tools
import gizmo_analysis as gizmo # https://bitbucket.org/awetzel/gizmo_analysis/src/master/
import halo_analysis as halo   # https://bitbucket.org/awetzel/halo_analysis/src/master/
import utilities as ut         # https://bitbucket.org/awetzel/utilities/src/master/

mpl.rcParams['text.usetex']        = True
# mpl.rcParams['text.latex.unicode'] = True
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


snap2zFIRE = {
    600: 0.0,
    382: 0.5,
    277: 1.0,
    214: 1.5,
    172: 2.0,
    142: 2.5,
    120: 3.0,
    102: 3.5,
    88 : 4.0,
    77 : 4.5,
    67 : 5.0,
    59 : 5.5,
    52 : 6.0,
    46 : 6.5,
    41 : 7.0,
    37 : 7.5,
    33 : 8.0,
    29 : 8.5,
    26 : 9.0,
    23 : 9.5,
    20 :10.0
}


def open_data(location, redshift, snap, ptType, tag, where_to_save=None, plot=False):
    
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
    
    r, rerr, oh, oherr, _rad_,_oh_, ri, ro, riprime = calczgrad_FIRE(part)
    
    gradient_SF = np.nan
    gradient_OE = np.nan
    this_rsfr   = np.nan
    this_SFR    = np.nan
    
    if len(r) > 0:    
        rsmall, rbig = riprime, ro
        fit_mask = ( r > rsmall ) & ( r < rbig ) & ~np.isnan(oh)
        if sum(fit_mask) > 0:
            gradient_SF, intercept_SF = np.polyfit( r[fit_mask], oh[fit_mask], 1 )

        gas_pos  = part['gas'].prop('host.distance')
        gas_sfr  = part['gas'].prop('sfr')
    
        # SFR within 10 kpc
        this_SFR = np.sum(gas_sfr[ part['gas'].prop('host.distance.total') < 10 ])
        
        this_rsfr = calcrsfr(gas_pos, gas_sfr)
        
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

            plt.text( 0.7, 0.9 , r'$z=%s$' %snap2zFIRE[snap], transform=plt.gca().transAxes )
            plt.text( 0.7, 0.8 , r'$\log M_* = %s$' %round( this_mass,2 ),transform=plt.gca().transAxes )

            plt.tight_layout()
            plt.savefig( './FIRE_diagnostics/' + str(tag) + '_FIRE_profile.pdf', bbox_inches='tight' )
        
        rsmall, rbig = 0.5 * this_RSHM, 2.0 * this_RSHM
        fit_mask = ( r > rsmall ) & ( r < rbig ) & ~np.isnan(oh)
        if sum(fit_mask) > 0:
            gradient_OE, intercept_OE = np.polyfit( r[fit_mask], oh[fit_mask], 1 )
        
    # else:
    #     print('Either too small SF region or no SFR (or both)')
    
    print('%'*100)
    print(
    gradient_SF,gradient_OE,this_mass,this_SFR,this_RSHM,this_rsfr,snap2zFIRE[snap]
    )
    print('%'*100)
    
    if where_to_save:
        
        #### THIS WILL ONLY WORK IF USING `high_redshift` sample
        this_subhalo = where_to_save.create_group('%s' %tag.replace('z5',''))
        
        this_subhalo.create_dataset( 'StarFormingRegion' , data = gradient_SF      )
        this_subhalo.create_dataset( 'ObservationalEquiv', data = gradient_OE      )
        this_subhalo.create_dataset( 'StellarMass'       , data = this_mass        )
        this_subhalo.create_dataset( 'StarFormationRate' , data = this_SFR         )
        this_subhalo.create_dataset( 'StellarHalfMassRad', data = this_RSHM        )
        this_subhalo.create_dataset( 'SFRHalfMassRad'    , data = this_rsfr        )
        this_subhalo.create_dataset( 'Redshift'          , data = snap2zFIRE[snap] )
    
def calczgrad_FIRE(part,species_name='gas',weight_name='mass',distance_max=30,distance_bin_width=0.1,
                   rotation=True,host_index=0,threshold=10):

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
    
    ############## TO DO ################
    # Need to add cuts here for SFR > 0 and (ro - riprime) > 1 kpc
    #####################################
    
    criteria1 = SFR_10_kpc > 0
    criteria2 = ( ro - riprime ) > 1
              
#     print( 'SFR within 10 kpc: %s Msun/yr' %SFR_10_kpc )
#     print( 'Size of SF region: %s kpc' %(ro - riprime) )
    
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
        
    return robs, stdrs, medohs, stdohs, rs, xyoh, ri, ro, riprime


if __name__ == "__main__":
    
    redshift_to_snap_core = {
         0.0:600,
         0.5:382,
         1.0:277,
         1.5:214,
         2.0:172,
         2.5:142,
         3.0:120,
         3.5:102,
         4.0:88 ,
         4.5:77 ,
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
    
    redshift_to_snap_high_res = {
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
    
    all_sims_core = [
        'm11d_res7100','m11q_res880' ,'m12m_res7100','m11e_res7100',
        'm12b_res7100','m12r_res7100','m11h_res7100','m12c_res7100',
        'm12f_res7100','m12w_res7100','m11b_res2100','m11i_res7100',
        'm12i_res7100','m12z_res4200'
    ]
    
    super_high_res = [
        'm09_res30','m10q_res30','m10v_res30'
    ]
    
    mergers = [
        'm12_elvis_RomeoJuliet_res3500','m12_elvis_RomulusRemus_res4000','m12_elvis_ThelmaLouise_res4000'
    ]
    
    all_sims_high_redshift = [
        'z5m09a','z5m10a','z5m10c','z5m10e','z5m11a','z5m11c','z5m11e','z5m11g','z5m11i','z5m12b','z5m12d',
        'z5m09b','z5m10b','z5m10d','z5m10f','z5m11b','z5m11d','z5m11f','z5m11h','z5m12a','z5m12c','z5m12e'
    ]
    
    SAVE_DATA   = True
    which_suite = 'core'
    
    try:
        h5py.File( 'FIRE_Gradients.hdf5', 'r+' )
    except:
        with h5py.File( 'FIRE_Gradients.hdf5', 'w' ) as f:
            print('file created')
    
    if which_suite == 'core':
        redshift_to_snap = redshift_to_snap_core
        all_sims         = all_sims_core
    elif which_suite == 'high_redshift':
        redshift_to_snap = redshift_to_snap_high_res
        all_sims         = all_sims_high_redshift
        
    for index, redshift in enumerate(redshift_to_snap.keys()):
    
        with h5py.File( 'FIRE_Gradients.hdf5', 'r+' ) as gradients_file:

            this_group = None
            if SAVE_DATA:
                if (index == 0):
                    this_suite = gradients_file.create_group( which_suite )
                else:
                    this_suite = gradients_file[ which_suite ]
                this_group = this_suite.create_group( 'z=%s' %redshift )
    
            for which_sim in all_sims:

                loc = '/orange/paul.torrey/FIRE/' + which_suite + '/' + which_sim 

                open_data(loc, redshift, redshift_to_snap[redshift], ptType=0, tag=which_sim,
                          where_to_save = this_group)
    
    # print('Hello World!')