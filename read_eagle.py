import sys
import h5py
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

EAGLE_SQL_TOOLS = '/home/alexgarcia/github/eagleSqlTools'
sys.path.insert(1,EAGLE_SQL_TOOLS)
import eagleSqlTools as sql
con = sql.connect( "rbs016", password="yigERne4" )

from sfms import sfmscut, center, calc_rsfr_io, calc_incl, trans, calczgrad, calcrsfr, grad_valid
from tqdm import tqdm

from os import path, mkdir

xh = 7.600E-01
zo = 3.500E-01
mh = 1.6726219E-24
kb = 1.3806485E-16
mc = 1.270E-02

snap2zEAGLE = {
    28:0,
    19:1,
    15:2,
    12:3,
    10:4,
     8:5,
     6:6,
     5:7,
     4:8
}

ALL_EAGLE_SPECIES = ["Carbon", "Helium", "Hydrogen", "Iron", "Magnesium", "Neon", "Nitrogen", "Oxygen", "Silicon"]

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

def get_SF_galaxies(snap, simulation_run='RefL0100N1504', m_star_min=8.0, m_star_max=10.5, m_gas_min=8.5,
                    verbose=False):
    # Create Query of EAGLE database
    myQuery = '''SELECT \
        SF_Metallicity as Zgas,\
        Stars_Metallicity as Zstar,\
        StarFormationRate as SFR,\
        MassType_Star as Stellar_Mass,\
        MassType_Gas as Gas_Mass,\
        HalfMassRad_Gas as R_gas,\
        HalfMassRad_Star as R_star,\
        GroupNumber as Grnr,\
        CentreOfMass_x as sub_pos_x,\
        CentreOfMass_y as sub_pos_y,\
        CentreOfMass_z as sub_pos_z,\
        Velocity_x as sub_vel_x,\
        Velocity_y as sub_vel_y,\
        Velocity_z as sub_vel_z,\
        HalfMassRad_Star as RSHM\
    FROM \
        %s_SubHalo as SH\
    WHERE \
        SnapNum = %s \
        and SH.SubGroupNumber = 0 
        and SH.StarFormationRate > 0.0\
        and SH.MassType_Star > 1E8\
        and SH.SubGroupNumber = 0''' %(simulation_run, snap) 

    if verbose:
        print('Starting Query... snapshot %s' %snap)
        sys.stdout.flush()
    myData = sql.execute_query(con, myQuery)
    if verbose:
        print('Query Complete')
        sys.stdout.flush()
    
    # Select SF galaxies
    gas_mass  = np.array(myData['Gas_Mass'][:])
    star_mass = np.array(myData['Stellar_Mass'][:])
    SFR       = np.array(myData['SFR'][:])
    
    sfms_idx = sfmscut(star_mass, SFR, m_star_min=8.0, m_star_max=10.5, m_gas_min=8.5)
    
    SFG_mask = ((star_mass > 1.00E+01**(m_star_min)) &
                (star_mass < 1.00E+01**(m_star_max)) &
                (gas_mass  > 1.00E+01**(m_gas_min))  &
                (sfms_idx))
    
    saveData = {}
    
    # You can change these, but need to be changed above accordingly!
    keys = ['Zgas','Zstar','SFR','Stellar_Mass','Gas_Mass',
            'R_gas','R_star','Grnr','sub_pos_x','sub_pos_y','sub_pos_z',
            'sub_vel_x','sub_vel_y','sub_vel_z','RSHM']
    
    # Save only star forming galaxies
    for key in keys:
        saveData[key] = np.array(myData[key])[SFG_mask]
    
    return saveData

def read_eagle_subhalo(file_path, group_index, snap, keys=['GroupNumber','SubGroupNumber'],
                       PartType=0, nfiles=256, sub_group_index=0, file_ext='z000p000',
                       run='RefL0100N1504',species=["Oxygen","Hydrogen"]):
    req_keys = ['GroupNumber','SubGroupNumber']
    for _k in req_keys:
        if _k not in keys:
            keys.append(_k)
    
    # Add the tag as many times as needed
    if "ElementAbundance" in keys:
        keys.remove("ElementAbundance")
        for atom in species:
            keys.append( atom )
    
    return_dic = {}
    
    galaxy_file_locator = np.load( './%s_SF_galaxies/snap_%s/file_lookup.npy' %(run,str(snap).zfill(3)) ,
                                  allow_pickle=True).item()
    
    files_to_look = galaxy_file_locator[group_index] 
    
    for file_counter in files_to_look:
        fname = file_path + 'snap_%s_%s.%s.hdf5' % (str(snap).zfill(3), file_ext, file_counter)
        with h5py.File(fname, 'r') as f:
            a       = f['Header'].attrs.get('Time')
            BoxSize = f['Header'].attrs.get('BoxSize')
            z       = (1.00E+00 / a - 1.00E+00)
            h       = f['Header'].attrs.get('HubbleParam')
                        
            pt = 'PartType' + str(int(PartType))
            
            this_grnrs  = np.array(f[pt]['GroupNumber'])
            this_subnrs = np.array(f[pt]['SubGroupNumber'])
            
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
    
    # Mask the array 
    if (data.ndim > 1):
        result = np.empty_like(data)
        for i in range(0, data.shape[0], chunk_size):
            chunk_data = data[i:i+chunk_size]
            result[i:i+chunk_size] = np.where(mask[i:i+chunk_size, None], chunk_data, np.nan)
    else:
        result = np.where(mask, data, np.nan)        
    return result
    
def get_which_files(file_path, group_indeces, snap, nfiles=256,
                    file_ext='z000p000',save_loc=''):
    all_files = {}

    for file_counter in range(nfiles):
        file_name = file_path + 'snap_{0}_{1}.{2}.hdf5'.format(str(snap).zfill(3), file_ext, file_counter)
        with h5py.File(file_name, 'r') as f:
            subgroup_mask = f['PartType0']['SubGroupNumber'][:] == 0
            unique_gals   = np.unique(f['PartType0']['GroupNumber'][subgroup_mask])

            overlap_gals  = np.intersect1d(unique_gals, group_indeces, assume_unique=True)
            
            for gal in overlap_gals:
                if gal not in all_files:
                    all_files[gal] = [file_counter]
                else:
                    all_files[gal].append(file_counter)

    all_files = {gal: np.array(files) for gal, files in all_files.items()}
    np.save(save_loc, all_files)
    
def reduce_eagle(snap, galaxy, run, group_cat, file_ext='', EAGLE='', res=1080):
    rmax = 1.00E+02
    
    sub_vel   = np.column_stack( (group_cat['sub_vel_x'], group_cat['sub_vel_y'], group_cat['sub_vel_z']) )
    sub_pos   = np.column_stack( (group_cat['sub_pos_x'], group_cat['sub_pos_y'], group_cat['sub_pos_z']) )
    sub_grnr  = np.array(group_cat['Grnr'],dtype=int)
    sub_Zgas  = np.array(group_cat['Zgas'])
    sub_Mstar = np.array(group_cat['Stellar_Mass'])
    sub_RSHM  = np.array(group_cat['RSHM'])

    print( 'Doing Galaxy: %s' %galaxy )

    this_sub_vel =          sub_vel  [np.where(sub_grnr == galaxy)][0]
    this_sub_pos =          sub_pos  [np.where(sub_grnr == galaxy)][0]
    this_Zgas    =          sub_Zgas [np.where(sub_grnr == galaxy)][0]
    this_Mstar   = np.log10(sub_Mstar[np.where(sub_grnr == galaxy)][0])
    this_RSHM    =          sub_RSHM [np.where(sub_grnr == galaxy)][0]

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
    # GFM_metal = np.array(gas_data['ElementAbundance' ])

    this_sub_pos *= scf # Convert from cMpc to Mpc
    this_sub_pos *= 1.00E+3 # Convert from Mpc to kpc

    rows_without_nan = ~np.isnan(gas_pos).any(axis=1)
    print('Rows without nan:',sum(rows_without_nan))

    gas_pos  = gas_pos [rows_without_nan]
    gas_vel  = gas_vel [rows_without_nan]
    gas_mass = gas_mass[rows_without_nan]
    gas_sfr  = gas_sfr [rows_without_nan]
    gas_rho  = gas_rho [rows_without_nan]
    gas_met  = gas_met [rows_without_nan]
    gas_sf   = gas_sf  [rows_without_nan]
    Hydrogen = Hydrogen[rows_without_nan]
    Oxygen   = Oxygen  [rows_without_nan]

    gas_pos  *= 3.24e-22 # Convert from cm to kpc
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
    
    sf_idx = gas_sf > 0
    print('SF gas particles in this halo:',sum(sf_idx))

    incl    = calc_incl(gas_pos[sf_idx], gas_vel[sf_idx], gas_mass[sf_idx], ri, ro)
    gas_pos = trans(gas_pos, incl)
    
    GFM_Metal = np.zeros( (len(Hydrogen),len(ALL_EAGLE_SPECIES)) )
    GFM_Metal[:,2] = Hydrogen
    GFM_Metal[:,7] = Oxygen
    
    r, rerr, oh, oherr, _rad_, _oh_ = calczgrad(gas_pos, gas_mass, gas_rho, GFM_Metal, rmax, res,
                                                O_index = 7, H_index = 2, EAGLE_rho=True, rhocutidx=sf_idx)
    
    rsmall, rbig = riprime, ro
    fit_mask     = ( r > rsmall ) & ( r < rbig ) & ~np.isnan(oh)
    
    print(rsmall,rbig)
    print(sum(fit_mask))
    # try:
    gradient_SF, intercept_SF = np.polyfit( r[fit_mask], oh[fit_mask], 1 )
    # except:
    #     gradient_SF = np.nan
    
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

    plt.text( 0.7, 0.9 , r'$z=%s$' %snap2zEAGLE[snap], transform=plt.gca().transAxes )
    plt.text( 0.7, 0.8 , r'$\log M_* = %s$' %round( this_Mstar,2 ),transform=plt.gca().transAxes )

    plt.tight_layout()
    plt.savefig( './diagnostic_figs/' + str(galaxy) + '_EAGLE_profile.pdf', bbox_inches='tight' )
    
def save_data(snap, EAGLE, sim_name, file_ext='z000p000', m_star_min = 9.0, m_star_max=11.0, m_gas_min=9.0):
    # Note that I am only interested in central galaxies here... can be modified to include satellite
    
    save_dir = '%s_SF_galaxies/' %sim_name + 'snap_%s' %str(snap).zfill(3) + '/' 
    
    # Get SF galaxies at this snapshot
    SF_galaxies = get_SF_galaxies(snap, simulation_run=sim_name, m_star_min=m_star_min,
                                  m_star_max=m_star_max, m_gas_min=m_gas_min,verbose=False)
    
    print('Number of star forming galaxies at snap %s: %s' %(snap,len(SF_galaxies['Grnr'])) )
    
    ##### Save the group catalog info
    # np.save(save_dir + 'grp_cat' + '.npy', SF_galaxies)
        
    subset = SF_galaxies['Grnr'][:2]
        
    # get_which_files(EAGLE, subset, snap, 
    #                 save_loc='./%s_SF_galaxies/' %sim_name + 'snap_%s' %str(snap).zfill(3) + '/' + 'file_lookup.npy',
    #                 file_ext=file_ext)
    
    # Loop over all galaxies
    ##########################################
    ## To do: all SF galaxies
    ##########################################

    DIR = './%s_SF_galaxies/snap_' %run + str(snap).zfill(3) + '/'

    group_cat = np.load( DIR + 'grp_cat.npy', allow_pickle=True ).item()

    for galaxy in subset:
        print('Getting data for central of FoF: %s' %galaxy)
        
        # ####### Save the data ####### #
        # gas_data  = read_eagle_subhalo(EAGLE, galaxy, snap, PartType=0, file_ext=file_ext,
        #                                keys=['Coordinates','Mass','Metallicity',
        #                                      'Density','StarFormationRate','Velocity'])
        
        reduce_eagle(snap, galaxy, sim_name, group_cat, file_ext=file_ext, EAGLE=EAGLE)
            
if __name__ == "__main__":
    snap_to_file_name = {
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
        
    run  = 'RefL0100N1504'
    
    for snap in [28]:#snap_to_file_name.keys():
        
        if (snap > 11):
            DIR = '/orange/paul.torrey/EAGLE/'
        else:
            DIR = '/orange/paul.torrey/alexgarcia/EAGLE/'
        
        EAGLE = DIR + run  + '/' + 'snapshot_%s_%s' %(str(snap).zfill(3),snap_to_file_name[snap]) + '/' 
    
        save_data( snap, EAGLE, run, file_ext=snap_to_file_name[snap], 
                   m_star_min=9.0, m_star_max=11.0, m_gas_min=9.0 )
