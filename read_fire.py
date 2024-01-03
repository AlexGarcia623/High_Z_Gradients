import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import h5py

from sfms import sfmscut, center, calc_rsfr_io, calc_incl, trans, calczgrad, calcrsfr, grad_valid
from matplotlib.colors import LogNorm
from tqdm import tqdm

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


def open_data(location, snap, ptType):
    
    with h5py.File( location + 'snapshot_%s' %str(snap).zfill(3) + '.hdf5' ,'r' ) as f:
        
        h   = f['Header'].attrs.get('HubbleParam')
        scf = f['Header'].attrs.get('Time')
        z   = f['Header'].attrs.get('Redshift')
                
        gas_data = f['PartType'+str(ptType)]
        
        print(gas_data['Metallicity'])


if __name__ == "__main__":
    
    DIR = '/orange/paul.torrey/FIRE/high_redshift/'
    
    sim = 'z5m09a'
    
    loc = DIR + sim + '/output/'
    
    snap = 20
    
    open_data(loc, snap, ptType=0)
    
    # print('Hello World!')