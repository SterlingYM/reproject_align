import astropy.units as u
from astropy.coordinates import SkyCoord

# ----------------------------------------
# target & cutout info
# ----------------------------------------
galaxy_name = 'N2525'
center_coord = SkyCoord.from_name(galaxy_name)
cutout_size = 3.0 * u.arcmin
filters = ['F555W','F814W','F090W','F150W','F160W','F277W']

# ----------------------------------------
# reprojection settings
# ----------------------------------------
target_resolution = 0.03 * u.arcsec
target_pixscale = target_resolution / u.pixel
filter_ref = 'F090W'

# ----------------------------------------
# source finder settings
# ----------------------------------------
Ncutouts = 100
Nsources_per_cutout = 5

# ----------------------------------------
# original files
# ----------------------------------------
ext_drive = '/Volumes/S-Express/'

original_paths = {
    'F555W': ext_drive+'HST/n2525v.fits',
    'F814W': ext_drive+'HST/n2525i.fits',
    'F160W': ext_drive+'HST/n2525h.fits',
    'F090W': ext_drive+'JWST/N2525/jw02875-o001_t001_nircam_clear-f090w_i2d.fits',
    'F150W': ext_drive+'JWST/N2525/jw02875-o001_t001_nircam_clear-f150w_i2d.fits',
    'F277W': ext_drive+'JWST/N2525/jw02875-o001_t001_nircam_clear-f277w_i2d.fits',
    }

fwhm_dict = {
    # https://hst-docs.stsci.edu/wfc3ihb/chapter-6-uvis-imaging-with-wfc3/6-6-uvis-optical-performance
    # https://hst-docs.stsci.edu/wfc3ihb/chapter-7-ir-imaging-with-wfc3/7-6-ir-optical-performance
    # https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-point-spread-functions
    'F555W': 0.067 * u.arcsec / target_pixscale,
    'F814W': 0.074 * u.arcsec / target_pixscale,
    'F090W': 0.033 * u.arcsec / target_pixscale,
    'F150W': 0.050 * u.arcsec / target_pixscale,
    'F160W': 0.151 * u.arcsec / target_pixscale,
    'F277W': 0.092 * u.arcsec / target_pixscale,
    }

# ----------------------------------------
# output settings
# ----------------------------------------
output_folder = ext_drive + f'SH0ES_reprojected/{galaxy_name}/'
