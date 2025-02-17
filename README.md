# contbin_python

This is a python re-implementation of the Jeremy Sanders contour binning algorithm, which was originally in c++. You can install it with `pip install pycontbin`, or upgrade it with `pip install --upgrade pycontbin`. Please note that this python build is under development and is subject to updates.

You can see the original C++ files here: https://github.com/jeremysanders/contbin

Original Copyright Jeremy Sanders <jeremy@jeremysanders.net> (2002-2016)
The reference paper is Sanders (2006), MNRAS, 371, 829,
http://adsabs.harvard.edu/abs/2006MNRAS.371..829S

This version was translated to python by me, and might still contain some minor bugs, although the current implementation results seem to reproduce the original code results.

I have additionally added a routine that automatically produces and outputs polygon region files for each bin in sexagesimal coordinates, this is an improvement on the previous work that output many box regions instead.

Update 17/Feb/2025 - I have added a new routine that takes a 'psf_arcsec' parameter as input. Using the WCS header information, it calculates how many pixels corresponds to this psf, then uses it as a constraint to increase the size of bins that fall underneath the psf size. For bins which are smaller than the psf, the routine goes pixel by pixel and reassigns the pixels to neighboring bins that have a similar smoothed flux. If psf_arcsec is equal to or less than 0, the routine skips this additional psf constraint on the bin size and returns the default map.

The jupyter notebook contains the testing code, also an example use case.

This software is licensed under the GNU Public License.

To do:
- Instead of a static PSF, it would be nice to feed a psf map and have more adaptive psf constraints for each bin.
- Adding GPU optimization would be wonderful.

Done:
- An static PSF constraint has been added to the code. Regions with height and width smaller than the given psf will be merged with neighboring bins until all bins are larger than the psf.
- Added polygon regions as output in sexagesimal coordinates, however this requires ciao and pyds9 installed.
- Created pip package

## Requirements 
- numpy
- astropy
- ciao
- pyds9

## Code use case example:
```
# from contour_binning import *
from pycontbin import ContourBin

os.chdir("/Users/jpbreuer/Scripts/contbin-python")
inputfile = "scaled_fluxed_sps_filth_fov.fits"

sn_ratio = 50
smooth = 30
constrain_val = 1.5
reg_bin = 1
psf_arcsec = 10

ContourBin(inputfile, sn_ratio, smooth, constrain_val, reg_bin, make_region_files=True, psf=psf_arcsec)
```
