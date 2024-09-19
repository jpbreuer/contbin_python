# contbin_python

This is a python re-implementation of the Jeremy Sanders contour binning algorithm, which was originally in c++.
You can see the original files here: https://github.com/jeremysanders/contbin

Original Copyright Jeremy Sanders <jeremy@jeremysanders.net> (2002-2016)
The reference paper is Sanders (2006), MNRAS, 371, 829,
http://adsabs.harvard.edu/abs/2006MNRAS.371..829S

This version was translated to python by me, and might still contain some minor bugs, although the current implementation results seem to reproduce the original code results.

The jupyter notebook contains the testing code, also an example use case.

This software is licensed under the GNU Public License.

To do: I want to implement PSF information in the code. I will be trying to work on this on another branch.


Code Use case example:

from contour_binning import *

os.chdir("/Users/jpbreuer/Scripts/contbin-python")
inputfile = "scaled_fluxed_sps_filth_fov.fits"

sn_ratio = 50
smooth = 30
constrain_val = 1.5
reg_bin = 1

ContourBin(inputfile, sn_ratio, smooth, constrain_val, reg_bin)
