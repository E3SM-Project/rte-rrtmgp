#! /usr/bin/env python
#
# This script compares RFMIP results from RTE+RRTMGP against a benchmark
#
import os
import numpy as np
import xarray as xr
import argparse
import urllib.request

#
# Comparing reference and test results
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compares all-sky example output to file in reference directory")
    parser.add_argument("--allsky_file", type=str, default="rrtmgp-allsky.nc", dest="file",
                        help="Name of file inputs and outputs for all-sky problem (same for test and reference)")
    parser.add_argument("--ref_dir", type=str, default="ref",
                        help="Directory where reference values are")
    parser.add_argument("--download_reference", action="store_true",
                        help="Download reference files even if they exist")
    parser.add_argument("--threshold", type=float, default=0.,
                        help="Threshold for reporting differences")
    args = parser.parse_args()

    tst_file = args.file
    ref_file = os.path.join(args.ref_dir, tst_file)
    # If a version of the file exists in the reference directory, no need to download (can be over-ridden)
    if (args.download_reference or not os.path.exists(ref_file)):
        os.makedirs(args.ref_dir, exist_ok=True)
        urllib.request.urlretrieve("https://owncloud.gwdg.de/index.php/s/OjbNzRTlXUk0G5w/download", ref_file)

    tst = xr.open_dataset(tst_file)
    ref = xr.open_dataset(ref_file)

    for v in ['lw_flux_up', 'lw_flux_dn', 'sw_flux_up', 'sw_flux_dn', 'sw_flux_dir']:

        if np.all(np.isnan(tst.variables[v].values)):
            raise Exception("All test values are missing. Were the tests run?")
        if np.any(np.isnan(tst.variables[v].values)):
            raise Exception("Some test values are missing. Now that is strange.")

        diff = abs((tst-ref).variables[v].values)
        avg  = 0.5*(tst+ref).variables[v].values
        # Division raises a runtime warning when we divide by zero even if the
        #   values in those locations will be ignored.
        with np.errstate(divide='ignore', invalid='ignore'):
            frac_diff = np.abs(np.where((avg > 2.*np.finfo(float).eps), diff/avg, 0))

        if diff.max() > args.threshold:
            print('Variable %s differs (max abs difference: %e; max percent difference: %e%%)'%(v, diff.max(), 100.0 * frac_diff.max()))
        else:
            print('Variable %s: No diffs'%(v))