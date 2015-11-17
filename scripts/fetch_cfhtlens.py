#!/usr/bin/env python

import os
import shutil
import requests
import re
import subprocess
from time import sleep

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy import wcs


def used_y_band(field):

    y_fields = [
        'W1m0m4', 'W1m1m4', 'W1m2m4', 'W1m3m4', 'W1m4m4',
        'W1p1m4', 'W1p1p1', 'W1p2m4', 'W1p3m4', 'W1p3p1',
        'W1p4m4', 'W3m0m1', 'W3m2m1', 'W3m2p1', 'W3p2m3',
        'W4m1p1', 'W4m1p2', 'W4m1p3', 'W4m2p2', 'W4m2p3',
        'W4m3p3'
    ]

    if field in y_fields:
        return True
    else:
        return False

def fetch_fits(df, dirname="temp"):

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for field in df["field"].unique():

        if used_y_band(field):
            bands = [c for c in 'ugryz']
        else:
            bands = [c for c in 'ugriz']

        url = "http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/vospace/CFHTLens/images/"

        for band in bands:

            filename = "{}_{}.V2.2A.swarp.cut.fits".format(field, band)
            filepath = os.path.join(dirname, filename)
            
            if os.path.exists(filepath):
                print("Skipped {}".format(filename))
                continue

            for _ in range(10):
                try:
                    resp = requests.get(url + filename, stream=True)
                except:
                    sleep(1)
                    continue
                
                if resp.status_code == 200:
                    with open(filepath, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=1024):
                            if chunk:
                                f.write(chunk)
                    print("Downloaded {}".format(filename))
                    break
                else:
                    sleep(1)
                    continue

            if not os.path.exists(filepath):
                raise Exception

def convert_catalog_to_pixels(df, dirname="temp"):

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    pixels = []
    fits_list = []

    for idx, row in df.iterrows():

        fits_file = "{}_r.V2.2A.swarp.cut.fits".format(row["field"])
        fits_path = os.path.join(dirname, fits_file)
            
        hdulist = fits.open(fits_path)

        w = wcs.WCS(hdulist[0].header, relax=False)
        
        px, py = w.all_world2pix(row["ALPHA_J2000"], row["DELTA_J2000"], 1)

        fits_list.append(fits_file)
        pixels.append((idx, px, py, row["true_class"]))

    for i, fits_file in enumerate(fits_list):
        ix, px, py, c = pixels[i]
        pixel_list = fits_file.replace(".fits", ".list")
        pixel_path = os.path.join(dirname, pixel_list)
        with open(pixel_path, "a") as fout:
            fout.write("{} {} {} {}\n".format(ix, px, py, c))

    return None

def write_default_conv():

    default_conv = (
        "CONV NORM\n"
        "# 3x3 ``all-ground'' convolution mask with FWHM = 2 pixels.\n"
        "1 2 1\n"
        "2 4 2\n"
        "1 2 1\n"
    ).format()

    with open("default.conv", "w") as f:
        f.write(default_conv)

    return None

def write_default_param():

    default_param = (
        "XMIN_IMAGE               Minimum x-coordinate among detected pixels                [pixel]\n"
        "YMIN_IMAGE               Minimum y-coordinate among detected pixels                [pixel]\n"
        "XMAX_IMAGE               Maximum x-coordinate among detected pixels                [pixel]\n"
        "YMAX_IMAGE               Maximum y-coordinate among detected pixels                [pixel]\n"
        "VECTOR_ASSOC(1)          #ASSOCiated parameter vector"
    ).format()

    with open("default.param", "w") as f:
        f.write(default_param)

    return None

def write_default_sex():

    default_sex = (
        "#-------------------------------- Catalog ------------------------------------\n"
        "\n"
        "CATALOG_NAME     test.cat       # name of the output catalog\n"
        "CATALOG_TYPE     ASCII_HEAD     # NONE,ASCII,ASCII_HEAD, ASCII_SKYCAT,\n"
        "                                # ASCII_VOTABLE, FITS_1.0 or FITS_LDAC\n"
        "PARAMETERS_NAME  default.param  # name of the file containing catalog contents\n"
        " \n"
        "#------------------------------- Extraction ----------------------------------\n"
        " \n"
        "DETECT_TYPE      CCD            # CCD (linear) or PHOTO (with gamma correction)\n"
        "DETECT_MINAREA   3              # min. # of pixels above threshold\n"
        "DETECT_THRESH    1.5            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n"
        "ANALYSIS_THRESH  1.5            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2\n"
        " \n"
        "FILTER           Y              # apply filter for detection (Y or N)?\n"
        "FILTER_NAME      default.conv   # name of the file containing the filter\n"
        " \n"
        "DEBLEND_NTHRESH  32             # Number of deblending sub-thresholds\n"
        "DEBLEND_MINCONT  0.005          # Minimum contrast parameter for deblending\n"
        " \n"
        "CLEAN            Y              # Clean spurious detections? (Y or N)?\n"
        "CLEAN_PARAM      1.0            # Cleaning efficiency\n"
        " \n"
        "MASK_TYPE        CORRECT        # type of detection MASKing: can be one of\n"
        "                                # NONE, BLANK or CORRECT\n"
        "\n"
        "#------------------------------ Photometry -----------------------------------\n"
        " \n"
        "PHOT_APERTURES   5              # MAG_APER aperture diameter(s) in pixels\n"
        "PHOT_AUTOPARAMS  2.5, 3.5       # MAG_AUTO parameters: <Kron_fact>,<min_radius>\n"
        "PHOT_PETROPARAMS 2.0, 3.5       # MAG_PETRO parameters: <Petrosian_fact>,\n"
        "                                # <min_radius>\n"
        "\n"
        "SATUR_LEVEL      50000.0        # level (in ADUs) at which arises saturation\n"
        "SATUR_KEY        SATURATE       # keyword for saturation level (in ADUs)\n"
        " \n"
        "MAG_ZEROPOINT    0.0            # magnitude zero-point\n"
        "MAG_GAMMA        4.0            # gamma of emulsion (for photographic scans)\n"
        "GAIN             0.0            # detector gain in e-/ADU\n"
        "GAIN_KEY         GAIN           # keyword for detector gain in e-/ADU\n"
        "PIXEL_SCALE      1.0            # size of pixel in arcsec (0=use FITS WCS info)\n"
        " \n"
        "#------------------------- Star/Galaxy Separation ----------------------------\n"
        " \n"
        "SEEING_FWHM      1.2            # stellar FWHM in arcsec\n"
        "STARNNW_NAME     default.nnw    # Neural-Network_Weight table filename\n"
        " \n"
        "#------------------------------ Background -----------------------------------\n"
        " \n"
        "BACK_SIZE        64             # Background mesh: <size> or <width>,<height>\n"
        "BACK_FILTERSIZE  3              # Background filter: <size> or <width>,<height>\n"
        " \n"
        "BACKPHOTO_TYPE   GLOBAL         # can be GLOBAL or LOCAL\n"
        " \n"
        "#------------------------------ Check Image ----------------------------------\n"
        " \n"
        "CHECKIMAGE_TYPE  SEGMENTATION   # can be NONE, BACKGROUND, BACKGROUND_RMS,\n"
        "                                # MINIBACKGROUND, MINIBACK_RMS, -BACKGROUND,\n"
        "                                # FILTERED, OBJECTS, -OBJECTS, SEGMENTATION,\n"
        "                                # or APERTURES\n"
        "CHECKIMAGE_NAME  check.fits     # Filename for the check-image\n"
        " \n"
        "#--------------------- Memory (change with caution!) -------------------------\n"
        " \n"
        "MEMORY_OBJSTACK  60000           # number of objects in stack\n"
        "MEMORY_PIXSTACK  3000000         # number of pixels in stack\n"
        "MEMORY_BUFSIZE   4096           # number of lines in buffer\n"
        " \n"
        "#----------------------------- Miscellaneous ---------------------------------\n"
        " \n"
        "VERBOSE_TYPE     QUIET          # can be QUIET, NORMAL or FULL\n"
        "HEADER_SUFFIX    .head          # Filename extension for additional headers\n"
        "WRITE_XML        N              # Write XML file (Y/N)?\n"
        "XML_NAME         sex.xml        # Filename for XML output\n"
        "\n"
        "#----------------------------- ASSOC parameters ---------------------------------\n"
        "\n"
        "ASSOC_NAME       sky.list       # name of the ASCII file to ASSOCiate, the expected pixel \n"
        "                                # coordinates list given as [id, xpos, ypos]\n"
        "ASSOC_DATA       1              # columns of the data to replicate (0=all), replicate id\n"
        "                                # of the object in the SExtractor output file\n"
        "ASSOC_PARAMS     2,3            # columns of xpos,ypos[,mag] in the expected pixel\n"
        "                                # coordinates list\n"
        "ASSOC_RADIUS     2.0            # cross-matching radius (pixels)\n"
        "ASSOC_TYPE       NEAREST        # ASSOCiation method: FIRST, NEAREST, MEAN,\n"
        "                                # MAG_MEAN, SUM, MAG_SUM, MIN or MAX\n"
        "ASSOCSELEC_TYPE  MATCHED        # ASSOC selection type: ALL, MATCHED or -MATCHED\n"
    ).format()

    with open("default.sex", "w") as f:
        f.write(default_sex)

def run_sex(df, dirname="temp"):
    """
    """

    print("Running sextractor...")

    cat = pd.DataFrame()

    registered_all = ["{}_r.V2.2A.swarp.cut.fits".format(f) for f in df["field"].unique()]
        
    for f in registered_all:
        
        fpath = os.path.join(dirname, f)
        
        list_file = f.replace(".fits", ".list")
        list_path = os.path.join(dirname, list_file)

        config_file = f.replace(".fits", ".sex")

        with open("default.sex", "r") as default:
            with open(config_file, "w") as temp:
                for line in default:
                    line = re.sub(
                        r"^ASSOC_NAME\s+sky.list",
                        "ASSOC_NAME       {}".format(list_file),
                        line
                    )
                    temp.write(line)
    
        shutil.copy(list_path, os.getcwd())
    
        subprocess.call(["sex", "-c", config_file, fpath])

        os.remove(config_file)
    
        try:
            assoc = pd.read_csv(
                "test.cat",
                skiprows=5,
                sep="\s+",
                names=["xmin", "ymin", "xmax", "ymax", "match"]
            )
            assoc["file"] = f
            cat = cat.append(assoc)
        except:
            pass
        
        os.remove(os.path.join(os.getcwd(), list_file))
    
    if len(cat) > 0:
        cat["true_class"] = df.ix[cat["match"], "true_class"].values
        cat["id"] = df.ix[cat["match"], "id"].values
        cat["field"] = df.ix[cat["match"], "field"].values

    #cat = cat.reset_index(drop=True)

    return cat


def flux_to_magnitude(flux, field):
    hdulist = fits.open("temp/{}_r.V2.2A.swarp.cut.fits".format(field))
    mag_zp = hdulist[0].header["MAGZP"]
    hdulist.close()
    mag = mag_zp - 2.5 * np.log10(flux)
    return mag

def save_cutout(df, size=96, image_dir="temp", save_dir="result"):

    print("Saving cutout images...")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def find_position(xmin, xmax, cut_size, frame_size):
        diff = 0.5 * ((xmax - xmin) - cut_size)
        if xmin + diff < 0:
            r = 0
            l = r + cut_size
        elif xmax + diff >= frame_size:
            l = frame_size
            r = l - cut_size
        else:
            r = int(xmin + diff)
            l = r + cut_size
        return r, l

    for i, row in df.iterrows():
        array = np.zeros((5, size, size))
   
        if used_y_band(row["field"]):
            bands = [c for c in 'ugryz']
        else:
            bands = [c for c in 'ugriz']

        for j, b in enumerate(bands):

            fpath = os.path.join(image_dir, row["file"])
            image_data = fits.getdata(fpath.replace("_r.", "_{}.".format(b)))
            
            y0, x0, y1, x1 = row[["xmin", "ymin", "xmax", "ymax"]].values

            right, left = find_position(x0, x1, size, image_data.shape[0])
            down, up = find_position(y0, y1, size, image_data.shape[1])

            cut_out = image_data[right: left, down: up]
            
            cut_out[cut_out <= 0] = 1.0e-20

            if cut_out.shape[0] == size and cut_out.shape[1] == size:
                array[j, :, :] = flux_to_magnitude(cut_out, row["field"])

        array[array > 30] = 30

        if np.isnan(array).sum() == 0 and array.sum() > 0:
            save_path = os.path.join(save_dir, "{0}.{1}x{1}.{2}.npy".format(row["true_class"], size, row["id"]))
            np.save(save_path, array)

def run_online_mode(filename):

    df = pd.read_csv(filename)

    if os.path.exists("result"):
        done = os.listdir("result")
        done = [d.split(".")[2].split("_")[0] for d in done]
        # check existing results and skip
        df = df[~df["field"].isin(done)]

    write_default_conv()
    write_default_param()
    write_default_sex()

    for field in df["field"].unique():
        print("field: {}".format(field))
        # download image fits files
        dff = df[df["field"] == field]
        fetch_fits(dff)
        convert_catalog_to_pixels(dff)
        cat = run_sex(dff)
        save_cutout(cat, size=96)
        print("Done processing {} field. Cleaning up...".format(field))
        shutil.rmtree("temp")

if __name__ == "__main__":

    run_online_mode("cfhtlens_matched.csv")
