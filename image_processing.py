# %% load libraries
from plantcv import plantcv as pcv
import os
from matplotlib import pyplot as plt
import numpy as np
from skimage import filters
import re
# %% set params
pcv.params.debug='plot'
plt.rcParams["figure.figsize"] = (9,9)

# %%
Fo_id = '0000'
Fm_id = '0007'
FsLss_id = '0084'
FmpLss_id = '0085'

# %% functions
def get_filenames(pn, grp):
    fns = os.listdir(os.path.join(pn,grp))
    fn = fns[2]
    for fn in fns:
        f = re.split('[_]', os.path.splitext(os.path.basename(fn))[0])
        if f[-1] == Fo_id:
            Fo_fn = fn
        if f[-1] == Fm_id:
            Fm_fn = fn
        if f[-1] == FsLss_id:
            FsLss_fn = fn
        if f[-1] == FmpLss_id:
            FmpLss_fn = fn

    return(Fo_fn, Fm_fn, FsLss_fn, FmpLss_fn)


def get_fluor(imgfns, pn):
    Fo_fn, Fm_fn, FsLss_fn, FmpLss_fn = imgfns
    Fo = pcv.readimage(os.path.join(pn,Fo_fn))[0][:,:,0]
    Fm = pcv.readimage(os.path.join(pn,Fm_fn))[0][:,:,0]
    FsLss = pcv.readimage(os.path.join(pn,FsLss_fn))[0][:,:,0]
    FmpLss = pcv.readimage(os.path.join(pn,FmpLss_fn))[0][:,:,0]

    return(Fo, Fm, FsLss, FmpLss)


# %%
basepath = os.path.join("data","020320 images")
grps = ['GroupA','GroupB','GroupC','GroupD','GroupE','GroupF']
outdir = 'output'
os.makedirs(outdir, exist_ok=True)
pcv.params.debug=None

for i,g in enumerate(grps):
    print(i,g)

    # read fluor data
    imgfns = get_filenames(basepath,g)
    Fo, Fm, FsLss, FmpLss = get_fluor(imgfns, os.path.join(basepath,g))

    # make mask
    # filters.try_all_threshold(Fm_gray)
    ty = filters.threshold_otsu(Fm)
    print(ty)
    mask = pcv.threshold.binary(Fm, ty, 255, 'light')
    mask = pcv.fill(mask, 100)
    _,rect,_,_ = pcv.rectangle_mask(mask,(300,180),(750,600),'white')
    mask = pcv.logical_and(mask,rect)

    # compute apply mask and compute paramters
    out_flt = np.zeros_like(Fm, dtype='float')
    
    # fv/fm
    Fv = np.subtract(Fm,Fo, out=out_flt.copy(), where=mask>0)
    FvFm = np.divide(Fv,Fm, out=out_flt.copy(), where=np.logical_and(mask>0, Fo>0))
    fvfm_fig = pcv.visualize.pseudocolor(FvFm,
                            mask=mask,
                            cmap='viridis',
                            max_value=1)
    outfn = g+'_fvfm.png'
    fvfm_fig.set_size_inches(6, 6, forward=False)
    fvfm_fig.savefig(os.path.join(outdir,outfn),
                    bbox_inches='tight',
                    dpi=150)
    fvfm_fig.clf()

    #npq
    outfn = g+'_ssnpq.png'
    npq = np.divide(Fm,FmpLss, out=out_flt.copy(), where=np.logical_and(mask>0,FmpLss>0))-1
    npq_fig = pcv.visualize.pseudocolor(npq,
                            mask=mask,
                            cmap='inferno',
                            max_value=2.5)
    npq_fig.set_size_inches(6, 6, forward=False)
    npq_fig.savefig(os.path.join(outdir, outfn),
                    bbox_inches='tight',
                    dpi=150)
    npq_fig.clf()

    #fraction open centers
    outfn = g + '_ssqP.png'
    num = np.subtract(FmpLss, FsLss, out=out_flt.copy(), where=mask>0)
    den = np.subtract(FmpLss, Fo, out=out_flt.copy(), where=mask>0)
    qP = np.divide(num, den, out=out_flt.copy(), where=np.logical_and(mask>0, den>0))
    qP_fig = pcv.visualize.pseudocolor(qP,
                                        mask=mask,
                                        cmap='BuPu_r',
                                        min_value=0,
                                        max_value=1)
    qP_fig.set_size_inches(6, 6, forward=False)
    qP_fig.savefig(os.path.join(outdir, outfn), bbox_inches='tight', dpi=150)
    qP_fig.clf()

    # electron transport rate
    # psi2 * light intensity * 0.84 * 0.5  (light intensity = 310 mmol/m^2/s)





#%%
# freq, bins = np.histogram(Fm[np.where(mask>0)], 30)
# plt.plot(bins[:-1], freq)
