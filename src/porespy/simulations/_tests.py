import numpy as np
import matplotlib.pyplot as plt
import porespy as ps

edt = ps.tools.get_edt()

im = ps.generators.blobs([200, 200], porosity=0.75, seed=0)
dt = edt(im).astype(int)

# This file compares the 4 different types of "logic" of the drainage and imbibition
# algorithms. The point is to make sure that each method produces EXACTLY the same
# output for a given size and 'smoothness'. This was especially useful when
# developing logic for finding the 'edges' for the brute-force method, and for
# clarifying how to find the seeds for the dilation if smooth=True or not. This is
# not run as an actual test (i.e. during the CI builds), but is here as a reference.

# %% Look at drainage logic
Rs = [13, 12]
smooth = True

# bf
nwp_bf = np.zeros_like(im)
seeds_prev = dt >= Rs[0]
seeds_bf = dt >= Rs[1]
edges = seeds_bf * ~seeds_prev
crds = np.vstack(np.where(edges))
nwp_bf = ps.tools._insert_disk_at_points(
    im=nwp_bf,
    coords=crds,
    r=Rs[1],
    v=1,
    smooth=smooth,
)
nwp_bf[seeds_bf] = True

# dt
seeds_dt = dt >= Rs[1]
nwp_dt = edt(~seeds_dt) < Rs[1] if smooth else edt(~seeds_dt) <= Rs[1]

# dt_fft
seeds_dt_fft = dt >= Rs[1]
se = ps.tools.ps_round(Rs[1], ndim=im.ndim, smooth=smooth)
nwp_dt_fft = ps.filters.fftmorphology(seeds_dt_fft, strel=se, mode='dilation')

# fft
se = ps.tools.ps_round(Rs[1], ndim=im.ndim, smooth=True)
seeds_fft = ~ps.filters.fftmorphology(~im, strel=se, mode='dilation')
se = ps.tools.ps_round(Rs[1], ndim=im.ndim, smooth=smooth)
nwp_fft = ps.filters.fftmorphology(seeds_fft, strel=se, mode='dilation')

fig, ax = plt.subplots(2, 2)
ax[0][0].imshow(nwp_bf/~edges/im)
ax[0][1].imshow(nwp_dt/~seeds_dt/im)
ax[1][0].imshow(nwp_dt_fft/~seeds_dt_fft/im)
ax[1][1].imshow(nwp_fft/~seeds_fft/im)

assert np.all(nwp_bf == nwp_dt)
assert np.all(nwp_bf == nwp_dt_fft)
assert np.all(nwp_bf == nwp_fft)

assert np.all(seeds_bf == seeds_dt)
assert np.all(seeds_bf == seeds_dt_fft)
assert np.all(seeds_bf == seeds_fft)

# %% Look at imbibition logic
Rs = Rs[-1::-1]

# bf
nwp_bf = np.zeros_like(im)
seeds_prev = (dt <= Rs[0]) * im
seeds_bf = (dt <= Rs[1]) * im
edges = seeds_bf * (~seeds_prev) * im
crds = np.vstack(np.where(edges))
nwp_bf = ps.tools._insert_disk_at_points(
    im=nwp_bf,
    coords=crds,
    r=Rs[1],
    v=1,
    smooth=smooth,
)
nwp_bf[(~seeds_bf)*im] = True
wp_bf = (~nwp_bf)*im

# dt
seeds_dt = dt >= Rs[1]
nwp_dt = edt(~seeds_dt) < Rs[1] if smooth else edt(~seeds_dt) <= Rs[1]
wp_dt = ~nwp_dt*im

# dt_fft
seeds_dt_fft = dt >= Rs[1]
se = ps.tools.ps_round(Rs[1], ndim=im.ndim, smooth=smooth)
nwp_dt_fft = ps.filters.fftmorphology(seeds_dt_fft, strel=se, mode='dilation')
wp_dt_fft = ~nwp_dt_fft*im

# fft
se = ps.tools.ps_round(Rs[1], ndim=im.ndim, smooth=~smooth)
seeds_fft = ~ps.filters.fftmorphology(~im, strel=se, mode='dilation')
se = ps.tools.ps_round(Rs[1], ndim=im.ndim, smooth=smooth)
nwp_fft = ps.filters.fftmorphology(seeds_fft, strel=se, mode='dilation')
wp_fft = ~nwp_fft*im

fig, ax = plt.subplots(2, 2)
ax[0][0].imshow(wp_bf/im)
ax[0][1].imshow(wp_dt/im)
ax[1][0].imshow(wp_dt_fft/im)
ax[1][1].imshow(wp_fft/im)

assert np.all(wp_bf == wp_dt)
assert np.all(wp_bf == wp_dt_fft)
assert np.all(wp_bf == wp_fft)

assert np.all(seeds_dt == (~seeds_bf + edges)*im)
assert np.all(seeds_dt == seeds_dt_fft)
assert np.all(seeds_dt == seeds_fft)
