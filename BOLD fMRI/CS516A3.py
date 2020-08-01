import nibabel as nib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from sklearn.metrics import mutual_info_score
import scipy.stats as stats
import mne
from autoreject import get_rejection_threshold

path = 'a3_data/'

"""
Hemodynamic reponse function
"""
hrf = pd.read_csv('a3_data/hrf.csv', header=None).values.ravel()
plt.plot(hrf); plt.xlabel('time(seconds)');

for i in range(1, 17):
    clean_bold = nib.load(path+str(i)+'/clean_bold.nii.gz')
    events = pd.read_csv(path+str(i)+'/events.tsv', delimiter='\t')
    tr = clean_bold.header.get_zooms()[3]
    
    """
    Ideal time series
    """
    ts = np.zeros(int(tr * clean_bold.shape[3]))
    
    ts[np.round(events[~events['stim_type'].isna()]['onset'].values).astype('uint16')] = 1
    
    #plt.plot(ts); plt.xlabel('time(seconds)');
    
    """
    Convolved time series
    """
    conved = signal.convolve(ts, hrf, mode='full')[:ts.shape[0]]
    
    #plt.plot(ts); plt.plot(conved*3); plt.xlabel('time(seconds)');
    
    conved = conved[::int(tr)]
    img = clean_bold.get_fdata()
    
    """
    Correlate
    """
    meansub_img = img - np.expand_dims(img.mean(-1), 3)
    meansub_conved = conved - conved.mean()
        
    corrs = (meansub_img * meansub_conved).sum(-1) / \
        np.sqrt((meansub_img * meansub_img).sum(-1)) / np.sqrt(np.dot(meansub_conved, meansub_conved))
        
    corrs[np.isnan(corrs)] = 0
    
    plt.imshow(np.rot90(corrs.max(2))); plt.colorbar();
    plt.imshow(np.rot90(corrs.max(2)), vmin=-0.25, vmax=0.25); plt.colorbar();
    
    corrs_nifti = nib.Nifti1Image(corrs, clean_bold.affine)
    nib.save(corrs_nifti, path+'corrs.nii.gz')
    
    

    
    """
    Mutual information
    """
    def MI(a):
        c_xy = np.histogram2d(a, conved)[0]
        return mutual_info_score(None, None, contingency=c_xy)
    
    mi = np.apply_along_axis(MI, 3, img)
        
    plt.imshow(np.rot90(mi.max(2))); plt.colorbar();
    plt.imshow(np.rot90(mi.max(2)), vmin=-0.25, vmax=0.25); plt.colorbar();

raw = nib.load(path+'1/bold.nii.gz').get_fdata()
    
mi_nifti = nib.Nifti1Image(mi, clean_bold.affine)
nib.save(mi_nifti, path+str(i)+'/mi.nii.gz')

meansub_raw = raw - np.expand_dims(raw.mean(-1), 3)

corrs_raw = (meansub_raw * meansub_conved).sum(-1) / \
    np.sqrt((meansub_raw * meansub_raw).sum(-1)) / np.sqrt(np.dot(meansub_conved, meansub_conved))
    
corrs_raw[np.isnan(corrs_raw)] = 0
plt.imshow(np.rot90(corrs_raw.max(2))); plt.colorbar();
plt.imshow(np.rot90(corrs_raw.max(2)), vmin=-0.25, vmax=0.25); plt.colorbar();
    
mi_raw = np.apply_along_axis(MI, 3, raw)

plt.imshow(np.rot90(mi_raw.max(2))); plt.colorbar();
plt.imshow(np.rot90(mi_raw.max(2)), vmin=-0.25, vmax=0.25); plt.colorbar();


"""
Group average
"""

corrs_nifti = nib.load(path+'1/corrs_in_template.nii.gz')
corrs = corrs_nifti.get_fdata()
for i in range(2, 17):
    if i == 3 or i == 10:
        continue
    corrs += nib.load(path+str(i)+'/corrs_in_template.nii.gz').get_fdata()
        
corrs /= 14
corrs_nifti = nib.Nifti1Image(corrs, corrs_nifti.affine)
nib.save(corrs_nifti, path+'corrs_in_template.nii.gz')

mi_nifti = nib.load(path+'1/mi_in_template.nii.gz')
mi = mi_nifti.get_fdata()
for i in range(2, 17):
    if i == 3 or i == 10:
        continue
    mi += nib.load(path+str(i)+'/mi_in_template.nii.gz').get_fdata()
        
mi /= 14
mi_nifti = nib.Nifti1Image(mi, mi_nifti.affine)
nib.save(mi_nifti, path+'mi_in_template.nii.gz')

"""
Famous and unfamiliar
"""

for i in range(1, 17):
    if i == 3 or i == 10:
        continue
    clean_bold = nib.load(path+str(i)+'/clean_bold.nii.gz')
    img = clean_bold.get_fdata()
    tr = clean_bold.header.get_zooms()[-1]
    events = pd.read_csv(path+str(i)+'/events.tsv', delimiter='\t')
    f1id = np.round((events['onset'][events['stim_type'] != 'FAMOUS'].values+4.5)/tr).astype('uint16')
    f2id = np.round((events['onset'][events['stim_type'] != 'UNFAMILIAR'].values+4.5)/tr).astype('uint16')
    f1 = img[:, :, :, f1id]
    f2 = img[:, :, :, f2id]
    tmap = stats.ttest_ind(f1, f2, axis=3)[0]
    tmap_nifti = nib.Nifti1Image(tmap, clean_bold.affine)
    nib.save(tmap_nifti, path+str(i)+'/tmap.nii.gz')
  
        
tmap_nifti = nib.load(path+'1/tmap_in_template.nii.gz')
tmap = tmap_nifti.get_fdata()
for i in range(2, 17):
    if i == 3 or i == 10:
        continue
    tmap += nib.load(path+str(i)+'/tmap_in_template.nii.gz').get_fdata()
        
tmap /= 14
tmap_nifti = nib.Nifti1Image(tmap, tmap_nifti.affine)
nib.save(tmap_nifti, path+'tmap_in_template.nii.gz')

"""
MEG
"""
# download the dataset
mne.datasets.sample.data_path()

# Load data
raw = mne.io.read_raw_fif(path+'meg.fif')
raw.del_proj()

# Power line noise
fig = raw.plot_psd(average=True)
# add some arrows at 50 Hz and its harmonics
for ax in fig.axes:
    freqs = ax.lines[-1].get_xdata()
    psds = ax.lines[-1].get_ydata()
    for freq in range(50, 550, 50):
        idx = np.searchsorted(freqs, freq)
        ax.arrow(x=freqs[idx], y=psds[idx]+18, dx=0, dy=-12, color='red',
                 width=0.1, head_width=3, length_includes_head=True)

picks = mne.pick_types(raw.info, eeg=True)
freqs = np.arange(50, 550, 50)
raw.load_data().notch_filter(freqs=freqs, picks=picks)

raw.plot_psd(average=True)


# Filtering to remove slow drifts
eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True)
raw.plot(order=eeg_channels, n_channels=len(eeg_channels))

regexp = r'EEG06.'
artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)
raw.plot(order=artifact_picks, n_channels=len(artifact_picks))


# set up and fit the ICA
raw_filt = raw.copy().load_data().filter(l_freq=1.0, h_freq=250)
ica = mne.preprocessing.ICA(n_components=40)
ica.fit(raw_filt)

ica.plot_components()

# bad components
ica.exclude = [0, 4, 5, 9, 39]

# find which ICs match the ECG pattern
ecg_indices, ecg_scores = ica.find_bads_ecg(raw, method='correlation')
ica.exculde += ecg_indices

# barplot of ICA component ECG match scores
ica.plot_scores(ecg_scores)

raw_reconst = raw.copy().load_data().filter(l_freq=0.2, h_freq=250)
ica.apply(raw_reconst.load_data())

# extract epochs and save them
picks = mne.pick_types(raw_reconst.info, meg=True, eeg=True)
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, picks=picks, preload=True)
# automated reject approach
reject = get_rejection_threshold(epochs)

epochs.drop_bad(reject=reject)
epochs.info['bads'] = ['EEG063']


# compute evoked response and plot evoked
evoked = epochs.average()
evoked.plot_joint()

# coregistration
mne.gui.coregistration()
