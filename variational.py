import tensorflow as tf
import tensorflow.keras as keras
import glob
import pandas as pd
import os
import numpy as np
import h5py
import atlas_mpl_style as ampl
from atlasify import atlasify
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
data_template = 'conversion/pt{}_taus/'
avail = glob.glob(data_template.format('*'))

def load_data(pt=200, const=False):
    fmt = data_template.format(pt).split('/')[1]
    ret = (
        pd.read_pickle('data/{}_dijet_hlf'.format(fmt)),
        pd.read_pickle('data/{}_ttbar_hlf'.format(fmt)),
    )
    if const:
        ret = ret + (
            pd.read_pickle('data/{}_dijet_const'.format(fmt)),
            pd.read_pickle('data/{}_ttbar_const'.format(fmt)),
        )
    return ret

hlf, tthlf, const, ttconst = load_data(pt=500, const=1)

hlf['tau21'] = hlf.tau2/(hlf.tau1 + 1e-12)
hlf['tau32'] = hlf.tau3/(hlf.tau2 + 1e-12)

tthlf['tau21'] = tthlf.tau2/(tthlf.tau1 + 1e-12)
tthlf['tau32'] = tthlf.tau3/(tthlf.tau2 + 1e-12)

hlf_to_train = [
    # 'Eta',
    # 'Phi',
    'Charge',
    'EhadOverEem',
    'PT',
    'Mass',
    'NCharged',
    'NNeutrals',
    'tau21',
    'tau32',
]

from sklearn.model_selection import KFold

seed = 42
N_kfolds = 4
shuffle = False

# significances of signal injection
injection_fracs = np.array([0., .005, 0.01, .02])

max_bkg = -1
if max_bkg < 0:
    max_bkg = len(hlf)

N_bkg = min((len(hlf), max_bkg))

N_sigs = np.round(injection_fracs*N_bkg).astype(int)
# N_sigs = np.round(np.sqrt(N_bkg - N_bkg/N_kfolds)*injection_sigs).astype(int)

data_x = [pd.concat([hlf.sample(N_bkg, random_state=seed), tthlf.sample(N_sig, random_state=seed)]).reset_index(drop=True) for N_sig in N_sigs]
data_y = [pd.Series(np.concatenate([np.zeros(N_bkg), np.ones(N_sig)]), index=data_x[i].index) for i,N_sig in enumerate(N_sigs)]
data_ttbar = [tthlf[~tthlf.index.isin(tthlf.sample(N_sig, random_state=seed).index)].reset_index(drop=True) for N_sig in N_sigs]

kf = KFold(N_kfolds, shuffle=True, random_state=seed)
fraction = 1/N_kfolds

train_test_idx = [[(data_x[i].index[sp[0]], data_x[i].index[sp[1]]) for sp in list(kf.split(data_x[i].index))] for i in range(len(data_x))]

import warnings
ampl.use_atlas_style()

warnings.filterwarnings('ignore', 'RuntimeWarning')

def minmax_norm(train, test, features):
    
    f_app = lambda x: (x - mi)/(ma - mi)
    train_c, test_c = train.copy(), test.copy()
    if 'EhadOverEem' in train:
        train_c['EhadOverEem'] = np.log(train_c.EhadOverEem + 1e-3)
        test_c['EhadOverEem'] = np.log(test_c.EhadOverEem + 1e-3)
    mi, ma = train_c[features].min(), train_c[features].max()
    
    tr,te = f_app(train_c[features]), f_app(test_c[features])
    
    return tr, te

def plot_train_test_datasets(train, test, ttbar, save_path=None, show=True, title='No Title', n_bins=50, ncols=3, figh=6, figw=7, frac_size=0.33):
    rows,cols = int(np.ceil(len(train.columns)/ncols)),int(ncols)

    fig, axs = plt.subplots(2*rows, cols, figsize=(cols*figw, rows*figh), gridspec_kw={'height_ratios': [1, frac_size,]*rows})

    axs = axs.T.reshape(cols, rows, -1).reshape(cols*rows,-1)
    
    ranges = pd.concat([pd.concat([train,test,ttbar]).min(), pd.concat([train, test, ttbar]).max()], axis=1)
    for i,f in enumerate(train.columns):
        bins = np.linspace(*ranges.loc[f], n_bins)
        
        axs[i,0].hist(train[f], histtype='step', bins=bins, density=1)
        axs[i,0].hist(test[f], histtype='step', bins=bins, density=1)
        axs[i,0].hist(ttbar[f], histtype='step', bins=bins, density=1)
        
        if f.startswith('M'): 
            axs[i,0].set_yscale('log')
        tr = np.histogram(train[f], bins=bins)[0]
        tr_normc = (np.diff(bins)*tr).sum()
        te = np.histogram(test[f], bins=bins)[0]
        te_normc = (np.diff(bins)*te).sum()
        for e in [-2, 0, 2]:
            axs[i,1].axhline(e, color='tab:grey', ls=':' if abs(e)>0 else '-')
        axs[i,1].axhspan(-1, 1, color='tab:grey', alpha=0.3)
        axs[i,1].plot(np.diff(bins)*.5 + bins[:-1], (te/te_normc - tr/tr_normc)/(np.sqrt(1e-8 + te/(te_normc**2.) + tr/(tr_normc**2.))),color='black', marker='o')
        axs[i,1].set_ylim(-5, 5)
        axs[i,1].set_yticks(np.arange(-4, 4.01, 2))
        axs[i,1].set_ylabel('(test - train)/\nerror')
        axs[i,0].set_ylabel('Count')
        axs[i,1].set_xlabel('{}, normalized'.format(f), fontsize=15)
    for j in range(i+1, len(axs)):
        axs[j,0].axis('off')
        axs[j,1].axis('off')
    
    fig.suptitle(title, fontsize=26, y=1.01, va='bottom')
#         axs[i,0].set_xticks([])
#         axs[i,0].set_xlim(axs[i,1].get_xlim())
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
        
def create_history_series(history, kfold_n, injection_frac, model_name):
    hist_df = pd.DataFrame(history.history).stack().swaplevel(0,1).sort_index()
    hist_df.index.names = ['metric', 'epoch']

    for idx_key, idx_name in zip([kfold_n, injection_frac, model_name], ['kfold', 'sb_ratio', 'model']):
        hist_df = pd.concat({idx_key: hist_df}, names=[idx_name])
    return hist_df

def update_pickle(key, new, path):
    comb = None
    if os.path.exists(path):
        comb = pd.read_pickle(path)
    if comb is not None:
        if key in comb.index:
            return
    comb = pd.concat([comb, new])
    comb.to_pickle(path)

def load_pickle(path):
    return pd.read_pickle(path)

def create_data_series(res, kfold_n, injection_frac, model_name):
    df = res.stack()
    for idx_key, idx_name in zip([kfold_n, injection_frac, model_name], ['kfold', 'sb_ratio', 'model']):
        df = pd.concat({idx_key: df}, names=[idx_name])
    return df

def check_data_key(key, path):
    exists = False
    try:
        exists = key in load_pickle(path).index
    except FileNotFoundError:
        pass
    return exists

from tensorflow.keras import layers
from tqdm.notebook import tqdm
from keras import backend as K




def variational_AE(columns=hlf_to_train, intermediate_dim=8, latent_dim=1):
    # This is the size of our encoded representations
    
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=0.1)
        return z_mean + K.exp(z_log_sigma) * epsilon
    original_dim = len(hlf_to_train)
    inputs = keras.Input(shape=(original_dim,))
    h = layers.Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_sigma = layers.Dense(latent_dim)(h)
    z = layers.Lambda(sampling)([z_mean, z_log_sigma])
    # Create encoder
    encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

    # Create decoder
    latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = layers.Dense(original_dim, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = keras.Model(inputs, outputs, name='vae_mlp')
    
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    
    return vae

N_early_stopping_epochs = 10
N_epochs = 200
batch_size = 500
verbose = False
model_name = 'variational1'
title_desc = '{} model | S/B = {} | Kfold #{}'
plot_desc = 'plots/train_test_verifications/{}-model_{}-sbr_kfold-{}.pdf'

data_save_path = 'results/results_{}-model.pkl'.format(model_name)
hist_save_path = 'results/history_{}-model.pkl'.format(model_name)

pbar = tqdm(
    total=len(injection_fracs)*len(train_test_idx),
)

for i,frac in enumerate(injection_fracs):
    x, y, tt = data_x[i], data_y[i], data_ttbar[i]
    
    for kfold,(train_idx, test_idx) in enumerate(train_test_idx[i]):
        
     
        key = (model_name, frac, kfold)
        pbar.set_description(title_desc.format(*key))
        
        if not check_data_key(key, data_save_path):
            x_train, x_test, y_train, y_test = x.loc[train_idx], x.loc[test_idx], y[train_idx], y[test_idx]
            x_tr_norm, x_te_norm = minmax_norm(x_train, x_test, hlf_to_train)

            _, x_ttbar_norm = minmax_norm(x_train, tt, hlf_to_train)

            model = variational_AE()

            # add a callback
            callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=N_early_stopping_epochs)

            # plot and save normalized data, right before saving
            plot_train_test_datasets(
                x_tr_norm, x_te_norm, x_ttbar_norm, title=title_desc.format(*key), 
                save_path=plot_desc.format(*key), show=False
            )

            # fit model
            history = model.fit(
                x=x_tr_norm.values,
                y=x_tr_norm.values,
                validation_data=(
                    x_te_norm.values,
                    x_te_norm.values
                ),
                epochs=N_epochs,
                batch_size=batch_size,
                shuffle=True,
                verbose=verbose,
                callbacks=[callback],
            )

            update_pickle(
                key,
                create_history_series(history, *reversed(key)),
                hist_save_path
            )

            x_eval_norm = pd.concat([x_te_norm, x_ttbar_norm]).reset_index(drop=1)
            y_eval = np.concatenate([y_test, np.ones(len(x_ttbar_norm))])
            is_kfold = kfold*np.ones_like(y_eval)
            kfold_part_tag = np.concatenate([np.ones_like(y_test), np.zeros(len(x_ttbar_norm))])

            x_eval_normhat = model.predict(x_eval_norm)
            x_err_norm = x_eval_norm - x_eval_normhat
            y_eval_hat = keras.losses.binary_crossentropy(x_eval_norm.values, x_eval_normhat).numpy()

            res = pd.concat([pd.DataFrame({'y': y_eval, 'yhat': y_eval_hat, 'kfold': is_kfold, 'in_fold': kfold_part_tag}, ), x_eval_norm.reset_index(drop=True)], axis=1)
            res.index.name = 'count'

            update_pickle(
                key,
                create_data_series(res, *reversed(key)),
                data_save_path
            )

        pbar.update(1)
