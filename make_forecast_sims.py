import numpy as np
from scipy import optimize
import healpy as hp
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u 
from scipy.stats import norm
from joblib import Parallel, delayed

def lnlike(s_arr,b_arr,lam,f):
    return np.sum(np.log(lam*s_arr+b_arr))-f*lam

#I think f=1 in the limit where we detect all signal AGNs?
f=1.
lam=1.
H0=70.
omegam=0.3
mapname='GW190521_NRSurPHM_skymap_moresamples_1024.fits.gz'
z_min_b = 0.01 #Minimum z for background events
z_max_b = 2.

N_followup_samples=10000
N_GW_followups=20
#Expected number of signal and back events for Poissoin distr
B_expected_n= 1e-4*20000. # How many do we expect???
njobs=40
cosmo = FlatLambdaCDM(H0=H0, Om0=omegam)

hs=fits.open(mapname)[1]
pb = hs.data['PROB']
distnorm= hs.data['DISTNORM']
distmu= hs.data['DISTMU']
distsigma= hs.data['DISTSIGMA']
NSIDE = hs.header['NSIDE']
pixarea = hp.nside2pixarea(NSIDE, degrees=True)
#hpixs=np.range(hp.nside2npix(NSIDE)) 
pb_frac = 0.90
idx_sort = np.argsort(pb)
idx_sort_up = list(reversed(idx_sort))
sum = 0.
id = 0
while sum<pb_frac:
    this_idx = idx_sort_up[id]
    sum = sum+pb[this_idx]
    id = id+1

idx_sort_cut = idx_sort_up[:id]


def followup_sample(k,lamb):
    S_expected_n=1.*lamb
    print("Follow-up sample n ",k)
    Delta_lnlike= []
    #These are all arrays w/ 1 entry per follow-up
    #make sure these are integers!
    S_cands= np.random.poisson(S_expected_n,N_GW_followups)
    B_cands= np.random.poisson(B_expected_n,N_GW_followups)
    N_cands= S_cands+B_cands
    for i in range(N_GW_followups):
        Ncand = N_cands[i]
        #Now get positions and redshifts for the signal candidates
        s_hpixs=np.random.choice(idx_sort_cut, p=pb[idx_sort_cut]/pb[idx_sort_cut].sum(),size=S_cands[i])
        s_zs=[]
        for j in range(S_cands[i]):
            #may have to add in a dist**2 here for the posterior along los...
            s_ds=np.random.normal(loc=distmu[s_hpixs[j]], scale=distsigma[s_hpixs[j]])
            while (s_ds<0):
                s_ds=np.random.normal(loc=distmu[s_hpixs[j]], scale=distsigma[s_hpixs[j]])
            s_zs.append(z_at_value(cosmo.luminosity_distance, s_ds * u.Mpc,zmin=0.00,zmax=5.) )
        #Now get positions and redshifts for the background candidates   
        #Let's assume they are just uniform in comoving volume between z_min and z_max
        s_zs=np.array(s_zs)
        b_hpixs=np.random.choice(idx_sort_cut, size=B_cands[i])
        zs_arr=np.linspace(z_min_b,z_max_b,num=1000)
        zs2=zs_arr**2
        b_zs=np.random.choice(zs_arr, p=zs2/zs2.sum(), size=B_cands[i])
        
        #Positions and z for All candidates in this follow up
        cand_hpixs=np.concatenate((s_hpixs,b_hpixs))
        cand_zs=np.concatenate((s_zs,b_zs))
        cand_ds=np.zeros(cand_zs.shape[0])
        for l in range(cand_zs.shape[0]):
            cand_ds[l]=cosmo.luminosity_distance(cand_zs[l]).value
        
        #s_arr = pb[cand_hpixs] * distnorm[cand_hpixs] * norm(distmu[cand_hpixs], distsigma[cand_hpixs]).pdf(cand_ds) / pixarea
        s_arr = pb[cand_hpixs] * distnorm[cand_hpixs] * norm(distmu[cand_hpixs], distsigma[cand_hpixs]).pdf(cand_ds) * cand_ds**2
        
        
        # The spatial pb for background is 1./npixels in 90%, and divided by pixarea?
        #I think this needs to be properly renormalized....
        #zs_arr_norm=np.linspace(0,10.,num=1000)
        #normaliz = np.trapz(zs2,zs_arr_norm)
        #b_arr = cand_zs**2/normaliz/len(idx_sort_cut) #/pixarea 
        ds_arr_norm=np.linspace(0,10000.,num=1000)
        normaliz = np.trapz(ds_arr_norm**2,ds_arr_norm)
        b_arr = cand_ds**2/normaliz/len(idx_sort_cut)
        
        #Initial value of lambda_hat
        if (s_arr.shape[0])>0:
            #lam_hat = optimize.root_scalar(lam_hat_func, bracket=[0., 1.], method='brentq').root
        
            #print(lam_hat,s_arr,b_arr)
            lam_hat_in =0.5
            conv=10.
            while conv>1e-5:
                lam_arr=np.zeros(s_arr.shape[0])
                for m in range(s_arr.shape[0]):
                    lam_arr[m]=lam_hat_in*s_arr[m]/(lam_hat_in*s_arr[m]+b_arr[m])
                lam_hat_new=lam_arr.sum()/f
                conv=abs(lam_hat_in-lam_hat_new)
                lam_hat_in=lam_hat_new
            lam_hat=lam_hat_in
            Delta_lnlike.append(lnlike(s_arr,b_arr,lam_hat,f)-lnlike(s_arr,b_arr,0.,f))
            
        else:
            #I think we need to do something different here, maybe s_arr needs to be set to 0
            print("s_arr empty") 
    
    np.savetxt('out/TS_'+str(lamb)+'_'+str(k)+'.dat',Delta_lnlike)
    #Here we sum over all GW events that are followed up

    return np.sum(Delta_lnlike)
    
TS = Parallel(n_jobs=njobs)(delayed(followup_sample)(k,lam) for k in range(N_followup_samples))
TS_0 = Parallel(n_jobs=njobs)(delayed(followup_sample)(k,0) for k in range(N_followup_samples))

np.savetxt('TS_GW190521_1000.dat',np.column_stack((TS,TS_0)))


