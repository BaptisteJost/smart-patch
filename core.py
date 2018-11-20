import numpy as np
import healpy as hp
import scipy
import sys
import matplotlib.pyplot as plt
import pysm
from fgbuster.pysm_helpers import get_instrument, get_sky
from fgbuster.component_model import CMB, Dust, Synchrotron
from fgbuster.separation_recipies import _get_prewhiten_factors,_A_evaluators
import fgbuster.algebra as fgal
import numdifftools
import copy

#P_i = list of list for component i, each list corresponds to the pixel of one patch, i.e. P_d=[patch0, patch1,...,patchn] & patchi=[pixel0,pixel1,...]
#input_set_of_betas is a list of all betas for each component and patch i.e. input_set_of_betas=[beta_d1,beta_d2,...,beta_dn,beta_s1,...,beta_sm,temp1,...,templ]
    #by convention the component are ordered as beta_dust,beta_sync,temperature_dust
"""

freq_maps: frequency maps given by the fgbuster package, see fgbuster documentations for more details.

P_i : Patch list -> list of list defining the patches in the sky P_i[0] is the first patch in the sky it's the list of pixels which constitutes the patch
P_d: dust patches
P_t: temperature patches
P_s: synchrotron patches
pixel_number: number of unmasked pixels in the sky
masked_pixels: list of masked pixel
pixel_list: list of unmasked pixels

input_set_of_betas

/!\ the noise was added manualy not using fgbuster due to issues in the randomness of the fgbuster noise

"""


#------------------------------------------------------SKY GENERATION----------------------------------------------------
nside_comparison_for_fine_tuning=False
MASK=True
if nside_comparison_for_fine_tuning!=True:
    nside = 2
    pixel_number=hp.nside2npix(nside)
    pysm_model='c1d1s1'
    sky = get_sky(nside, pysm_model)
    nu = np.array([40.0, 50.0, 60.0, 68.4, 78.0, 88.5, 100.0, 118.9, 140.0, 166.0, 195.0, 234.9, 280.0, 337.4, 402.1])

    pixel_number=hp.nside2npix(nside)

    npix=hp.nside2npix(nside)
    instrument = get_instrument(nside, 'litebird')
    self=instrument

    pix_amin2 = 4. * np.pi / float(hp.nside2npix(self.Nside)) * (180. * 60. / np.pi) ** 2
    """sigma_pix_I/P is std of noise per pixel. It is an array of length
    equal to the number of input maps."""
    sigma_pix_I = np.sqrt(self.Sens_I ** 2 / pix_amin2)
    sigma_pix_P = np.sqrt(self.Sens_P ** 2 / pix_amin2)

    np.random.seed (seed=None)
    noise = np.random.randn(len(self.Sens_I), 3, npix)
    noise[:, 0, :] *= sigma_pix_I[:, None]
    noise[:, 1, :] *= sigma_pix_P[:, None]
    noise[:, 2, :] *= sigma_pix_P[:, None]

    """
    /!\ the noise was added manualy not using fgbuster due to issues in the randomness of the fgbuster noise
    """

    freq_maps = instrument.observe(sky, write_outputs=False)[0] + noise
    freq_maps=freq_maps[:,1:,:]  #on retire la temperature (1:) car on ne prend en compte que Q&U pas I
    """
    the intensity part of the freq maps is removed here since we only care for the Q & U components
    """
    components = [CMB(), Dust(150.), Synchrotron(20.)]
    prewhiten_factors = _get_prewhiten_factors(instrument, freq_maps.shape)



    #----------------------------------------------------NO MASK---------------------------------------------------------------------
    if MASK==False:
        mask_bin=[1]*pixel_number
        masked_pixels=[]



    #----------------------------------------------------MASK-------------------------------------------------------------------------
    else:
        """
        A mask is applied on the frequency maps to cover the galaxy
        """
        mask=hp.read_map('../map_test/HFI_Mask_GalPlane-apo2_2048_R2.00.fits',field=2)
        mask_bin=mask*0
        mask_bin=hp.ud_grade(mask_bin,nside)
        mask_bin[np.where(hp.ud_grade(mask,nside)!=0)[0]]=1
        masked_pixels=[np.where(hp.ud_grade(mask,nside)==0)[0]]
        print ' number of masked_pixels=',len(masked_pixels[0])

    #--------------------------------------------------------------------------------------------------------------------------
    freq_maps_save=freq_maps[:][:][:]
    if len(masked_pixels)!=0:
        freq_maps= np.zeros((len(freq_maps_save), len(freq_maps_save[0]),pixel_number-len(masked_pixels[0])))
        for i in range(len(freq_maps_save)):
            for j in range(len(freq_maps_save[i])):
                freq_maps[i][j]=np.delete(freq_maps_save[i][j],masked_pixels)
        pixel_list=[]
        temp=[]
        for j in range(pixel_number):
            for i in range(len(masked_pixels[0])):
                if j==masked_pixels[0][i]:
                    temp.append(j)
            if len(temp)==0:
                pixel_list.append(j)
            temp=[]
        pixel_number=pixel_number-len(masked_pixels[0])
    else:
        pixel_list=range(pixel_number)

    noise=noise[:,1:,:]

def data_patch(freq_maps,P_i,pixel_list): #this function returns the data corresponding to the pixels in the input patch. Works for patches as well as for meta patches
    """
    takes the frequency maps and a Patch list as input to return the data corresponding to the patch i.e. in the same format as the patch list
    """
    data=[]
    temp_list=[[[]for k in range(len(freq_maps[0]))] for i in range(len(freq_maps))]
    index_list=[]
    for i in range(len(P_i)):
        for f in range(len(freq_maps)):
            for s in range(len(freq_maps[0])):
                for p in range(len(P_i[i])):
                    temp_list[f][s].append(freq_maps[f][s][pixel_list.index(P_i[i][p])])
        data.append(temp_list)# for p in range(len(meta_P[i])))
        temp_list=[[[]for k in range(len(freq_maps[0]))] for i in range(len(freq_maps))]
    del temp_list
    return data

def patch_map(input_set_of_betas,P_d,P_t,P_s,nside):
    """ takes the set of betas as well as the patches definition for each of the three CMB parameters to return a map of the sky for each of the parameter,
        each parameter being indexed in the list at the place of the corresponding pixel
        e.g. the dust parameter for pixel 3 will be located at map_bs[2]"""
    ind=0
    pixel_number1=hp.nside2npix(nside)
    map_bd=[None]*pixel_number1
    map_bt=[None]*pixel_number1
    map_bs=[None]*pixel_number1
    for i in range(len(P_d)):
        for j in range(len(P_d[i])):
            map_bd[P_d[i][j]]=input_set_of_betas[ind]
        ind+=1

    for i in range(len(P_t)):
        for j in range(len(P_t[i])):
            map_bt[P_t[i][j]]=input_set_of_betas[ind]
        ind+=1

    for i in range(len(P_s)):
        for j in range(len(P_s[i])):
            map_bs[P_s[i][j]]=input_set_of_betas[ind]
        ind+=1
    return map_bd,map_bt,map_bs

def joint_spectral_likelihood(input_set_of_betas,fun,P_d,P_t,P_s,pixel_list,nside,jac): #computes the joint likelihood for the whole sky taking into account the patches
    """
    adds all the log(likelihood) for each pixels to get the global likelihood. Each individual pixel likelihood being generated using fgbuster
    """
    logL_spec=0
    map_bd,map_bt,map_bs=patch_map(input_set_of_betas,P_d,P_t,P_s,nside)

    for p in pixel_list:
        logL_spec+=fun[p]([map_bd[p],map_bt[p],map_bs[p]])
    del map_bd,map_bt,map_bs
    return logL_spec


def patch_making_pbyp(input_beta_zeroth_iteration,fisher_list,pixel_number,pixel_list):
    """
    gives a new set of patches for the pixels, one set for each of the three components. First it ranges the pixels by increasing order of the amplitude of the corresponding component.
    Patches are then created taking the first two pixels of this list, the fisher matrices resulting from the estimation of the component are then summed up and reversed.
    The resulting fisher components are then compared with the standard deviation of the pixel components.
    If the inverse of the sum of the fisher matrices (sigma) is bigger than the std deviation squared then the next pixel from the list is added to the patch. Until this condition is met pixels will be added.
    Once the condition is met an new patch will be created and the process starts again until all the pixels are part of a patchself.
    Then again the process is repeated for the two remaining components.

    the function needs the list of the beta "input_beta_zeroth_iteration" as well as the fisher matrices corresponding for each pixel "fisher_list", the number of pixels and the list of unmasked pixels.
    """
    sorted_beta_d_pixel_list=sorted (range(pixel_number),key=lambda k: input_beta_zeroth_iteration[k])
    sorted_beta_t_pixel_list=sorted (range(pixel_number),key=lambda k: input_beta_zeroth_iteration[k+pixel_number])
    sorted_beta_s_pixel_list=sorted (range(pixel_number),key=lambda k: input_beta_zeroth_iteration[k+pixel_number*2])
    """
    print 'sorted beta d'
    for p in range(pixel_number):
        # print 'pixel_list[sorted_beta_d_pixel_list[p]]',pixel_list[sorted_beta_d_pixel_list[p]]
        # print 'sorted_beta_d_pixel_list[p]',sorted_beta_d_pixel_list[p]
        # print 'p',p
        print input_beta_zeroth_iteration[sorted_beta_d_pixel_list[p]]

    print 'sorted beta t'
    for p in range(pixel_number):
        print input_beta_zeroth_iteration[sorted_beta_t_pixel_list[p]+pixel_number]

    print 'sorted beta s'
    for p in range(pixel_number):
        print input_beta_zeroth_iteration[sorted_beta_s_pixel_list[p]+2*pixel_number]
    """
    P_d_new=[]
    P_t_new=[]
    P_s_new=[]
    sigma_patch_list=[]

    beta2_patch_list=[]



    #-----------------------------------------------------------------------------------P_d_new--------------------------------------------------------------------

    sigma_patch_temp=0 #arbitrary
    delta_beta_temp=0.0 #arbitrary
    pixel_patch_temp=[]
    p_list=[]
    patch_list=[]
    fisher_tot=0
    p=0


    patchnum=1

    while p!=pixel_number:

        fisher_tot+=fisher_list[pixel_list[sorted_beta_d_pixel_list[p]]]
        pixel_patch_temp.append(pixel_list[sorted_beta_d_pixel_list[p]])
        p_list.append(p)
        sigma_patch_temp=(np.linalg.inv(fisher_tot))[0][0]

        p+=1
        while sigma_patch_temp >= delta_beta_temp*delta_beta_temp:
            if p==pixel_number:

                break


            fisher_tot+=fisher_list[pixel_list[sorted_beta_d_pixel_list[p]]]

            sigma_patch_temp=(np.linalg.inv(fisher_tot))[0][0]
            pixel_patch_temp.append(pixel_list[sorted_beta_d_pixel_list[p]])
            p_list.append(p)

            delta_beta_temp=np.std([input_beta_zeroth_iteration[sorted_beta_d_pixel_list[x]] for x in p_list])
            p+=1

        patchnum+=1

        P_d_new.append(pixel_patch_temp)
        sigma_patch_list.append(sigma_patch_temp)
        beta2_patch_list.append(delta_beta_temp*delta_beta_temp)
        pixel_patch_temp=[]
        p_list=[]
        sigma_patch_temp=0 #arbitrary
        delta_beta_temp=0.0 #arbitrary
        fisher_tot=0
    print 'patch number for beta d =',patchnum
    #-----------------------------------------------------------------------------------P_t_new--------------------------------------------------------------------

    sigma_patch_temp=0 #arbitrary
    delta_beta_temp=0.0 #arbitrary
    pixel_patch_temp=[]
    p_list=[]
    patch_list=[]
    fisher_tot=0
    p=0
    patchnum=1
    while p!=pixel_number:
        fisher_tot+=fisher_list[pixel_list[sorted_beta_t_pixel_list[p]]]
        pixel_patch_temp.append(pixel_list[sorted_beta_t_pixel_list[p]])
        p_list.append(p)
        sigma_patch_temp=(np.linalg.inv(fisher_tot))[0][0]

        p+=1
        while sigma_patch_temp >= delta_beta_temp*delta_beta_temp:
            if p==pixel_number:
                break
            fisher_tot+=fisher_list[pixel_list[sorted_beta_t_pixel_list[p]]]

            sigma_patch_temp=(np.linalg.inv(fisher_tot))[0][0]
            pixel_patch_temp.append(pixel_list[sorted_beta_t_pixel_list[p]])
            p_list.append(p)
            delta_beta_temp=np.std([input_beta_zeroth_iteration[sorted_beta_t_pixel_list[x]+pixel_number] for x in p_list])
            p+=1

        patchnum+=1
        sigma_patch_list.append(sigma_patch_temp)
        beta2_patch_list.append(delta_beta_temp*delta_beta_temp)
        P_t_new.append(pixel_patch_temp)
        pixel_patch_temp=[]
        p_list=[]
        sigma_patch_temp=0 #arbitrary
        delta_beta_temp=0.0 #arbitrary
        fisher_tot=0

    print 'patch number for beta t =',patchnum
    #-----------------------------------------------------------------------------------P_s_new--------------------------------------------------------------------

    sigma_patch_temp=0 #arbitrary
    delta_beta_temp=0.0 #arbitrary
    pixel_patch_temp=[]
    p_list=[]
    patch_list=[]
    fisher_tot=0
    p=0
    patchnum=1
    while p!=pixel_number:

        fisher_tot+=fisher_list[pixel_list[sorted_beta_s_pixel_list[p]]]
        pixel_patch_temp.append(pixel_list[sorted_beta_s_pixel_list[p]])
        p_list.append(p)
        sigma_patch_temp=(np.linalg.inv(fisher_tot))[0][0]

        p+=1
        while sigma_patch_temp >= delta_beta_temp*delta_beta_temp:
            if p==pixel_number:
                break
            fisher_tot+=fisher_list[pixel_list[sorted_beta_s_pixel_list[p]]]

            sigma_patch_temp=(np.linalg.inv(fisher_tot))[0][0]
            pixel_patch_temp.append(pixel_list[sorted_beta_s_pixel_list[p]])
            p_list.append(p)
            delta_beta_temp=np.std([input_beta_zeroth_iteration[sorted_beta_s_pixel_list[x]+2*pixel_number] for x in p_list])
            p+=1

        patchnum+=1
        sigma_patch_list.append(sigma_patch_temp)
        beta2_patch_list.append(delta_beta_temp*delta_beta_temp)
        P_s_new.append(pixel_patch_temp)
        pixel_patch_temp=[]
        p_list=[]
        sigma_patch_temp=0 #arbitrary
        delta_beta_temp=0.0 #arbitrary
        fisher_tot=0

    print 'patch number for beta d =',patchnum
    return P_d_new,P_t_new,P_s_new,sigma_patch_list,beta2_patch_list

def fisher_pixel(map_bd,map_bt,map_bs,fun,pixel_list,freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB,Hessian=False):
    """
    computes the fisher matrices for each pixel and outputs it as a list of matrices with indices matching the ones of the pixels in the sky
    """
    fisher_list=[None]*hp.nside2npix(nside)

    s_list=[None]*hp.nside2npix(nside)
    n_cmb_list=[None]*hp.nside2npix(nside)
    fisher_s_list=[None]*hp.nside2npix(nside)
    fisher_n_list=[None]*hp.nside2npix(nside)
    fisher_hess_list=[None]*hp.nside2npix(nside)
    i=0
    for p in pixel_list:
        s=fgal.Wd(A_ev([map_bd[p],map_bt[p],map_bs[p]]),np.transpose(freq_maps)[i], np.diag(prewhiten_factors**2), return_svd=False)
        n_cmb_list[p]=fgal.Wd(A_ev([map_bd[p],map_bt[p],map_bs[p]]),np.transpose(noise)[i], np.diag(prewhiten_factors**2), return_svd=False)

        if Hessian == True:
            fisher_hess_list[p] = numdifftools.Hessian(fun[p])([map_bd[p],map_bt[p],map_bs[p]])
        i+=1

        fisher_list[p]=fgal.fisher_logL_dB_dB(A_ev([map_bd[p],map_bt[p],map_bs[p]]), s, A_dB_ev([map_bd[p],map_bt[p],map_bs[p]]), comp_of_dB, np.diag(prewhiten_factors**2), return_svd=False)

        s_list[p]=s



        if np.isnan(s[0][0]) or np.isnan(s[0][1]) or np.isnan(s[0][2]) or np.isnan(s[1][0]) or np.isnan(s[1][1]) or np.isnan(s[1][2]):
            print ' /!\ NAN in S /!\ '
            print 'at pixel %s'%p
            print 's',s

    if Hessian==True:
        return fisher_list,s_list,n_cmb_list,fisher_s_list,fisher_n_list,fisher_hess_list
    return fisher_list,s_list,n_cmb_list,fisher_s_list,fisher_n_list


def return_fun(pixel,freq_maps,prewhiten_factors,A_ev):
    return lambda x: -fgal.logL(A_ev(x), np.transpose(freq_maps)[pixel], np.diag(prewhiten_factors**2 ) )


def return_jac(pixel,freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB):
    return lambda x :-fgal.logL_dB(A_ev(x), np.transpose(freq_maps)[pixel], np.diag(prewhiten_factors**2), A_dB_ev(x), comp_of_dB, return_svd=False)

def return_jac_super_list(input_set_of_betas,fun,P_d,P_t,P_s,pixel_list,nside,jac): # /!\!!!!!!  NE MARCHE QUE DANS CE CAS PRECIS OU IL Y A AUTANT D'ELEMENT DANS P_i QUE DE PIXEL !!!!
    map_bd,map_bt,map_bs=patch_map(input_set_of_betas,P_d,P_t,P_s,nside)
    jac_super_temp=0
    jac_super_list=[]
    for patch in range(len(P_d)):
        for p in P_d[patch]:
            jac_super_temp+=return_jac(pixel_list.index(p),freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB) \
                                        ( [map_bd[p] , map_bt[p] , map_bs[p]] ) [0]
        jac_super_list.append(jac_super_temp)
        jac_super_temp=0
    for patch in range(len(P_t)):
        for p in P_t[patch]:
            jac_super_temp+=return_jac(pixel_list.index(p),freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB) \
                                        ( [map_bd[p] , map_bt[p] , map_bs[p]] ) [1]
        jac_super_list.append(jac_super_temp)
        jac_super_temp=0
    for patch in range(len(P_s)):
        for p in P_s[patch]:
            jac_super_temp+=return_jac(pixel_list.index(p),freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB) \
                                        ( [map_bd[p] , map_bt[p] , map_bs[p]] ) [2]
        jac_super_list.append(jac_super_temp)
        jac_super_temp=0

    return np.array(jac_super_list)



#-----------------------------------------------ZEROTH ITERATION----------------------------------------------------------------------
minimization=True
plot_FOM=False



P_d=[]
P_s=[]
P_t=[]
h=0
for i in range(len(freq_maps[0][1])): P_d.append([pixel_list[i]])
for i in range(len(freq_maps[0][1])): P_s.append([pixel_list[i]])
for i in range(len(freq_maps[0][1])): P_t.append([pixel_list[i]])


#beta initialisation
input_set_of_betas=[]
for i in range(len(freq_maps[0][0])):
    input_set_of_betas.append(1.59)
for i in range(len(freq_maps[0][0])):
    input_set_of_betas.append(19.6)
for i in range(len(freq_maps[0][0])):
    input_set_of_betas.append(-3.1)

if pysm_model=='c1d1s1':
    bounds=((1.45,1.7),(1.0,100),(-7.5,0.5)) #GOOOOOD ONES FOR C1D1S1 DON'T TOUCH!!!!!!!!!!
if pysm_model=='d0s0':
    bounds=((0.5,2.5),(1.0,75),(-7.5,0.5)) # Not the best one yet, needs some more fine tuning

fun=[]
last_values=[]
data=data_patch(freq_maps,P_d,pixel_list)
A_ev, A_dB_ev, comp_of_param, x0, params = _A_evaluators(components, instrument, prewhiten_factors=None)#prewhiten_factors)

A_dB_ev, comp_of_dB = fgal._A_dB_ev_and_comp_of_dB_as_compatible_list(A_dB_ev, comp_of_param, x0)


fun=[None]*hp.nside2npix(nside)
last_values=[None]*hp.nside2npix(nside)
jac_list=[]

comp=[]
minimization_result_pixel=[]
input_beta_zeroth_iteration=[]

# for l in range(pixel_number):
#     fun[pixel_list[l]] = return_fun(l,freq_maps,prewhiten_factors,A_ev)
#     jac_list.append(return_jac(l,freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB))

for p in range(pixel_number):
    fun[pixel_list[p]]=return_fun(p,freq_maps,prewhiten_factors,A_ev)
    jac=return_jac(p,freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB)
    minimization_result_pixel.append(scipy.optimize.minimize(fun[pixel_list[p]],np.array([input_set_of_betas[p],input_set_of_betas[p+len(P_d)],input_set_of_betas[p+len(P_d)+len(P_t)]]),\
    jac=jac, tol=1e-18,bounds=bounds).x)
    # minimization_result_pixel.append(scipy.optimize.minimize(fun[pixel_list[p]],np.array([input_set_of_betas[p],input_set_of_betas[p+len(P_d)],input_set_of_betas[p+len(P_d)+len(P_t)]]),\
    # jac=jac_list[p], tol=1e-18,bounds=bounds).x)



for i in range(pixel_number):
    input_beta_zeroth_iteration.append(minimization_result_pixel[i][0])
for i in range(pixel_number):
    input_beta_zeroth_iteration.append(minimization_result_pixel[i][1])
for i in range(pixel_number):
    input_beta_zeroth_iteration.append(minimization_result_pixel[i][2])
minimization_result_pixel=[]

map_bd,map_bt,map_bs=patch_map(input_beta_zeroth_iteration,P_d,P_t,P_s,nside)
fisher_list,s_list,n_cmb_list,fisher_s_list,fisher_n_list,fisher_hess_list=fisher_pixel(map_bd,map_bt,map_bs,fun,pixel_list,freq_maps,prewhiten_factors,A_ev,A_dB_ev,comp_of_dB,Hessian=True)


#---------------------------------------------------------------New patches with pbyp method-------------------------------------------------------------------

P_d_pbyp , P_t_pbyp , P_s_pbyp , sigma_patch_list , beta2_patch_list = patch_making_pbyp(input_beta_zeroth_iteration,fisher_list,pixel_number,pixel_list)
map_bd_new_sigma,map_bt_new_sigma,map_bs_new_sigma=patch_map(sigma_patch_list,P_d_pbyp,P_t_pbyp,P_s_pbyp,nside)
map_bd_new_beta2,map_bt_new_beta2,map_bs_new_beta2=patch_map(beta2_patch_list,P_d_pbyp,P_t_pbyp,P_s_pbyp,nside)

#---------------------------------------------------------new beta estimation over patches and corresponding map display------------------------------------
input_set_of_betas_pbyp=[]
for i in range(len(P_d_pbyp)):
    input_set_of_betas_pbyp.append(np.mean([map_bd[x] for x in P_d_pbyp[i]]))
for i in range(len(P_t_pbyp)):
    input_set_of_betas_pbyp.append(np.mean([map_bt[x] for x in P_t_pbyp[i]]))
for i in range(len(P_s_pbyp)):
    input_set_of_betas_pbyp.append(np.mean([map_bs[x] for x in P_s_pbyp[i]]))

import itertools
jac=[]
point_number=25
list_min_khi_jsl_dust=[]
list_min_khi_jsl_temp=[]
list_min_khi_jsl_sync=[]
beta_d_range=np.linspace(0.5,2.5,point_number)
beta_t_range=np.linspace(15,35,point_number)
beta_s_range=np.linspace(-4.0,-2.0,point_number)
khi_list_jsl_dust=np.zeros(point_number)
khi_list_jsl_temp=np.zeros(point_number)
khi_list_jsl_sync=np.zeros(point_number)

for patch in range(len(P_d_pbyp)):
    for x in range(point_number):
        khi_list_jsl_dust[x]=joint_spectral_likelihood(list(itertools.chain.from_iterable([input_set_of_betas_pbyp[:patch],[beta_d_range[x]],input_set_of_betas_pbyp[patch+1:]])),fun,P_d_pbyp,P_t_pbyp,P_s_pbyp,pixel_list,nside,jac)
    list_min_khi_jsl_dust.append(beta_d_range[np.argmin(khi_list_jsl_dust)])
    khi_list_jsl_dust=np.zeros(point_number)

for patch in range(len(P_t_pbyp)):
    for x in range(point_number):
        khi_list_jsl_temp[x]=joint_spectral_likelihood(list(itertools.chain.from_iterable([input_set_of_betas_pbyp[:len(P_d_pbyp)+patch],[beta_t_range[x]],input_set_of_betas_pbyp[len(P_d_pbyp)+patch+1:]])),fun,P_d_pbyp,P_t_pbyp,P_s_pbyp,pixel_list,nside,jac)
    list_min_khi_jsl_temp.append(beta_t_range[np.argmin(khi_list_jsl_temp)])
    khi_list_jsl_temp=np.zeros(point_number)

for patch in range(len(P_s_pbyp)):
    for x in range(point_number):
        khi_list_jsl_sync[x]=joint_spectral_likelihood(list(itertools.chain.from_iterable([input_set_of_betas_pbyp[:len(P_d_pbyp)+len(P_t_pbyp)+patch],[beta_s_range[x]],input_set_of_betas_pbyp[len(P_d_pbyp)+len(P_t_pbyp)+patch+1:]])),fun,P_d_pbyp,P_t_pbyp,P_s_pbyp,pixel_list,nside,jac)
    list_min_khi_jsl_sync.append(beta_s_range[np.argmin(khi_list_jsl_sync)])
    khi_list_jsl_sync=np.zeros(point_number)

init_jsl = list_min_khi_jsl_dust + list_min_khi_jsl_temp + list_min_khi_jsl_sync

super_bounds=[]
for p in range(len(P_d_pbyp)):
    super_bounds.append((1.45,1.7))
for p in range(len(P_t_pbyp)):
    super_bounds.append((1.0,100))
for p in range(len(P_s_pbyp)):
    super_bounds.append((-7.5,0.5))

new_set_of_beta_pbyp=scipy.optimize.minimize(joint_spectral_likelihood,init_jsl,(fun,P_d_pbyp,P_t_pbyp,P_s_pbyp,pixel_list,nside,return_jac_super_list), tol=1e-18,bounds=super_bounds,jac=return_jac_super_list).x
map_bd_pbyp,map_bt_pbyp,map_bs_pbyp=patch_map(new_set_of_beta_pbyp,P_d_pbyp,P_t_pbyp,P_s_pbyp,nside)
