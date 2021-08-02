import os
import glob
import nibabel as nib
import nibabel.processing as nip
import nibabel.orientations as nio
import numpy as np
import sys
import json
import pandas
sys.path.append("/home/fercho/projects/neuroradiology")


template= nib.load('/media/HD/datasets/neuroradiology/CT_TRI/rawdata/sub-tri001/sub-tri001_ce-art_ct.nii.gz')

def normalize_volume(vol_data):
    """
    Normalize volume, if filter is False then take into acount all volume voxels
    otherwise take only voxels with labels
    :param vol_data:
    :param seeg_data:
    :param filter:
    :param axis:
    :return:
    """
    h, w, d = np.shape(vol_data)
    # print("Min ", np.min(vol_data))
    # print("Max ", np.max(vol_data))
    mean = np.sum(vol_data)/(h*w*d)
    std = np.std(vol_data)
    return (vol_data - mean) / std


def resample_nib(img, voxel_spacing=(1, 1, 1), order=3, cval=-1024):
    """Resamples the nifti from its original spacing to another specified spacing

    Parameters:
    ----------
    img: nibabel image
    voxel_spacing: a tuple of 3 integers specifying the desired new spacing
    order: the order of interpolation

    Returns:
    ----------
    new_img: The resampled nibabel image

    """
    # resample to new voxel spacing based on the current x-y-z-orientation
    aff = img.affine
    shp = img.shape
    zms = img.header.get_zooms()
    # Calculate new shape
    new_shp = tuple(np.rint([
        shp[0] * zms[0] / voxel_spacing[0],
        shp[1] * zms[1] / voxel_spacing[1],
        shp[2] * zms[2] / voxel_spacing[2]
    ]).astype(int))
    new_aff = nib.affines.rescale_affine(aff, shp, voxel_spacing, new_shp)
    new_img = nip.resample_from_to(img, (new_shp, new_aff), order=order, cval=cval)
    print("[*] Image resampled to voxel size:", voxel_spacing)
    return new_img


def rescale_centroids(ctd_list, img, voxel_spacing=(1, 1, 1)):
    """rescale centroid coordinates to new spacing in current x-y-z-orientation

    Parameters:
    ----------
    ctd_list: list of centroids
    img: nibabel image
    voxel_spacing: desired spacing

    Returns:
    ----------
    out_list: rescaled list of centroids

    """
    ornt_img = nio.io_orientation(img.affine)
    ornt_ctd = nio.axcodes2ornt(ctd_list[0])
    if np.array_equal(ornt_img, ornt_ctd):
        zms = img.header.get_zooms()
    else:
        ornt_trans = nio.ornt_transform(ornt_img, ornt_ctd)
        aff_trans = nio.inv_ornt_aff(ornt_trans, img.dataobj.shape)
        new_aff = np.matmul(img.affine, aff_trans)
        zms = nib.affines.voxel_sizes(new_aff)
    ctd_arr = np.transpose(np.asarray(ctd_list[1:]))
    v_list = ctd_arr[0].astype(int).tolist()  # vertebral labels
    ctd_arr = ctd_arr[1:]
    ctd_arr[0] = np.around(ctd_arr[0] * zms[0] / voxel_spacing[0], decimals=1)
    ctd_arr[1] = np.around(ctd_arr[1] * zms[1] / voxel_spacing[1], decimals=1)
    ctd_arr[2] = np.around(ctd_arr[2] * zms[2] / voxel_spacing[2], decimals=1)
    out_list = [ctd_list[0]]
    ctd_list = np.transpose(ctd_arr).tolist()
    for v, ctd in zip(v_list, ctd_list):
        out_list.append([v] + ctd)
    print("[*] Rescaled centroid coordinates to spacing (x, y, z) =", voxel_spacing, "mm")
    return out_list

def reorient_to(img, axcodes_to=('L', 'A', 'S')):
    # Note: nibabel axes codes describe the direction not origin of axes
    # PIR+ = ASL origin
    aff = img.affine

    arr = np.asanyarray(img.dataobj)  # preserve original data type
    fr_ornt = nio.io_orientation(aff)
    # print('input code:', nib.aff2axcodes(aff))
    # print('input code:', nio.ornt2axcodes(fr_ornt))

    to_ornt = nio.io_orientation(template.affine)#nio.axcodes2ornt(axcodes_to)
    ornt_trans = nio.ornt_transform(fr_ornt, to_ornt)
    arr = nio.apply_orientation(arr, ornt_trans)
    aff_trans = nio.inv_ornt_aff(ornt_trans, arr.shape)
    newaff = np.matmul(aff, aff_trans)
    newimg = nib.Nifti1Image(arr, newaff)
    # print("[*] Reoriented from", nio.ornt2axcodes(fr_ornt), "to", nio.ornt2axcodes(to_ornt))
    return newimg


def reorient_centroids_to(ctd_list, img, decimals=1, verb=False):
    """reorient centroids to image orientation

    Parameters:
    ----------
    ctd_list: list of centroids
    img: nibabel image
    decimals: rounding decimal digits

    Returns:
    ----------
    out_list: reoriented list of centroids

    """
    ctd_arr = np.transpose(np.asarray(ctd_list[1:]))
    if len(ctd_arr) == 0:
        print("[#] No centroids present")
        return ctd_list
    v_list = ctd_arr[0].astype(int).tolist()  # vertebral labels
    ctd_arr = ctd_arr[1:]
    ornt_fr = nio.axcodes2ornt(ctd_list[0])  # original centroid orientation
    axcodes_to = nio.aff2axcodes(img.affine)
    ornt_to = nio.axcodes2ornt(axcodes_to)
    trans = nio.ornt_transform(ornt_fr, ornt_to).astype(int)
    perm = trans[:, 0].tolist()
    shp = np.asarray(img.dataobj.shape)
    ctd_arr[perm] = ctd_arr.copy()
    for ax in trans:
        if ax[1] == -1:
            size = shp[ax[0]]
            ctd_arr[ax[0]] = np.around(size - ctd_arr[ax[0]], decimals)
    out_list = [axcodes_to]
    ctd_list = np.transpose(ctd_arr).tolist()
    for v, ctd in zip(v_list, ctd_list):
        out_list.append([v] + ctd)
    if verb:
        print("[*] Centroids reoriented from", nio.ornt2axcodes(ornt_fr), "to", axcodes_to)
    return out_list


def load_centroids(ctd_path):
    """loads the json centroid file

    Parameters:
    ----------
    ctd_path: the full path to the json file

    Returns:
    ----------
    ctd_list: a list containing the orientation and coordinates of the centroids

    """
    with open(ctd_path) as json_data:
        dict_list = json.load(json_data)
        json_data.close()
    ctd_list = []
    for d in dict_list:
        if 'direction' in d:
            ctd_list.append(tuple(d['direction']))
        elif 'nan' in str(d):  # skipping NaN centroids
            continue
        else:
            ctd_list.append([d['label'], d['X'], d['Y'], d['Z']])
    return ctd_list


"""
Read verse data
"""
##vers19
datafolder='/media/HD/datasets/neuroradiology/verse/v19_BIDS_structure'
out_folder='/media/HD/datasets/neuroradiology/npz_files/updated_data/verse19/anatomy'
dataindex=0
prefix = 'verse19'


##vers20
# datafolder='/media/HD/datasets/neuroradiology/verse/v20_BIDS_structure'
# out_folder='/media/HD/datasets/neuroradiology/npz_files/updated_data/verse20/anatomy'
# dataindex=1
# prefix = 'verse20'



data_folders = glob.glob(datafolder+ '/*/')

excel_file= '/media/HD/datasets/neuroradiology/results/google_drive_copy/VerSe_full_dataset_gt_python_read.xlsx'
excel_data_df = pandas.read_excel(excel_file, sheet_name='verse')
ids= excel_data_df['VerseIDs'].tolist()
contrast= excel_data_df['Contrast'].tolist()
dataset= excel_data_df['Verse Dataset'].tolist()
index= np.where(np.asarray(dataset)==dataindex)
ids = np.asarray(ids)[index]
contrast= np.asarray(contrast)[index]
dataset = np.asarray(dataset)[index]



allfiles=[]
alllabels=[]
indices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
modality = ['art', 'nat', 'pv']
vertebrae= ['clear','C1', 'C2','C3','C4','C5','C6','C7', 'T1','T2','T3','T4','T5','T6','T7','T8','T9','T10','T11','T12', 'L1', 'L2', 'L3','L4','L5', 'L6', 'Sacrum', 'Cocygis','T13']


all_files = []
all_json=[]
all_ids=[]
all_contrast=[]
exceptions=[]
all_segs=[]

for id, ce, dataset in zip(ids, contrast, dataset):
    for folder in data_folders:
        subfolder1= os.path.join(folder, 'rawdata', id)
        subfolders2 = os.path.join(folder, 'derivatives', id)

        ctfiles = glob.glob(subfolder1 + '/*_ct.nii.gz')

        if len(ctfiles) >= 1:
            for file1 in ctfiles:
                # idname=os.path.basename(file1)
                base= os.path.basename(file1).split('.')[0][:-3]
                # file2= os.path.join(subfolders2, base + '_seg-vert_msk.nii.gz')
                file2 = os.path.join(subfolders2, base+ '_seg-subreg_ctd.json')
                file3= os.path.join(subfolders2, base+ '_seg-vert_msk.nii.gz')


                if os.path.exists(file1) and os.path.exists(file2) and os.path.exists(file3):
                    print(file1)
                    print(file2)
                    print('='*20)
                    all_files.append(file1)
                    all_json.append(file2)
                    all_contrast.append(ce)
                    all_ids.append(base)
                    all_segs.append(file3)
                else:
                    exceptions.append(base)


print('File not found')
print(exceptions)
print('')

sizes= []

for c,(ctfile,segfile, json_f, id, label) in enumerate(zip(all_files,all_segs,all_json,all_ids,all_contrast)):
    if c>-1:
        print('{}/{}'.format(c + 1, len(all_ids)))
        print(ctfile)
        print(json_f)

        # files, segs, labels, json_files= list_files(folder)
        # allfiles.append(files)
        # alllabels.extend(labels)

        auxfolder = os.path.join(out_folder, id)
        if not os.path.exists(auxfolder):
            os.makedirs(auxfolder)


        print(ctfile)
        img_nib = nib.load(ctfile)
        msk_nib = nib.load(segfile)
        ctd_list = load_centroids(json_f)

        img_nib = reorient_to(img_nib)
        msk_nib = reorient_to(msk_nib)
        ctd_list = reorient_centroids_to(ctd_list, img_nib)

        print(img_nib.header.get_zooms())

        img_iso = resample_nib(img_nib, voxel_spacing=(1, 1, 1), order=3)
        msk_iso = resample_nib(msk_nib, voxel_spacing=(1, 1, 1), order=0, cval=0)
        ctd_iso = rescale_centroids(ctd_list, img_nib, (1, 1, 1))


        vol_data = img_iso.get_data()
        vol_data = normalize_volume(vol_data)

        seg_data = msk_iso.get_data()

        sizes.append(list(np.shape(vol_data)))



        ### get indexes
        ctd_iso.pop(0)
        slices_index = []
        vertindex= []
        for idx, c in enumerate(ctd_iso):
            if c[0] in indices:
                slices_index.append(int(np.ceil(c[-1])))
                vertindex.append(int(c[0]))


        if label =='ce-art':
            gt=0
        elif label=='ce-no':
            gt= 1
        else:
            gt= 2



        for j,idx in zip(slices_index,vertindex):
        # for j in unique:
            if j!=0:
                auxseg= np.copy(seg_data)
                auxseg[auxseg!=idx]=0
                auxseg[auxseg != 0] = 1
                vol_slice = vol_data[:, :, j]
                # plt.imshow(vol_slice, cmap='gray')
                # plt.show()
                seg_slice = auxseg[:, :, j]
                out_file = os.path.join(auxfolder, base + '_'+ str(label) + '_'+ str(vertebrae[idx]) + '.npz')
                print(out_file)
                np.savez(out_file, data=vol_slice, label=gt, seg=seg_slice)

np.savez('file_list.npz', files=allfiles, labels=alllabels)


print('median volume sizes')
print(np.median(np.asarray(sizes), axis=0))
print('median volume mean')
print(np.mean(np.asarray(sizes), axis=0))







