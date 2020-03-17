# common functions used throughout the project

# ONLY python import in this file.
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle,sys,os,io,cv2
import numpy as np

def np_log(np_array):
    if type(np_array) is not np.ndarray:
        if type(np_array) is not list:
            np_array = [np_array]
        np_array = np.array(np_array)
    return np.ma.log(np_array).filled(-np.infty)

def np_binarize(np_array,threshold=0.5,inplace=False):
    if inplace:
        np_array[np_array >= threshold] = 1
        np_array[np_array < threshold] = 0
        np_array.astype(np.bool)
        return
    else:
        c_array = np.copy(np_array)
        c_array[c_array >= threshold] = 1
        c_array[c_array < threshold] = 0
        c_array.astype(np.bool)
        return c_array

def write_pickle(data,fn,verbose=True):
    if verbose: print("Writing pickle to [{}]".format(fn))
    with open(fn,'wb') as f:
        pickle.dump(data,f)
    if verbose: print("Save successful.")

def read_pickle(fn,verbose=True):
    if verbose: print("Reading pickle file [{}]".format(fn))
    data = None
    with open(fn,'rb') as f:
        data = pickle.load(f)
    if verbose: print("Load successful.")
    return data

def ndarray_groupby(ndarray_y,ndarray_x):
    # only groupby for first column
    uniq = np.unique(ndarray_y)
    grouped = {yi: np.array(ndarray_x[ndarray_y==yi]) for yi in uniq}
    return grouped
    
#
# matplotlib to numpy
#

def matplotlib_to_numpy(fig, im_shape = None, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf,format='png',transparent=True,
                facecolor=fig.get_facecolor(),
                edgecolor='none',
                dpi=dpi)
    buf.seek(0)
    img = np.frombuffer(buf.getvalue(),dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img,1)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # re-align
    # print(img.shape,im_shape)
    # img = np.rot90(img,k=2)
    # img = np.fliplr(img)
    # print(img.shape,im_shape)
    # re-shape
    if im_shape:
        img = cv2.resize(img,im_shape)
    return img
    
def invert_np_image(ndarray):
    return 255 - ndarray
#
# numpy to image
#

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

#
# Bounding Box functions
#

# tlbr: top-left, bottom-right coordinates (as opposed to tlwh; top-left, width-height)
def bbox_iou_tlbr(bbox_a,bbox_b):
    # bbox intersection
    i = bbox_intersection_area_tlbr(bbox_a,bbox_b)
    u = bbox_union_area_tlbr(bbox_a,bbox_b)
    if i > u:
        print("Error: i > u")
        print(i,u)
        print('a,b',bbox_a,bbox_b)
        print('aa',bbox_area_tlbr(bbox_a))
        print('ab',bbox_area_tlbr(bbox_b))
        exit()
    return i / u

def bbox_union_area_tlbr(bbox_a,bbox_b):
    a = bbox_area_tlbr(bbox_a)
    b = bbox_area_tlbr(bbox_b)
    i = bbox_intersection_area_tlbr(bbox_a,bbox_b)
    return a + b - i

def bbox_intersection_tlbr(bbox_a,bbox_b):
    # verify any intersection
    if ( bbox_a[0] > bbox_b[2] )  or (bbox_b[0] > bbox_a[2]):
        return []
    if ( bbox_a[1] > bbox_b[3] )  or (bbox_b[1] > bbox_a[3]):
        return []

    x1 = max( [ bbox_a[0],bbox_b[0] ] )
    y1 = max( [ bbox_a[1],bbox_b[1] ] )
    x2 = min( [ bbox_a[2],bbox_b[2] ] )
    y2 = min( [ bbox_a[3],bbox_b[3] ] )
    return [x1,y1,x2,y2]

def bbox_intersection_area_tlbr(bbox_a,bbox_b):
    bbox = bbox_intersection_tlbr(bbox_a,bbox_b)
    return bbox_area_tlbr(bbox)

def bbox_area_tlbr(bbox):
    if len(bbox) == 0:
        return 0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return float(w*h)


