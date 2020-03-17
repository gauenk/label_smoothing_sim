"""
Demo for Miami, FL 02/2020

Contains the API functions to display the object detection outputs
"""
# python imports
import cv2
import numpy as np
import numpy.random as npr
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy import stats

# project imports
from base.utils import matplotlib_to_numpy,invert_np_image,np_log

# torch imports
from torchvision import models
import torch

def overlay_bboxes(img,bboxes,labels=None,corner_text=None,thresh=0.7,save_fn=None,fill=False):
    """
    Include the bounding boxes over the input image ndarray
    """
    b_color_raw = np.array([1,0,0]) # blue, BGR order here
    color = (255,255,255)

    # get image max value :: [0,255]? or [0,1]?
    b_color = 1. * b_color_raw
    if img.max() > 1:
        b_color = 255 * b_color_raw

    #print(b_color)
    thickness = 5
    for i,box in enumerate(bboxes):
        score = box[4]
        if score < thresh:
            continue
        box = box.astype(int)
        x,y = box[0],box[1]
        w = box[2] - box[0]
        h = box[3] - box[1]
        if score < thresh:
            continue
        # # up-and-down
        # img[y:y+h,x:x+thickness,:] = b_color
        # img[y:y+h,x+w:x+w+thickness,:] = b_color
        
        # # side-to-side
        # img[y:y+thickness,x:x+w,:] = b_color
        # img[y+h:y+h+thickness,x:x+w,:] = b_color
        
        # fill in box values
        loc1,loc2 = (box[0],box[1]),(box[2],box[3])
        if fill:
            cv2.rectangle(img,loc1,loc2,(255,0,0),cv2.FILLED)
        else:
            cv2.rectangle(img,loc1,loc2,(255,0,0))
                
        # draw ids
        if labels is not None and labels[i] is not None:
            size = 3.5
            thickness = 3
            label = str(labels[i])
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN,
                                        size, thickness)
            center = x + 5 + text_size[0][0],y + 5 + text_size[0][1]
            cv2.putText(img, label, center, cv2.FONT_HERSHEY_PLAIN,
                        size, color, thickness, cv2.LINE_AA)
            
    # write counter
    if corner_text is not None:

        corner_text = str(int(corner_text))
        size = 7.
        thickness = 5
        color = (255,255,255)
        text_size = cv2.getTextSize(corner_text, cv2.FONT_HERSHEY_PLAIN,
                                    size, thickness)
        print(text_size)
        w_start = (5, 15 + text_size[0][1]) # bottom left-corner
        center = (15 + text_size[0][0], 25 + text_size[0][1])
        cv2.rectangle(img,(0,0),center,(0,0,0), -1) # background of text
        cv2.putText(img, corner_text, w_start, cv2.FONT_HERSHEY_PLAIN,
                    size, color, thickness, cv2.LINE_AA)
    return img

def show_bboxes(img,bboxes,save_fn=None,thresh=0.5,getFig=False,ax=None,label=None):
    """ Draw detected bounding boxes """
    #print(bboxes)
    inds = np.where(bboxes[:,-1] >= thresh)[0]
    if len(inds) == 0:
        print("Nothing detected. Nothing saved.")
        return None,None
    img = img[:,:,(0,1,2)] # fix colors

    if ax is None:
        fig,ax = plt.subplots(figsize=(12,12))

    ax.imshow(img,aspect='equal',interpolation='none')
    for i in inds:
        bbox = np.array(bboxes[i,:4],dtype=np.uint32)
        x,y = bbox[0],bbox[1]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        # x,y,w,h = bbox[0],bbox[1],bbox[2],bbox[3] # USE THIS IF GT
        score = bboxes[i, 4]
        ax.add_patch(
            plt.Rectangle( (x,y),
                           w,h,
                           fill = False,
                           edgecolor='green',
                           linewidth=3.5)
        )
        if label is not None:
            ax.text(bbox[0],bbox[1] - 2,
                    label[i],
                    bbox=dict(facecolor='blue',alpha=0.5),
                    fontsize=14,color='white')
    ax.set_title(('Detections with P(person | box) >= {:.2f}'.format(thresh)))
    plt.axis('off')
    plt.tight_layout()
    if getFig: # don't save and get the figures
        return fig,ax
    if save_fn:
        plt.savefig(save_fn,dpi=150,bbox_inches="tight")
    

def show_demographics(det_info,history,save_fn):
    # setup plotting variables
    demograhpics = det_info['demographics']
    im_shapes = det_info['im_shapes']
    bboxes = det_info['bboxes']
    bcenters = get_bbox_locations(im_shapes,bboxes)
    xgrid,ygrid = 300,300
    canvas = np.zeros( (xgrid,ygrid,3), dtype=np.float32)

    # only plot with good enough score
    inds = np.where(bboxes[:,-1] >= thresh)[0]
    if len(inds) == 0:
        print("Nothing detected. Nothing saved.")
        return 

    fig,ax = plt.subplots(figsize=(12,12))
    ax.imshow(img,aspect='equal')
    for i in inds:
        bbox = bboxes[i,:4]
        score = bboxes[i,:5]
        ax.add_path(
            plt.Rectangle( (bbox[0],bbox[1]),
                           bbox[2] - bbox[0],
                           bbox[3] - bbox[1],
                           fill = False,
                           edgecolor='green',
                           linewidth=3.5)
        )
        ax.text(bbox[0],bbox[1] - 2,
                '{:.3f}'.format(score),
                bbox=dict(facecolor='blue',alpha=0.5),
                fontsize=14,color='white')
    ax.set_title(('Detections with P(person | box) >= {:.2f}'.format(thresh)))
    plt.axis('off')
    plt.tight_layout()
    if getFig: # don't save and get the figures
        return fig,ax
    plt.savefig(save_fn,dpi=150,bbox_inches="tight")

def overlay_centers(img,centers,labels=None,thresh=0.7,fill=True,size='big'):
    """
    Include the bounding boxes over the input image ndarray
    """
    b_color = np.array([1,0,0]) # blue, BGR order here

    # get image max value :: [0,255]? or [0,1]?
    img_max = 1. * b_color
    if img.max() > 1:
        img_max = 255 * b_color
    #print(img_max)
    if size == 'small':
        radius = 1
    else:
        radius = 30

    thickness = 5
    if fill:
        thickness = -1
    c_color = (255,255,0)
    img = img # reduce previous frame intensity by half
    for i,center in enumerate(centers):
        if center[2] < thresh:
            continue
        # up-and-down
        center = tuple([int(c) for c in center[:2]])
        cv2.circle(img, center, radius, c_color, thickness=thickness)
        
        # draw ids
        if labels is not None:
            size = 4.
            label_thickness = 5
            color = (255,255,255)
            label = str(labels[i])
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN,
                                        size, label_thickness)
            center = center[0] + text_size[0][0],center[1] + 5 + text_size[0][1]
            cv2.putText(img, label, center, cv2.FONT_HERSHEY_PLAIN,
                        size, color, label_thickness,cv2.LINE_AA)

    return img.astype(np.uint8)
    
def index_img_to_plt(ndarray,nrows):
    """
    ndarray(ndarray): N x 2 for [x,y] locations
    values of the array represent indices of an image
    for an image: (0,0) is top left corner
    for plotting: (0,0) is bottom left corner
    
    This function converts the values of the array 
    from an image index to a plotting index
    """
    ndarray[:,1] = nrows - ndarray[:,1]
    return ndarray

def get_density_image(img):
    """
    im_shape: [ # pixels left-to-right, # pixels top-to-bottom ]
    """

    #
    # setup variables
    #
    # im_shape = im_shape[::-1]

    img = img / img.max() # discrete image
    im_shape = img.shape
    #print(im_shape)
    thresh = 0.85
    sigma = 250.
    z_thresh = 10e-3
    my_dpi = 100
    my_figsize = (im_shape[1] / my_dpi, im_shape[0] / my_dpi) 
    cov = .5 * 10e-3 * np.array([1,3])
    cmap = plt.cm.Purples
    cmap.set_bad(color="white")

    #
    # create figure
    #

    fig = plt.figure(frameon=False,facecolor='white',figsize=my_figsize)
    ax = plt.Axes(fig,[0.,0.,1.,1.])
    fig.add_axes(ax)
    ax.set_axis_off()    
    ax.set_xlim([0,im_shape[1]])
    ax.set_ylim([0,im_shape[0]])


    #
    # plot and ndarray
    #

    ax.heatmap(img)
    img = matplotlib_to_numpy(fig,im_shape[::-1],dpi=my_dpi)
    img = invert_np_image(img)
    # img = np.rot90(img,k=2)
    # img = np.fliplr(img)
    #print("shape := [ # rows x # cols ]",img.shape)
    #cv2.imwrite("./tmp.png",img)
    plt.close(fig)
    return img
    

def get_density_image_old(im_shape,x,y):
    """
    im_shape: [ # pixels left-to-right, # pixels top-to-bottom ]
    """

    #
    # setup variables
    #
    # im_shape = im_shape[::-1]

    #print(im_shape)
    thresh = 0.85
    sigma = 250.
    z_thresh = 10e-3
    my_dpi = 100
    my_figsize = (im_shape[1] / my_dpi, im_shape[0] / my_dpi) 
    centers = np.c_[x,y]
    def nice_cov(*args):
        return .5 * 10e-3 * np.array([1,3])
    cmap = plt.cm.Purples
    cmap.set_bad(color="white")
    
    #
    # create dummy data (testing only)
    #

    # nsamples = 100
    # centers = np.ones( (nsamples,2) )
    # centers[:,0] *= 300 + 10 * npr.normal(size=nsamples)
    # centers[:,1] *= 1500 + 10 * npr.normal(size=nsamples)
    # centers.astype(np.int)

    #
    # create the kde
    #

    #centers += 0.01 * np.random.normal(size=centers.shape)
    centers = index_img_to_plt(centers,im_shape[0])
    kde = stats.gaussian_kde(centers.T)
    kde.set_bandwidth(nice_cov)
    gap_size = 2
    xx,yy = np.mgrid[0:im_shape[1]:gap_size,0:im_shape[0]:gap_size]
    z = kde(np.c_[xx.flat,yy.flat].T).reshape(xx.shape)
    z = np_log(z)
    z = np.ma.masked_where(z > -10e-4, z)

    #
    # create figure
    #

    fig = plt.figure(frameon=False,facecolor='white',figsize=my_figsize)
    ax = plt.Axes(fig,[0.,0.,1.,1.])
    fig.add_axes(ax)
    ax.set_axis_off()    
    ax.set_xlim([0,im_shape[1]])
    ax.set_ylim([0,im_shape[0]])


    #
    # plot and ndarray
    #

    ax.contourf(xx,yy,z,25,cmap=cmap)
    img = matplotlib_to_numpy(fig,im_shape[::-1],dpi=my_dpi)
    img = invert_np_image(img)
    # img = np.rot90(img,k=2)
    # img = np.fliplr(img)
    #print("shape := [ # rows x # cols ]",img.shape)
    #cv2.imwrite("./tmp.png",img)
    plt.close(fig)
    return img
    

def overlay_centers_with_gaussian(img,means,covs,labels=None,fill=True):
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')

    nsamples = len(means)
    thresh = 0.85
    sigma = 2.
    im_shape = img.shape[:2]
    z_thresh = -10e-4
    cmap = plt.cm.Reds
    cmap.set_bad(color="white")
    my_dpi = 100
    my_figsize = (im_shape[0] / my_dpi, im_shape[1] / my_dpi) 

    # plot only the picture content
    fig = plt.figure(frameon=False,facecolor='black',figsize=my_figsize)
    ax = plt.Axes(fig,[0.,0.,1.,1.])
    #fig,ax = plt.subplots(1)
    ax.set_facecolor('black')
    ax.set_axis_off()    
    fig.add_axes(ax)


    levels = [0.001,0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    x = np.linspace(0,im_shape[1]-1,im_shape[1])
    y = np.linspace(0,im_shape[0]-1,im_shape[0])
    xv, yv = np.meshgrid(x,y)
    grid = np.array( [ xv.ravel(),  yv.ravel() ] ).transpose()
    for i in range(nsamples):
        center = means[i,:2]
        if np.all(center <= 1):
            center = center * im_shape[::-1]
        center = center.astype(np.int)
        cov = sigma * covs[i]
        if means[i,2] < thresh:
            continue
        # up-and-down
        z = multivariate_normal.pdf(grid,mean=center,cov=cov)
        z = z.reshape(len(y),len(x)) 
        # z = z / z.max()
        # z[np.where(z < z_thresh)] = 0
        z = np_log(z)
        z = np.ma.masked_where(z > z_thresh, z)
        # print(z.max(),z.shape,len(x),len(y),center)
        ax.contourf( x, y, z, 25, cmap=cmap)#, levels=levels)
    #ax.axis('off')
    img = matplotlib_to_numpy(fig,im_shape[::-1],dpi=my_dpi)
    img = np.rot90(img,k=2)
    img = np.fliplr(img)
    plt.close(fig)
    return img


def get_bbox_centers(im_shape,bboxes,percent=True):
    # bbox = [w1,w2,h1,h2]
    # really bboxes = [x1,y1,x2,y2]
    # im_shape = [ Rows x Columns := X x Y ]
    bcenters = np.zeros( (len(bboxes),3), dtype=np.float32)
    for i,bbox in enumerate(bboxes):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        assert w > 0, "w must be positive"
        assert h > 0, "h must be positive"
        wc = (bbox[2] + bbox[0]) / 2.
        hc = (bbox[3] + bbox[1]) / 2.
        w_perc = wc / im_shape[1]
        h_perc = hc / im_shape[0]
        # print(wc,hc)
        if percent:
            bcenters[i,0] = w_perc
            bcenters[i,1] = h_perc
        else:
            bcenters[i,0] = int(wc)
            bcenters[i,1] = int(hc)
        bcenters[i,2] = float(bbox[4])
        # print('igc',i,bbox,w,h,wc,hc,bcenters[i,0])
    # print('bc',bcenters)
    return bcenters


