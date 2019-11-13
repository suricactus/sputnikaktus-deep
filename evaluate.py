from itertools import product

import numpy as np
import rasterio as rio
from rasterio.io import DatasetWriter

from utils import get_patch_offsets, get_buffered_patch



# a = get_patch_offsets((321, 800), (150, 150), 'overlap')

with rio.open('data_slum/train/images/top_60cm_qb_area1.tif', 'r') as img_src:
    b = get_buffered_patch(img_src, (1, 2, 3), (150, 150), (100, 100), 'overlap', 'zeros')
    print(b)


# print(list(a))


# def evaluate_predictions(
#     Xtest, 
#     w8fname,
#     cut, 
#     overlap, 
#     Ytest,
#     model_builder
# ):
#     """
#     Function to predict full tiles strip-wise (as loading whole tiles might not fit in the memory).
#     """
#     Ytest = Ytest.copy()
#     Ytest[Ytest != 2] = 0
#     Ytest[Ytest == 2] = 1
#     ncols, nrows, nbands = Xtest.shape
#     Xtest = np.expand_dims(Xtest, axis=0)
#     total_map = np.zeros((nrows, ncols), dtype=np.uint8)
#     first = True
#     last = False

#     if nrows % cut == 0:
#         numstrips = int(nrows/cut)
#     else:
#         numstrips = int(nrows/cut) + 1


#     for i in range(numstrips):
#         print("Strip number: %d" % i)
#         if first:
#             striptop = 0
#             stripbottom = overlap
#             height = cut
#             X_sub = Xtest[:, cut*i-striptop:(cut*i)+height+stripbottom, :, :]
#             first = False
#         elif (not first) and (cut*(i+1)+1+overlap < nrows):
#             striptop = overlap
#             stripbottom = overlap
#             height = cut
#             X_sub = Xtest[:, cut*i-striptop:(cut*i)+height+stripbottom, :, :]
#         else:
#             print("Last hit!")
#             striptop = overlap
#             stripbottom = 0
#             height = nrows - cut*i
#             if (striptop+height) % (4) != 0:
#                 height = height*4
#             X_sub = Xtest[:, -(striptop+height):, :, :]
#             last = True
#         sub_nrows = X_sub.shape[1]
#         sub_ncols = X_sub.shape[2]
#         model = model_builder(OPT, sub_nrows, sub_ncols,
#                               nbands, NUMBER_CLASSES)
#         model.load_weights(w8fname)
#         sub_ns, sub_nb, _, __ = X_sub.shape
#         cmap = model.predict_on_batch([X_sub])
#         cmap = np.argmax(cmap[0], axis=2)
#         if not last:
#             total_map[cut*i:cut*i+height,
#                       :] = cmap[striptop:striptop+height, 0:total_map.shape[1]]
#         else:
#             total_map[-(height):, :] = cmap[-(height):, 0:total_map.shape[1]]
#     return total_map







# train_history_A = "FCN_DK3_train_history"
# weights_pretrained_A = "pretrained_FCN_DK3.hdf5"
# # let's visualize the learning curve of the pre-trained network
# with open(train_history_A, "rb") as f:
#     FCN_DK3_history = pickle.load(f)
# plt.plot(FCN_DK3_history["loss"])
# plt.title("FCN-DK3 training curve loss")
# plt.ylabel("loss")
# plt.xlabel("epoch")
# plt.show()
# plt.plot(1-np.array(FCN_DK3_history["acc"]))
# plt.title("FCN-DK3 training curve error rate")
# plt.ylabel("error rate")
# plt.xlabel("epoch")
# plt.show()
# # now let's evaluate it on the test tile
# cut = 128
# overlap = 96
# predictions_FCNDK3 = evaluate_predictions(Xtest, weights_pretrained_A, cut, overlap, Ytest, build_FCNDK3)
