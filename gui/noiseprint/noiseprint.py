#
# Copyright (c) 2019 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#
"""
@author: davide.cozzolino
"""

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import os.path
from .network import FullConvNet

slide = 1024  # 3072
largeLimit = 1050000  # 9437184
overlap = 34

chkpt_folder = os.path.join(os.path.dirname(__file__), "./nets/%s_jpg%d/model")
tf.reset_default_graph()
x_data = tf.placeholder(tf.float32, [1, None, None, 1], name="x_data")
net = FullConvNet(x_data, 0.9, tf.constant(False), num_levels=17)
saver = tf.train.Saver(net.variables_list)

configSess = tf.ConfigProto()
configSess.gpu_options.allow_growth = True
# configSess = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.95))


def genNoiseprint(img, QF=101, model_name="net", progress=None, cache_dir=None):
    """Original tile geometry/arithmetic, with optional resumable tile files.

    cache_dir must belong to one immutable input/model, as provided by the
    application's isolated worker. Complete tiles are atomically published.
    """
    from pathlib import Path
    if QF > 100:
        QF = 101
    chkpt_fname = chkpt_folder % (model_name, QF)
    cache = Path(cache_dir) if cache_dir is not None else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    tiled = img.shape[0]*img.shape[1] > largeLimit
    step = slide if tiled else max(img.shape)
    positions = [(y,x) for y in range(0,img.shape[0],step)
                 for x in range(0,img.shape[1],step)]
    result = np.zeros(img.shape,np.float32)
    with tf.Session(config=configSess) as sess:
        saver.restore(sess, chkpt_fname)
        for number,(y,x) in enumerate(positions):
            height = min(step,img.shape[0]-y)
            width = min(step,img.shape[1]-x)
            path = cache/f"{y}-{x}.npy" if cache is not None else None
            if path is not None and path.exists():
                block = np.load(path,allow_pickle=False)
                if block.shape != (height,width) or block.dtype != np.float32:
                    raise ValueError('Invalid cached Noiseprint tile.')
            else:
                clip = img[max(y-overlap,0):min(y+step+overlap,img.shape[0]),
                           max(x-overlap,0):min(x+step+overlap,img.shape[1])]
                block = sess.run(net.output,feed_dict={x_data:clip[None,:,:,None]})[0,:,:,0]
                if y > 0:
                    block = block[overlap:,:]
                if x > 0:
                    block = block[:,overlap:]
                block = block[:height,:width]
                if path is not None:
                    temporary = path.with_suffix('.partial.npy')
                    np.save(temporary,block,allow_pickle=False)
                    temporary.replace(path)
            result[y:y+height,x:x+width] = block
            if progress is not None:
                progress(number+1,len(positions))
    return result
