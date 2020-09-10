# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
import numpy as np
import scipy.misc 
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x
from PIL import Image

import tempfile
import moviepy.editor as mpy
import os
import os.path as osp
import numpy as np

from tensorflow.python.framework import constant_op 
from tensorflow.python.ops import summary_op_util
from IPython import embed

def py_encode_gif(im_thwc, tag, fps=4):
    """
    Given a 4D numpy tensor of images, encodes as a gif.
    """
    with tempfile.NamedTemporaryFile() as f: fname = f.name + '.gif'
    clip = mpy.ImageSequenceClip(list(im_thwc), fps=fps)
    clip.write_gif(fname, verbose=False, logger=None)
    with open(fname, 'rb') as f: enc_gif = f.read()
    os.remove(fname)
    # create a tensorflow image summary protobuf:
    thwc = im_thwc.shape
    im_summ = tf.Summary.Image()
    im_summ.height = thwc[1]
    im_summ.width = thwc[2]
    im_summ.colorspace = 3 # fix to 3 == RGB
    im_summ.encoded_image_string = enc_gif
    return im_summ

    # create a summary obj:
    #summ = tf.Summary()
    #summ.value.add(tag=tag, image=im_summ)
    #summ_str = summ.SerializeToString()
    #return summ_str


class Logger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""

        # Switching to Tensorflow 2 syntax. 
        self.writer = tf.summary.create_file_writer(log_dir)

        # Tensorflow 1 syntax (for logging purposes):
        # self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""

        # Tensorflow 1 syntax (for logging purposes):
        # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        # self.writer.add_summary(summary, step)

        # Switching to Tensorflow 2 syntax. 
        # Adding scalar summar to file writer.  
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)

    def gif_summary(self, tag, images, step):
        """Log a list of TXHXWX3 images."""
        # from https://github.com/tensorflow/tensorboard/issues/39

        img_summaries = []
        for i, img in enumerate(images):
            # Create a Summary value
            img_sum = py_encode_gif(img, '%s/%d' % (tag, i), fps=4)
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    # def image_summary(self, tag, images, step):
    #     """Log a list of images."""

    #     img_summaries = []
    #     for i, img in enumerate(images):
    #         # Write the image to a string
    #         try:
    #             s = StringIO()
    #         except:
    #             s = BytesIO()

    #         # Switching to PIL Image fromarray instead. 
    #         scipy.misc.toimage(img).save(s, format="png")
    #         # Image.fromarray(img).save(s, format="png")

    #         # Create an Image object
    #         img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
    #                                    height=img.shape[0],
    #                                    width=img.shape[1])
    #         # Create a Summary value
    #         img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

    #     # Create and write Summary
    #     summary = tf.Summary(value=img_summaries)
    #     self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):

            img = np.expand_dims(img.transpose([1,2,0]),0)
            # img = np.expand_dims(img,0)
            with self.writer.as_default():
                tf.summary.image(tag, img, step=step)     

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()