import os
import time
import random
from datetime import datetime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import scipy.io
import cv2
#from utils_sceneparsing import *

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))
os.system('rm tmp')
print(os.environ['CUDA_VISIBLE_DEVICES'])

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

crop_size_h = 481
crop_size_w = 641

train_phase = False
# load caffe weight
def weight_from_caffe(caffenet):
    def func(shape, dtype, partition_info=None):
        sc = tf.get_variable_scope()
        name = sc.name.split('/')[-1]
        print ('init: ', name, shape, caffenet[name][0].shape)
        return tf.transpose(caffenet[name][0], perm=[2, 3, 1, 0])
    return func

# load caffe bias
def bias_from_caffe(caffenet):
    def func(shape, dtype, partition_info=None):
        sc = tf.get_variable_scope()
        name = sc.name.split('/')[-1]
        return caffenet[name][1]
    return func

# data process
def myfunc(x):
    try:
        data_dic = scipy.io.loadmat(x)
        data_img = data_dic['img']
        #print "aaaaa"
        data_depth = data_dic['depth']
        depth_mask = np.zeros((crop_size_h, crop_size_w), dtype=np.float32)
        depth_mask[np.where(data_depth < 0.1)] = 0.0
        depth_mask[np.where(data_depth >= 0.1)] = 1.0
        data_norm = data_dic['norm']
        data_mask = data_dic['mask']
        grid = data_dic['grid']
    except:
        data_img = np.zeros((crop_size_h, crop_size_w, 3), dtype=np.float32)
        data_depth = np.zeros((crop_size_h, crop_size_w), dtype=np.float32)
        data_mask = np.zeros((crop_size_h, crop_size_w), dtype=np.float32)
        data_norm = np.zeros((crop_size_h, crop_size_w, 3), dtype=np.float32)
        depth_mask = np.zeros((crop_size_h, crop_size_w), dtype=np.float32)
        grid = np.zeros((crop_size_h, crop_size_w,3), dtype=np.float32)

    return data_img, data_depth, data_norm, data_mask,depth_mask,grid

# canny edge extractor
def myfunc_canny(img):
    img = np.squeeze(img)
    img = img + 128.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(img.shape())
    img = ((img-img.min())/(img.max()-img.min()))*255.0
    edges = cv2.Canny(img.astype('uint8'), 100, 220)
    edges = edges.astype(np.float32)
    edges = edges.reshape((1,crop_size_h,crop_size_w,1))
    edges = 1 - edges/255.0
    return edges

#edge aware refinement
def propagate(input_data,dlr,drl,dud,ddu,dim):
    if dim>1:
        dlr = tf.tile(dlr,[1,1,1,dim])
        drl = tf.tile(drl,[1,1,1,dim])
        dud = tf.tile(dud,[1,1,1,dim])
        ddu = tf.tile(ddu,[1,1,1,dim])
    x = tf.zeros((1,crop_size_h,1,dim),dtype = tf.float32)
    current_data = tf.concat([x,input_data],axis=2)
    current_data,_ = tf.split(current_data,[crop_size_w,-1],axis=2)
    output_data = tf.multiply(current_data,dlr) + tf.multiply(input_data,1-dlr)

    x = tf.zeros((1, crop_size_h, 1, dim), dtype=tf.float32)
    current_data = tf.concat([output_data,x], axis=2)
    _, current_data = tf.split(current_data, [-1, crop_size_w], axis=2)
    output_data = tf.multiply(current_data, drl) + tf.multiply(output_data, 1 - drl)

    x = tf.zeros((1, 1, crop_size_w, dim), dtype=tf.float32)
    current_data = tf.concat([x, output_data], axis=1)
    current_data, _ = tf.split(current_data, [crop_size_h, -1], axis=1)
    output_data = tf.multiply(current_data, dud) + tf.multiply(output_data, 1 - dud)

    x = tf.zeros((1, 1, crop_size_w, dim), dtype=tf.float32)
    current_data = tf.concat([output_data,x], axis=1)
    _, current_data = tf.split(current_data, [-1, crop_size_h], axis=1)
    output_data = tf.multiply(current_data, ddu) + tf.multiply(output_data, 1 - ddu)

    return output_data

class DEEPLAB(object):
    def __init__(self, fcn_ver=32):
        self.deeplab_ver = 'largeFOV'
        self.mean_BGR = [104.008, 116.669, 122.675]
        self.pretrain_weight = np.load('./initilization_model/model_denoise_depth_norm.npy',allow_pickle=True).tolist()

        self.crop_size = 320
        self.crop_size_h = 481
        self.crop_size_w = 641
        self.batch_size = 1
        self.max_steps = int(400000)
        self.train_dir = './trainmodel/'
        self.data_list = open('./list/traindata_grid.txt', 'rt').read().splitlines()
        self.starter_learning_rate = 1e-5
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.float32)
        self.end_learning_rate = 1e-6
        self.decay_steps = int(200000)
        self.k = 9
        self.rate = 4
        self.clip_norm = 20.0
        self.thresh = 0.95
        random.shuffle(self.data_list)

    def input_producer(self):
        def read_data():
            image, depth, norm,mask, depth_mask, grid= tf.py_func(myfunc,[self.data_queue[0]],[tf.float32,tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])
            image, depth, norm, mask, depth_mask, grid = preprocessing(image, depth, norm, mask,depth_mask,grid)
            return image, depth, norm, mask, depth_mask,grid

        # data loader + data augmentation
        def preprocessing(image, depth, norm, mask,depth_mask, grid):

            image = tf.cast(image, tf.float32)
            depth = tf.cast(depth, tf.float32)
            norm = tf.cast(norm, tf.float32)
            mask = tf.cast(mask, tf.float32)
            depth_mask = tf.cast(depth_mask, tf.float32)
            grid = tf.cast(grid, tf.float32)
            random_num = tf.random_uniform([], minval=0, maxval=1.0, dtype=tf.float32, seed=None, name=None)

            mirror_cond = tf.less(random_num, 0.5)
            stride = tf.where(mirror_cond, -1, 1)
            image = image[:, ::stride, :]
            depth = depth[:, ::stride]
            mask = mask[:, ::stride]
            depth_mask = depth_mask[:, ::stride]
            norm = norm[:, ::stride, :]
            norm_x, norm_y, norm_z = tf.split(value=norm, num_or_size_splits=3, axis=2)
            norm_x = tf.scalar_mul(tf.cast(stride, dtype=tf.float32), norm_x)
            norm = tf.cast(tf.concat([norm_x, norm_y, norm_z], 2), dtype=tf.float32)


            img_r, img_g, img_b = tf.split(value=image, num_or_size_splits=3, axis=2)
            image = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)

            image.set_shape((crop_size_h, crop_size_w, 3))
            depth.set_shape((crop_size_h, crop_size_w))
            norm.set_shape((crop_size_h, crop_size_w, 3))
            mask.set_shape((crop_size_h, crop_size_w))
            depth_mask.set_shape((crop_size_h, crop_size_w))
            grid.set_shape((crop_size_h,crop_size_w,3))
            return image, depth, norm, mask, depth_mask,grid

        with tf.variable_scope('input'):
            imglist = tf.convert_to_tensor(self.data_list, dtype=tf.string)
            self.data_queue = tf.train.slice_input_producer([imglist], capacity=100)
            images, depths,norms,masks,depth_masks, grid = read_data()
            batch_images, batch_depths, batch_norms, batch_masks,batch_depth_masks, grid = tf.train.batch([images, depths, norms, masks,depth_masks, grid], batch_size=self.batch_size, num_threads=4, capacity=60)
        return batch_images, batch_depths, batch_norms, batch_masks, batch_depth_masks, grid

    def forward(self,inputs, grid,is_training=True, reuse=False):
        def preprocessing(inputs):
            dims = inputs.get_shape()
            if len(dims) == 3:
                inputs = tf.expand_dims(inputs, dim=0)
            mean_BGR = tf.reshape(self.mean_BGR, [1, 1, 1, 3])
            inputs = inputs[:, :, :, ::-1] + mean_BGR
            return inputs

         ## -----------------------depth and normal FCN--------------------------
        inputs = preprocessing(inputs)
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.relu, stride=1,
                                padding='SAME',
                                weights_initializer=weight_from_caffe(self.pretrain_weight),
                                biases_initializer=bias_from_caffe(self.pretrain_weight)):

            with tf.variable_scope('fcn', reuse=reuse):
                ##---------------------vgg depth------------------------------------
                conv1 = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                pool1 = slim.max_pool2d(conv1, [3, 3], stride=2, padding='SAME', scope='pool1')

                conv2 = slim.repeat(pool1, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                pool2 = slim.max_pool2d(conv2, [3, 3], stride=2, padding='SAME', scope='pool2')

                conv3 = slim.repeat(pool2, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                pool3 = slim.max_pool2d(conv3, [3, 3], stride=2, padding='SAME', scope='pool3')

                conv4 = slim.repeat(pool3, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                pool4 = slim.max_pool2d(conv4, [3, 3], stride=1, padding='SAME', scope='pool4')

                conv5 = slim.repeat(pool4, 3, slim.conv2d, 512, [3, 3], rate=2, scope='conv5')
                pool5 = slim.max_pool2d(conv5, [3, 3], stride=1, padding='SAME', scope='pool5')
                pool5a = slim.avg_pool2d(pool5, [3, 3], stride=1, padding='SAME', scope='pool5a')

                fc6 = slim.conv2d(pool5a, 1024, [3, 3], stride=1, rate=12, scope='fc6')
                fc6 = slim.dropout(fc6, 0.5, is_training=is_training, scope='drop6')
                fc7 = slim.conv2d(fc6, 1024, [1, 1], scope='fc7')
                fc7 = slim.dropout(fc7, 0.5, is_training=is_training, scope='drop7')

                pool6_1x1 = slim.avg_pool2d(fc7, [61, 81], stride=[61, 81], padding='SAME', scope='pool6_1x1')
                pool6_1x1_norm = slim.unit_norm(pool6_1x1, dim=3, scope='pool6_1x1_norm_new')
                pool6_1x1_norm_scale = pool6_1x1_norm * 10
                pool6_1x1_norm_upsample = tf.tile(pool6_1x1_norm_scale, [1, 61, 81, 1], name='pool6_1x1_norm_upsample')

                out = tf.concat([fc7, pool6_1x1_norm_upsample], axis=-1, name='out')

                out_reduce = slim.conv2d(out, 256, [1, 1], activation_fn=tf.nn.relu, stride=1, scope='out_reduce',
                                         padding='SAME',
                                         weights_initializer=weight_from_caffe(self.pretrain_weight),
                                         biases_initializer=bias_from_caffe(self.pretrain_weight))
                out_conv = slim.conv2d(out_reduce, 256, [3, 3], activation_fn=tf.nn.relu, stride=1, scope='out_conv',
                                       padding='SAME',
                                       weights_initializer=weight_from_caffe(self.pretrain_weight),
                                       biases_initializer=bias_from_caffe(self.pretrain_weight))
                out_conv_increase = slim.conv2d(out_conv, 1024, [1, 1], activation_fn=tf.nn.relu, stride=1,
                                                scope='out_conv_increase',
                                                padding='SAME',
                                                weights_initializer=weight_from_caffe(self.pretrain_weight),
                                                biases_initializer=bias_from_caffe(self.pretrain_weight))

                fc8_nyu_depth = slim.conv2d(out_conv_increase, 1, [1, 1], activation_fn=None, scope='fc8_nyu_depth')
                fc8_upsample = tf.image.resize_images(fc8_nyu_depth, [self.crop_size_h, self.crop_size_w], method=0,
                                                      align_corners=True)
                #---------------------------------------vgg depth end ---------------------------------------
                ## ----------------- vgg norm---------------------------------------------------------------
                conv1_norm = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1_norm')
                pool1_norm = slim.max_pool2d(conv1_norm, [3, 3], stride=2, padding='SAME', scope='pool1_norm')

                conv2_norm = slim.repeat(pool1_norm, 2, slim.conv2d, 128, [3, 3], scope='conv2_norm')
                pool2_norm = slim.max_pool2d(conv2_norm, [3, 3], stride=2, padding='SAME', scope='pool2_norm')

                conv3_norm = slim.repeat(pool2_norm, 3, slim.conv2d, 256, [3, 3], scope='conv3_norm')
                pool3_norm = slim.max_pool2d(conv3_norm, [3, 3], stride=2, padding='SAME', scope='pool3_norm')

                conv4_norm = slim.repeat(pool3_norm, 3, slim.conv2d, 512, [3, 3], scope='conv4_norm')
                pool4_norm = slim.max_pool2d(conv4_norm, [3, 3], stride=1, padding='SAME', scope='pool4_norm')

                conv5_norm = slim.repeat(pool4_norm, 3, slim.conv2d, 512, [3, 3], rate=2, scope='conv5_norm')
                pool5_norm = slim.max_pool2d(conv5_norm, [3, 3], stride=1, padding='SAME', scope='pool5_norm')
                pool5a_norm = slim.avg_pool2d(pool5_norm, [3, 3], stride=1, padding='SAME', scope='pool5a_norm')

                fc6_norm = slim.conv2d(pool5a_norm, 1024, [3, 3], stride=1, rate=12, scope='fc6_norm')
                fc6_norm = slim.dropout(fc6_norm, 0.5, is_training=is_training, scope='drop6_norm')
                fc7_norm = slim.conv2d(fc6_norm, 1024, [1, 1], scope='fc7_norm')
                fc7_norm = slim.dropout(fc7_norm, 0.5, is_training=is_training, scope='drop7_norm')

                pool6_1x1_norm_new = slim.avg_pool2d(fc7_norm, [61, 81], stride=[61, 81], padding='SAME',
                                                     scope='pool6_1x1_norm_new')

                pool6_1x1_norm_norm = slim.unit_norm(pool6_1x1_norm_new, dim=3, scope='pool6_1x1_norm_new')
                pool6_1x1_norm_scale_norm = pool6_1x1_norm_norm * 10
                pool6_1x1_norm_upsample_norm = tf.tile(pool6_1x1_norm_scale_norm, [1, 61, 81, 1],
                                                       name='pool6_1x1_norm_upsample')
                out_norm = tf.concat([fc7_norm, pool6_1x1_norm_upsample_norm], axis=-1, name='out_norm')
                fc8_nyu_norm_norm = slim.conv2d(out_norm, 3, [1, 1], activation_fn=None, scope='fc8_nyu_norm_norm')
                fc8_upsample_norm = tf.image.resize_images(fc8_nyu_norm_norm, [self.crop_size_h, self.crop_size_w],
                                                           method=0, align_corners=True)

                fc8_upsample_norm = slim.unit_norm(fc8_upsample_norm, dim=3)
                #-------------------------------------vgg norm end---------------------------------------------


            # ------------- depth to normal + norm refinement---------------------------------------------------
            with tf.variable_scope('noise', reuse=reuse):

                fc8_upsample_norm = tf.squeeze(fc8_upsample_norm)
                fc8_upsample_norm = tf.reshape(fc8_upsample_norm,
                                               [self.batch_size, self.crop_size_h, self.crop_size_w, 3])

                norm_matrix = tf.extract_image_patches(images=fc8_upsample_norm, ksizes=[1, self.k, self.k, 1],
                                                       strides=[1, 1, 1, 1],
                                                       rates=[1, self.rate, self.rate, 1], padding='SAME')

                matrix_c = tf.reshape(norm_matrix,
                                      [self.batch_size, self.crop_size_h, self.crop_size_w, self.k * self.k, 3])

                fc8_upsample_norm = tf.expand_dims(fc8_upsample_norm, axis=4)

                angle = tf.matmul(matrix_c, fc8_upsample_norm)

                valid_condition = tf.greater(angle, self.thresh)
                valid_condition_all = tf.tile(valid_condition, [1, 1, 1, 1, 3])

                exp_depth = tf.exp(fc8_upsample * 0.69314718056)
                depth_repeat = tf.tile(exp_depth, [1, 1, 1, 3])
                points = tf.multiply(grid, depth_repeat)
                point_matrix = tf.extract_image_patches(images=points, ksizes=[1, self.k, self.k, 1],
                                                        strides=[1, 1, 1, 1],
                                                        rates=[1, self.rate, self.rate, 1], padding='SAME')

                matrix_a = tf.reshape(point_matrix, [self.batch_size, self.crop_size_h, self.crop_size_w, self.k * self.k, 3])

                matrix_a_zero = tf.zeros_like(matrix_a, dtype=tf.float32)
                matrix_a_valid = tf.where(valid_condition_all, matrix_a, matrix_a_zero)

                matrix_a_trans = tf.matrix_transpose(matrix_a_valid, name='matrix_transpose')
                matrix_b = tf.ones(shape=[self.batch_size, self.crop_size_h, self.crop_size_w, self.k * self.k, 1])
                point_multi = tf.matmul(matrix_a_trans, matrix_a_valid, name='matrix_multiplication')
                with tf.device('cpu:0'):
                    matrix_deter = tf.matrix_determinant(point_multi)
                inverse_condition = tf.greater(matrix_deter, 1e-5)
                inverse_condition = tf.expand_dims(inverse_condition, axis=3)
                inverse_condition = tf.expand_dims(inverse_condition, axis=4)
                inverse_condition_all = tf.tile(inverse_condition, [1, 1, 1, 3, 3])

                diag_constant = tf.ones([3], dtype=tf.float32)
                diag_element = tf.diag(diag_constant)
                diag_element = tf.expand_dims(diag_element, axis=0)
                diag_element = tf.expand_dims(diag_element, axis=0)
                diag_element = tf.expand_dims(diag_element, axis=0)


                diag_matrix = tf.tile(diag_element, [self.batch_size, self.crop_size_h, self.crop_size_w, 1, 1])

                inversible_matrix = tf.where(inverse_condition_all, point_multi, diag_matrix)
                with tf.device('cpu:0'):
                    inv_matrix = tf.matrix_inverse(inversible_matrix)

                generated_norm = tf.matmul(tf.matmul(inv_matrix, matrix_a_trans),matrix_b)

                norm_normalize = slim.unit_norm((generated_norm), dim=3)
                norm_normalize = tf.reshape(norm_normalize,
                                            [self.batch_size, self.crop_size_h, self.crop_size_w, 3])
                norm_scale = norm_normalize * 10.0


                conv1_noise = slim.repeat(norm_scale, 2, slim.conv2d, 64, [3, 3], scope='conv1_noise')
                pool1_noise = slim.max_pool2d(conv1_noise, [3, 3], stride=2, padding='SAME', scope='pool1_noise')  #

                conv2_noise = slim.repeat(pool1_noise, 2, slim.conv2d, 128, [3, 3], scope='conv2_noise')
                conv3_noise = slim.repeat(conv2_noise, 3, slim.conv2d, 256, [3, 3], scope='conv3_noise')

                fc1_noise = slim.conv2d(conv3_noise, 512, [1, 1], activation_fn=tf.nn.relu, stride=1,
                                        scope='fc1_noise',
                                        padding='SAME')
                encode_norm_noise = slim.conv2d(fc1_noise, 3, [3, 3], activation_fn=None, stride=1,
                                                scope='encode_norm_noise',
                                                padding='SAME')
                encode_norm_upsample_noise = tf.image.resize_images(encode_norm_noise,
                                                                    [self.crop_size_h, self.crop_size_w], method=0,
                                                                    align_corners=True)

                sum_norm_noise = tf.add(norm_normalize, encode_norm_upsample_noise)

                norm_pred_noise = slim.unit_norm(sum_norm_noise, dim=3)

                norm_pred_all = tf.concat([tf.expand_dims(tf.squeeze(fc8_upsample_norm),axis=0),norm_pred_noise,inputs*0.00392156862],axis = 3)

                norm_pred_all = slim.repeat(norm_pred_all, 3, slim.conv2d, 128, [3, 3],rate=2, weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                              biases_initializer=tf.constant_initializer(0.0),scope='conv1_norm_noise_new')
                norm_pred_all = slim.repeat(norm_pred_all, 3, slim.conv2d, 128, [3, 3],weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                              biases_initializer=tf.constant_initializer(0.0), scope='conv2_norm_noise_new')
                norm_pred_final = slim.conv2d(norm_pred_all, 3, [3, 3], activation_fn=None,
                                              weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                              biases_initializer=tf.constant_initializer(0.0), scope='norm_conv3_noise_new')
                norm_pred_final = slim.unit_norm((norm_pred_final), dim=3)


            # ------------- normal to depth  + depth refinement---------------------------------------------------
            with tf.variable_scope('norm_depth', reuse=reuse):
                 grid_patch = tf.extract_image_patches(images=grid, ksizes=[1, self.k, self.k, 1], strides=[1, 1, 1, 1],
                                                            rates=[1, self.rate, self.rate, 1], padding='SAME')
                 grid_patch = tf.reshape(grid_patch, [self.batch_size, self.crop_size_h, self.crop_size_w, self.k*self.k, 3])
                 _, _, depth_data = tf.split(value=matrix_a, num_or_size_splits=3, axis=4)
                 tmp_matrix_zero = tf.zeros_like(angle, dtype=tf.float32)
                 valid_angle = tf.where(valid_condition,angle,tmp_matrix_zero)


                 lower_matrix = tf.matmul(matrix_c,tf.expand_dims(grid,axis = 4))
                 condition = tf.greater(lower_matrix,1e-5)
                 tmp_matrix = tf.ones_like(lower_matrix)
                 lower_matrix = tf.where(condition,lower_matrix,tmp_matrix)
                 lower = tf.reciprocal(lower_matrix)
                 valid_angle = tf.where(condition,valid_angle,tmp_matrix_zero)
                 upper = tf.reduce_sum(tf.multiply(matrix_c,grid_patch),[4])
                 ratio = tf.multiply(lower,tf.expand_dims(upper,axis=4))
                 estimate_depth = tf.multiply(ratio,depth_data)


                 valid_angle = tf.multiply(valid_angle, tf.reciprocal(tf.tile(tf.reduce_sum(valid_angle,[3,4],keep_dims = True)+1e-5,[1,1,1,81,1])))

                 depth_stage1 = tf.reduce_sum(tf.multiply(estimate_depth, valid_angle), [3, 4])
                 depth_stage1 = tf.expand_dims(tf.squeeze(depth_stage1), axis=2)
                 depth_stage1 = tf.clip_by_value(depth_stage1, 0, 10.0)
                 exp_depth = tf.expand_dims(tf.squeeze(exp_depth), axis=2)

                 depth_all = tf.expand_dims(tf.concat([depth_stage1, exp_depth,tf.squeeze(inputs)*0.00392156862], axis=2), axis=0)

                 depth_pred_all = slim.repeat(depth_all, 3, slim.conv2d, 128, [3, 3], rate=2,weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                              biases_initializer=tf.constant_initializer(0.0),
                                             scope='conv1_depth_noise_new')
                 depth_pred_all = slim.repeat(depth_pred_all, 3, slim.conv2d, 128, [3, 3],weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                              biases_initializer=tf.constant_initializer(0.0), scope='conv2_depth_noise_new')
                 final_depth = slim.conv2d(depth_pred_all, 1, [3, 3], activation_fn=None,
                                               weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                               biases_initializer=tf.constant_initializer(0.0),
                                               scope='depth_conv3_noise_new')
            with tf.variable_scope('edge_refinemet', reuse=reuse):
                print(inputs.shape)
                edges = tf.py_func(myfunc_canny, [inputs], tf.float32)
                edges = tf.reshape(edges,[1,self.crop_size_h,self.crop_size_w,1])
                edge_input_depth = final_depth
                edge_input_norm = norm_pred_final

                #edge prediction for depth
                edge_inputs = tf.concat([edges,inputs*0.00784],axis=3)
                edges_encoder = slim.repeat(edge_inputs, 3, slim.conv2d, 32, [3, 3],rate = 2,weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                              biases_initializer=tf.constant_initializer(0.0),
                                             scope='conv1_edge_refinement')
                edges_encoder = slim.repeat(edges_encoder, 3, slim.conv2d, 32, [3, 3],
                                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                            biases_initializer=tf.constant_initializer(0.0),
                                            scope='conv2_edge_refinement')

                edges_predictor = slim.conv2d(edges_encoder, 8, [3, 3], activation_fn=None,
                                               weights_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                               biases_initializer=tf.constant_initializer(0.0),
                                               scope='edge_weight')
                edges_all = edges_predictor + tf.tile(edges,[1,1,1,8])
                edges_all = tf.clip_by_value(edges_all,0.0,1.0)

                dlr,drl,dud,ddu,nlr,nrl,nud,ndu = tf.split(edges_all,num_or_size_splits=8,axis=3)

                # 4 iteration depth
                final_depth = propagate(edge_input_depth,dlr,drl,dud,ddu,1)
                final_depth = propagate(final_depth, dlr, drl, dud, ddu, 1)
                final_depth = propagate(final_depth, dlr, drl, dud, ddu, 1)
                final_depth = propagate(final_depth, dlr, drl, dud, ddu, 1)

                # 4 iteration norm
                norm_pred_final = propagate(edge_input_norm, nlr, nrl, nud, ndu, 3)
                norm_pred_final = slim.unit_norm((norm_pred_final), dim=3)
                norm_pred_final = propagate(norm_pred_final, nlr, nrl, nud, ndu, 3)
                norm_pred_final = slim.unit_norm((norm_pred_final), dim=3)
                norm_pred_final = propagate(norm_pred_final, nlr,nrl, nud, ndu, 3)
                norm_pred_final = slim.unit_norm((norm_pred_final), dim=3)
                norm_pred_final = propagate(norm_pred_final, nlr, nrl, nud, ndu, 3)
                norm_pred_final = slim.unit_norm((norm_pred_final), dim=3)

        return final_depth,fc8_upsample_norm,norm_pred_final,fc8_upsample

    def train_op(self, loss):

        lr = tf.train.polynomial_decay(learning_rate = self.starter_learning_rate, global_step = self.global_step,
                                          decay_steps = self.decay_steps, end_learning_rate = self.end_learning_rate,
                                          power=0.9)

        print (self.decay_steps)
        print (self.end_learning_rate)
        opt = tf.train.AdamOptimizer(learning_rate = lr)
        grads_vars = opt.compute_gradients(loss)
        vars_name = []
        grads_vars_mult = []
        grads_value = []

        for grad, vars in grads_vars:

            t = 1
            if 'fcn' in vars.name:
                t = 0
            # t to 1 if finetune fcn part
            if 'edge_refinemet' in vars.name:
                t = 1

            grad *= t
            grads_vars_mult.append((grad, vars))
            vars_name.append(vars)
            grads_value.append(grad)
        global_norm_tmp = tf.global_norm(grads_value)
        if self.clip_norm > 0 :
            grads_value, _ = tf.clip_by_global_norm(grads_value, self.clip_norm)

        return opt.apply_gradients(zip(grads_value,vars_name), global_step = self.global_step),global_norm_tmp,lr

    def train(self):

        sess = tf.Session()
        self.sess = sess
        print(len(self.data_list))
        inputs, batch_depths, batch_norms, batch_masks,batch_depth_masks, batch_grids = self.input_producer()
        print(inputs)

        final_depth,fc8_upsample_norm, norm_pred_noise,fc8_upsample = self.forward(inputs,batch_grids)

        fc8_upsample = tf.squeeze(fc8_upsample)
        fc8_upsample_norm = tf.reshape(fc8_upsample_norm, [self.batch_size, self.crop_size_h, self.crop_size_w, 3])


        exp_depth = tf.exp(fc8_upsample * 0.69314718056)
        diff_square = tf.multiply(tf.abs(tf.subtract(exp_depth,batch_depths)),batch_depth_masks)
        loss1 = tf.reduce_sum(diff_square)/(tf.reduce_sum(batch_depth_masks)+1.0)

        final_depth = tf.squeeze(final_depth)
        diff_square_2 = tf.multiply(tf.abs(tf.subtract(final_depth, batch_depths)), batch_depth_masks)
        loss2 = tf.reduce_sum(diff_square_2) / (tf.reduce_sum(batch_depth_masks) + 1.0)

        batch_masks = tf.reshape(batch_masks,[self.batch_size,self.crop_size_h,self.crop_size_w,1])
        batch_masks = tf.tile(batch_masks,[1,1,1,3])
        diff_square_norm = tf.multiply(tf.abs(tf.subtract(fc8_upsample_norm, batch_norms)), batch_masks)
        diff_square_norm_noise = tf.multiply(tf.abs(tf.subtract(norm_pred_noise,batch_norms)), batch_masks)
        loss3 = tf.reduce_sum(diff_square_norm) / (tf.reduce_sum(batch_masks) + 1)
        loss4 = tf.reduce_sum(diff_square_norm_noise)/(tf.reduce_sum(batch_masks)+1)


        loss = loss2 + loss4 + 0*loss1 +0*loss3



        # train_op
        train_op,global_norm_tmp,lr = self.train_op(loss)
        #sum_grad = tf.

        sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=50, keep_checkpoint_every_n_hours=1)
        self.load()


        '''
        saver_a = tf.train.Saver([v for v in tf.trainable_variables()])
        saver_a.restore(sess,'./trainmodel/norm_refine_depth_denoise_conv_depth_complex_v3_edge/checkpoints/SRCNN.model-399999')
        '''
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        loss_matrix = np.zeros((self.max_steps+1,4),dtype=np.float32)



        for step in range(0,self.max_steps):

            start_time = time.time()
            _, loss_value,loss1_value,loss2_value,loss3_value,loss4_value,global_norm_tmp_value,lr_val,global_step_val = sess.run([train_op, loss, loss1, loss2, loss3,loss4,global_norm_tmp,lr,self.global_step])
            duration = time.time() - start_time


            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = self.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.5f, loss1 = %.5f, loss2 = %.5f, loss3 = %.5f, loss4 = %.5f lr = %.10f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print (global_norm_tmp_value)
                print (format_str % (datetime.now(), step, loss_value, loss1_value,loss2_value,loss3_value,loss4_value, lr_val,
                                     examples_per_sec, sec_per_batch))

            if step % 10000 == 0 or (step + 1) == self.max_steps:
                checkpoint_path = os.path.join(self.train_dir, 'checkpoints')
                self.save(checkpoint_path, step)
            loss_matrix[step, 0] = loss1_value
            loss_matrix[step, 1] = loss2_value
            loss_matrix[step, 2] = loss3_value
            loss_matrix[step, 3] = loss4_value

    def load(self, checkpoint_dir='checkpoints', step=None):
        print(" [*] Reading checkpoints...")

        checkpoint_dir = os.path.join(self.train_dir, checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def save(self, checkpoint_dir, step):
        model_name = "SRCNN.model"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    def test(self):
        inputs = tf.placeholder(shape=[1, crop_size_h, crop_size_w, 3], dtype=tf.float32)
        grid = tf.placeholder(shape=[1, crop_size_h, crop_size_w, 3], dtype=tf.float32)
        estimate_depth, fc8_upsample_norm, norm_pred_noise, fc8_upsample =  self.forward(inputs, grid, is_training=False)
        exp_depth  = tf.exp(fc8_upsample * 0.69314718056)

        sess = tf.Session()
        self.sess = sess
        sess.run(tf.global_variables_initializer())


        saver_a = tf.train.Saver([v for v in tf.trainable_variables()])
        saver_a.restore(sess, './trainmodel/checkpoints/SRCNN.model-399999')

        list = scipy.io.loadmat('./data/splits.mat')
        list = list['testNdxs']-1

        images = scipy.io.loadmat('./data/images_uint8.mat')
        images = images['images']
        images = images[:,:,:,list]

        grid_dic = scipy.io.loadmat('./data/grid.mat')
        grid_data = grid_dic['grid']
        grid_data = np.expand_dims(grid_data, axis=0)
        num = list.shape[0]
        depths_pred = np.zeros((crop_size_h, crop_size_w, num), dtype=np.float32)
        norms_pred = np.zeros((crop_size_h, crop_size_w, 3, num), dtype=np.float32)

        norms_pred_estimate = np.zeros((crop_size_h, crop_size_w, 3, num), dtype=np.float32)
        depths_pred_estimate = np.zeros((crop_size_h, crop_size_w, num), dtype=np.float32)
        input1 = np.zeros((1, crop_size_h, crop_size_w, 3), dtype=np.float32)

        for i in range(0, images.shape[3]):
            print(i)
            img_data = images[:, :, :, i]

            img_data = np.expand_dims(img_data, axis=0)
            img_data_r = img_data[0, :, :, 0] - 122.675 * 2
            img_data_g = img_data[0, :, :, 1] - 116.669 * 2
            img_data_b = img_data[0, :, :, 2] - 104.008 * 2

            input1[0, 0:crop_size_h-1, 0:crop_size_w-1, 0] = np.squeeze(img_data_r)
            input1[0, 0:crop_size_h-1, 0:crop_size_w-1, 1] = np.squeeze(img_data_g)
            input1[0, 0:crop_size_h-1, 0:crop_size_w-1, 2] = np.squeeze(img_data_b)

            original_depth, original_norm, refined_norm, refined_depth = sess.run(
                [exp_depth, fc8_upsample_norm, norm_pred_noise, estimate_depth],
                feed_dict={inputs: input1, grid: grid_data})
            depths_pred[:, :, i] = np.squeeze(original_depth)
            norms_pred[:, :, :, i] = np.squeeze(original_norm)
            norms_pred_estimate[:, :, :, i] = np.squeeze(refined_norm)
            depths_pred_estimate[:,:,i] = np.squeeze(refined_depth)

        #scipy.io.savemat(self.train_dir+'/depths_pred.mat',{'depths':depths_pred})
        #scipy.io.savemat(self.train_dir + '/norms_pred.mat', {'norms':norms_pred})
        scipy.io.savemat(self.train_dir+'/norms_estimate.mat', {'norms': norms_pred_estimate})
        scipy.io.savemat(self.train_dir + '/depths_estimate.mat', {'depths': depths_pred_estimate})


def main(_):
    model = DEEPLAB()
    if train_phase:
        model.train()
    else:
        model.test()


if __name__ == '__main__':
    tf.app.run()
