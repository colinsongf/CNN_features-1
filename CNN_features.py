__author__ = 'chensi'
import numpy as np
import sys
caffe_root = '/home/chensi/mylocal/sichen/caffe/'
sys.path.insert(0,caffe_root + 'python')
import caffe
import glob
imageset_path = ''
import cPickle
from optparse import OptionParser
import time
import scipy.io as sio
import os.path
import os
from scipy.sparse import csr_matrix

def initial_network(only_center):
    if only_center:
        net = caffe.Classifier(caffe_root+'examples/imagenet/imagenet_deploy_center.prototxt',
                           caffe_root+'examples/imagenet/caffe_reference_imagenet_model',
                           image_dims=[256,256])
    else:
        net = caffe.Classifier(caffe_root+'examples/imagenet/imagenet_deploy.prototxt',
                           caffe_root+'examples/imagenet/caffe_reference_imagenet_model',
                           image_dims=[256,256])

    net.set_phase_test()
    net.set_mode_gpu()
    net.set_mean('data',caffe_root+'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    net.set_channel_swap('data',(2,1,0))
    net.set_input_scale('data',255)
    return net


def get_options_parser():
    parser = OptionParser()
    parser.add_option('-i','--input_path',dest='img_input_path')
    parser.add_option('-l','--input_list',dest='img_input_list')
    parser.add_option('-c','--center',action='store_true',dest='only_center',default=False)
    parser.add_option('-o','--output',dest='feature_output_path',default=None)
    parser.add_option('--layer',dest='layer',default='conv5')
    parser.add_option('--location',action='store_true',dest='location',default=False)
    return parser

def calculate_location_table():
    print 'calculating locations for each layer...'
    centers = [[113,113],[113,142],[142,113],[142,142],[128,128]]
    centers = np.tile(centers,(2,1))
    #########################
    location_table1 = np.zeros((10,55,55,2))

    pix_pos1 = np.linspace(6,222,55)
    pix_pos1 = pix_pos1 - 114

    for i in range(5):
        sub_image_center = centers[i]
        for j in range(55):
            for k in range(55):
                location_table1[i,j,k,0] = sub_image_center[0]+pix_pos1[j]
                location_table1[i,j,k,1] = sub_image_center[1]+pix_pos1[k]
    for i in range(5,10):
        sub_image_center = centers[i]
        for j in range(55):
            for k in range(55):
                location_table1[i,j,k,0] = sub_image_center[0]+pix_pos1[j]
                location_table1[i,j,k,1] = sub_image_center[1]-pix_pos1[k]
    ############################
    location_table2 = np.zeros((10,27,27,2))
    pix_pos2 = np.linspace(10,218,27)
    pix_pos2 = pix_pos2 - 114

    for i in range(5):
        sub_image_center = centers[i]
        for j in range(27):
            for k in range(27):
                location_table2[i,j,k,0] = sub_image_center[0]+pix_pos2[j]
                location_table2[i,j,k,1] = sub_image_center[1]+pix_pos2[k]
    for i in range(5,10):
        sub_image_center = centers[i]
        for j in range(27):
            for k in range(27):
                location_table2[i,j,k,0] = sub_image_center[0]+pix_pos2[j]
                location_table2[i,j,k,1] = sub_image_center[1]-pix_pos2[k]
    ###############################
    location_table = np.zeros((10,13,13,2))
    pix_pos = np.linspace(18,210,13)
    pix_pos = pix_pos - 114

    for i in range(5):
        sub_image_center = centers[i]
        for j in range(13):
            for k in range(13):
                location_table[i,j,k,0] = sub_image_center[0]+pix_pos[j]
                location_table[i,j,k,1] = sub_image_center[1]+pix_pos[k]
    for i in range(5,10):
        sub_image_center = centers[i]
        for j in range(13):
            for k in range(13):
                location_table[i,j,k,0] = sub_image_center[0]+pix_pos[j]
                location_table[i,j,k,1] = sub_image_center[1]-pix_pos[k]
    ################################
    location_table_fc = np.zeros((10,1,1,2))
    pix_pos = np.array([0])

    for i in range(5):
        sub_image_center = centers[i]
        for j in range(1):
            for k in range(1):
                location_table_fc[i,j,k,0] = sub_image_center[0]+pix_pos[j]
                location_table_fc[i,j,k,1] = sub_image_center[1]+pix_pos[k]
    for i in range(5,10):
        sub_image_center = centers[i]
        for j in range(1):
            for k in range(1):
                location_table_fc[i,j,k,0] = sub_image_center[0]+pix_pos[j]
                location_table_fc[i,j,k,1] = sub_image_center[1]-pix_pos[k]

    print 'done'
    return {'conv1':location_table1,'conv2':location_table2,'conv3':location_table,'conv4':location_table,'conv5':location_table,
            'fc6':location_table_fc,'fc7':location_table_fc}


def save_sparse_file(csr_matrix,filename):
    data = csr_matrix.data
    rows, cols = csr_matrix.nonzero()
    f = open(filename,'w')
    
    
    for i in range(len(data)):
        f.write(str(rows[i]+1)+' ')
        f.write(str(cols[i]+1)+' ')
        f.write(str(data[i])+'\n')
        
    f.close()

    




def main():
    layer_list = ['conv1','conv2','conv3','conv4','conv5','fc6','fc7']
    parser = get_options_parser()

    (options, args) = parser.parse_args()
    if options.layer not in layer_list:
        raise NameError("Not valid layer name, layer name must be 'conv1','conv2','conv3','conv4','conv5','fc6','fc7'")

    with open(options.img_input_list,'r') as f:
        file_names = [options.img_input_path+line.strip() for line in f]
    with open(options.img_input_list,'r') as f:
	file_names_rel = [line.strip() for line in f]
    print 'total_image_detected: ' + str(len(file_names))
    if options.location:
        location_table = calculate_location_table()
    out_path = options.feature_output_path
    if not out_path:
	out_path = options.img_input_path
    file_name, fileExtension = os.path.splitext(file_names[0])

    layer = options.layer
    net = initial_network(options.only_center)

    if options.only_center:
        net.predict([caffe.io.load_image(file_names[0])],oversample=False)
    else:
        net.predict([caffe.io.load_image(file_names[0])])
    feature_temp = net.blobs[layer].data
    feature_shp = feature_temp.shape
    feature = np.zeros((feature_shp[0]*feature_shp[2]*feature_shp[3],feature_shp[1]))

    print 'feature_shape:',feature.shape
    
    if options.location:
        os.makedirs(out_path)
        loc_path = out_path+'/locations.dat'
        print 'saving location information to:' + loc_path
        locations = np.zeros((feature_shp[0]*feature_shp[2]*feature_shp[3],2))
        feature = np.zeros((feature_shp[0]*feature_shp[2]*feature_shp[3],feature_shp[1]))
        index = 0
        if feature_shp[0] == 1:
            for k in range(feature_shp[2]):
                for l in range(feature_shp[3]):
                    locations[index,0] = location_table[layer][4,k,l,0]
                    locations[index,1] = location_table[layer][4,k,l,1]
                    index += 1
        else:
            for j in range(feature_shp[0]):
                for k in range(feature_shp[2]):
                    for l in range(feature_shp[3]):
                        #feature[index,:] = feature_temp[j,:,k,l]
                        locations[index,0] = location_table[layer][j,k,l,0]
                        locations[index,1] = location_table[layer][j,k,l,1]
                        index += 1
        loc_file = open(loc_path,'w')
        for m in range(locations.shape[0]):
            loc_file.write(str(int(locations[m,0]))+' ')
            loc_file.write(str(int(locations[m,1]))+'\n')
        loc_file.close()
         
    start_time_total = time.time()
    for i in range(len(file_names)):
        start_time = time.time()
        print 'extracting the CNN feature, layer: %s, image No. %d/%d' %(layer,i+1,len(file_names))
        if options.only_center:
            net.predict([caffe.io.load_image(file_names[i])],oversample=False)
        else:
            net.predict([caffe.io.load_image(file_names[i])])
        feature_temp = net.blobs[layer].data
        feature_shp = feature_temp.shape
               
            

        feature = np.zeros((feature_shp[0]*feature_shp[2]*feature_shp[3],feature_shp[1]))
        index = 0
        for j in range(feature_shp[0]):
            for k in range(feature_shp[2]):
                for l in range(feature_shp[3]):
                    feature[index,:] = feature_temp[j,:,k,l]
                    index += 1
        feature = csr_matrix(feature)
        out_path1 = out_path+file_names_rel[i].replace(fileExtension,'_CNN_'+layer+'_feature'+'.dat')
        #np.savetxt(feature_path+str(i)+'.txt',feature,'%f',delimiter=',')
        try:
            #sio.savemat(out_path1,{'CNN_feature':feature})
            save_sparse_file(feature,out_path1)
        except:
            os.makedirs(os.path.dirname(out_path1))
            #sio.savemat(out_path1,{'CNN_feature':feature})
            save_sparse_file(feature,out_path1)

        print 'time used:'+str(time.time()-start_time)+'s'
    print 'time used totally:'+ str(time.time()-start_time_total)




if __name__ == '__main__':
    main()





