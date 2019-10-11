from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import scipy.io
from IMLib.utils import *
from abc import abstractmethod
import PyLib.const as const

class Dataset(object):

    def __init__(self, config):
        self._data_dir = config.data_dir
        self._image_size = config.image_size
        self._batch_size = config.batchSize
        self._channel = config.input_nc
        self._num_source_class = config.num_source_class

    @property
    def data_dir(self):
        return self._data_dir

    @data_dir.setter
    def data_dir(self, value):
        self._data_dir = value

    @property
    def image_size(self):
        return self._image_size

    @image_size.setter
    def image_size(self, image_size):
        self._image_size = image_size

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, channel):
        self._channel = channel

    @property
    def num_source_class(self):
        return self._num_source_class

    @num_source_class.setter
    def num_source_class(self, num_source_class):
        self._num_source_class = num_source_class

    @abstractmethod
    def read_image_dict(self):
        pass

    @abstractmethod
    def getNextBatch(self):
        pass
    
    def getShapeForData(self, filenames):
        array = [get_image(batch_file, crop_size=self._image_size, is_crop=True, resize_w=140,
                           is_grayscale=False) for batch_file in filenames]
        sample_images = np.array(array)
        return sample_images

class Flowers(Dataset):
    #link: http://www.robots.ox.ac.uk/~vgg/data/flowers/
    const.dataname1 = "Flowers"
    def __init__(self, config):
        super().__init__(config)
        self.images_dict = self.read_image_dict()

    def read_image_dict(self):

        images_dict = dict()
        label_mat = scipy.io.loadmat(os.path.join(self._data_dir, 'imagelabels.mat'))
        label_id = label_mat.get('labels')[0]
        img_folder = 'jpg'
        img_id = 1
        for f in range(len(label_id)):
            format_id_x = 'image_%05d.jpg' % img_id
            img_label = label_id[img_id - 1]
            if img_label not in images_dict:
                images_dict[img_label] = []
            if os.path.exists(os.path.join(self._data_dir, img_folder+'/'+ format_id_x)):
                images_dict[img_label].append(os.path.join(self._data_dir, img_folder+'/'+ format_id_x))
            img_id += 1

        return images_dict

    def getNextBatch(self):

        source_image_x = []
        target_image_y1 = []
        target_image_y2 = []
        x_cls = []
        y_cls = []

        for i in range(self.batch_size):

            # 1-70 for train; 70-102 for tesst
            id_domain = range(1, self._num_source_class)
            id_x, id_y = random.sample(id_domain, 2)
            x_cls.append(id_x)
            y_cls.append(id_y)
            format_id_x = id_x  # source class
            format_id_y = id_y  # target class

            #print self.images_dict[format_id_x]
            source_image_name = random.sample(self.images_dict[format_id_x], 1)[0]
            the_path = os.path.join(self.data_dir, source_image_name)
            source_image_x.append(the_path)
            target_image_path_1, target_image_path_2 = random.sample(self.images_dict[format_id_y], 2)
            target_image_y1.append(target_image_path_1)
            target_image_y2.append(target_image_path_2)

        return np.asarray(source_image_x), np.asarray(target_image_y1), np.asarray(x_cls) - 1, np.asarray(y_cls) - 1

class CUB_bird(Dataset):
    #CUB bird: http://www.vision.caltech.edu/visipedia/CUB-200.html
    const.dataname2 = "CUB_200_2011"
    def __init__(self, config):
        super().__init__(config)
        self.images_dict = self.read_image_dict()

    def read_image_dict(self):

        #fh = open(self.data_dir + "/eye_position_2.txt")
        fh = open('images.txt')
        images_dict = dict()

        for f in fh.readlines():
            f = f.strip('\n').split(' ')[-1]
            info = f.split('/')
            image_folder = info[0]
            image_folder_num = info[0].split('.')[0]
            imagename = info[1]
            if not images_dict.has_key(image_folder_num):
                images_dict[image_folder_num] = []
            if os.path.exists(os.path.join(self.data_dir, image_folder+'/'+imagename)):
                images_dict[image_folder_num] .append(image_folder+'/'+imagename)

        fh.close()
        #print(images_dict)
        #print(len(images_dict))# 200
        return images_dict

    def getNextBatch(self):

        source_image_x = []
        target_image_y1 = []
        target_image_y2 = []
        x_cls = []
        y_cls = []

        for i in range(self.batch_size):

            # using 1-170 for train , 170-200 for test
            id_domain = range(1, self._num_source_class)
            id_x, id_y = random.sample(id_domain, 2) - 1
            x_cls.append(id_x)
            y_cls.append(id_y)
            format_id_x = '%03d' % id_x  # source class
            format_id_y = '%03d' % id_y  # target class
            #print self.images_dict[format_id_x]
            source_image_name = random.sample(self.images_dict[format_id_x], 1)[0]
            the_path = os.path.join(self.data_dir, source_image_name)
            source_image_x.append(the_path)
            target_image_name_1, target_image_name_2 = random.sample(self.images_dict[format_id_y], 2)
            the_path_1 = os.path.join(self.data_dir,target_image_name_1)
            the_path_2 = os.path.join(self.data_dir, target_image_name_2)
            target_image_y1.append(the_path_1)
            target_image_y2.append(the_path_2)

        return np.asarray(source_image_x), np.asarray(target_image_y1), \
                    np.asarray(target_image_y2), np.asarray(x_cls), np.asarray(y_cls)

class Animals(Dataset):
    #ImageNet ILSVRC2012 : http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar
    const.dataname3 = "Animals"
    def __init__(self, config):
        super().__init__(config)
        self.images_dict = self.read_image_dict()
        self.keys_list = list(self.images_dict.keys())

    def read_image_dict(self):

        fh = open('./datasets/animals_list_train.txt')
        images_dict = dict()

        for f in fh.readlines():
            info = f.strip('\n').split('/')
            image_folder = info[0]
            imagename = info[1]
            if not images_dict.has_key(image_folder):
                images_dict[image_folder] = []
            if os.path.exists(os.path.join(self.data_dir, image_folder+'/'+imagename)):
                images_dict[image_folder].append(image_folder+'/'+imagename)

        fh.close()
        return images_dict

    def getNextBatch(self):

        source_image_x = []
        target_image_y1 = []
        x_cls = []
        y_cls = []

        for i in range(self.batch_size):

            # using 1-170 for train , 170-200 for test
            id_domain = range(0, len(self.images_dict))
            id_x, id_y = random.sample(id_domain, 2)
            x_cls.append(id_x)
            y_cls.append(id_y)
            #print self.images_dict[format_id_x]
            source_image_name = random.sample(self.images_dict[self.keys_list[id_x]], 1)[0]
            the_path = os.path.join(self.data_dir, source_image_name)
            source_image_x.append(the_path)
            target_image_name_1 = random.sample(self.images_dict[self.keys_list[id_y]], 1)[0]
            the_path_1 = os.path.join(self.data_dir, target_image_name_1)
            target_image_y1.append(the_path_1)

        return np.asarray(source_image_x), np.asarray(target_image_y1), np.asarray(x_cls), np.asarray(y_cls)





