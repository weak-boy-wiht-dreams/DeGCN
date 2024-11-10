import numpy as np;
from .feeder_uav import Feeder;
from feeders import tools;

class FeederUAVHuman(Feeder):
    #构造函数初始化
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False 
                 ):  
        #初始化父类
        super().__init__(data_path, label_path, 
                        p_interval, split, 
                        random_choose, random_shift, 
                        random_move, random_rot, 
                        window_size, normalization, 
                        debug, use_mmap, 
                        bone, vel);
        self.load_data();
        if normalization:
            self.get_mean_map()

    # 加载数据的方法
    def load_data(self):
        # 两个都加载, 按照不同阶段返回东西就好
        # data和label可以代表训练集的，也可以代表测试集的
        data = np.load(self.data_path);
        label = np.load(self.label_path);
        # 这里sample_name需要一点判断，还是写上好了
        data_type_name = 'test_' if self.split == 'test' else 'train_';

        # 大概看了一下，还是按照一起加载的逻辑来写比较好(先暂时这样), 不然其他地方也要改
        if not self.debug:
            self.data = data;
            self.label = label;
            self.sample_name = [data_type_name + str(i) for i in range(len(self.data))];  #还是给一个sample_name吧
        else:
            self.data = data[0:100];
            self.label = label[0:100];
            self.sample_name = [data_type_name + str(i) for i in range(100)];

        # N, T, _ = self.data.shape
        # self.data = self.data.reshape((N, T, 2, 17, 3)).transpose(0, 4, 1, 3, 2)
        #N,C,T,V,M <-我们的
        #N,M,C,V,T <-原来的
        # self.data = self.data.transpose(0, 4, 1, 3, 2)

    def __getitem__(self, index):
        C, T, V, M = self.data[1].shape  #获取到每个样本数据的shape
        data_numpy = self.data[index]    #获取到一个样本
        #print(f"xshape: data_numpy shape:{data_numpy.shape}")
        label = self.label[index]        #获取到对应的label
        data_numpy = np.array(data_numpy)        #将数据转换为numpy兼容格式
        if not(np.any(data_numpy)):
            data_numpy = np.array(self.data[0])
            
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0) #和下一个函数冲突，全0会被报错
        # reshape Tx(MVC) to CTVM

        if(valid_frame_num == 0):
            data_numpy = np.zeros((2,64,17,300));

        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import coco_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in coco_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        #print(f"xshape: getitem: {data_numpy.shape}");
        return data_numpy, label, index
