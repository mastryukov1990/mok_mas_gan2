import torch
import numpy as np
import os
from pdb import set_trace as st

class BaseModel:
    def __init__(self):
        pass

    def name(self):
        return 'BaseModel'

    def initializer(self, use_gpu=True, gpu_ids_list=None):
        if gpu_ids_list is None:
            gpu_ids_list = [0]
        self.use_gpu = use_gpu
        self.gpu_ids_list = gpu_ids_list

    def forward(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_params(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    def save_network(self, network, path, network_label, epoch_label):
        save_name = '%s_net%s.pth' % (epoch_label, network_label)
        sv_path = os.path.join(path, save_name)
        torch.save(network.state_dict(), sv_path)

    def load_network(self, network, network_label, epoch_label):
        save_name = '%s_net_%s.pth' % (epoch_label, network_label)
        sv_path = os.path.join(self.save_dir, save_name)
        print('Loading network from %s' % sv_path)
        network.load_state_dict(torch.load(sv_path))

    def update_lrn_rate(self):
        pass

    def get_image_paths(self):
        return self.image_paths

    def save_done(self, flag=False):
        np.save(os.path.join(self.save_dir, 'done_flag'), flag)
        np.save(os.path.join(self.save_dir, 'done_flag'), [flag, ], fmt='%i')
