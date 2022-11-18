import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy
import datetime
import os
import os.path as osp
import pickle
import torch
import numpy as np
import scipy.io as sio

BEST = "best_model"
LAST = "last_model"

LOGDIR = osp.join(osp.split(osp.dirname(osp.realpath(__file__)))[0], "logs")
if not osp.exists(LOGDIR):
    os.mkdir(LOGDIR)

class Logger(object):
    '''
    This logger will print reported properties and log properties into files and
    '''
    BEST = BEST
    LAST = LAST

    def __init__(self, printstr=[], save_every=100, dir_name=None, dir_path_with_time=True):

        self.dir_path = str()
        self.data_dict = {}
        self.save_every = save_every
        self.log_count = 0
        self.printstr = printstr


        if dir_name is None:
            dir_name = datetime.datetime.now().strftime("LogRun_%Y-%m-%d_%H-%M-%S")
        else:
            time = datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S") if dir_path_with_time else ""
            dir_name = dir_name + time

        self.dir_path = osp.join(LOGDIR, dir_name)
        if not osp.exists(self.dir_path):
            os.mkdir(self.dir_path)

        print("Start Logging to {}".format(self.dir_path))
        print(("LogCount:{}, " + printstr[0].replace("{:.2f}", "{}")).format(self.log_count, *printstr[1:]))


    def log(self, log_dict=dict(), model=None, id=None):
        """
        do the logging
        :param log_dict: the dict of data {"valid_acc": value, ...}
        :param model: optional pytorch model to log to
        :param id: the best model is kept by the best value from the key in the data dict
        :return:
        """
        # log things
        for k in log_dict:
            if k not in self.data_dict:
                self.data_dict[k] = []
            v = log_dict[k]
            self.data_dict[k].append(v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v)
        self.log_count += 1

        # log model
        if model != None:
            if id != None:
                assert id in self.data_dict
                cond = lambda d: d[id][-1] > np.max(d[id][:-1]) if len(d[id]) > 1 else True
            else:
                cond = lambda d: False
            self.log_model(model, cond)

        # save everything and make plots
        if self.log_count % self.save_every == 0:

            # save profile
            self.save_profile(self.dir_path)

            # save dict separately
            self.save_data_dict(self.dir_path)

            # save matlab mat file containing all from the best state dict
            self.save_matlab(self.dir_path)

            # plots
            for k in self.data_dict:
                if isinstance(self.data_dict[k], dict):
                    continue
                # fig = plt.figure(figsize=(4, 5))
                plt.plot(self.data_dict[k])
                plt.title(k)
                plt.xlabel('batches')
                plt.savefig(osp.join(self.dir_path, "{}.png".format(k)))
                plt.close()

        print(("LogCount:{}, " + self.printstr[0]).format(self.log_count, *[log_dict[k] for k in self.printstr[1:]]))

    def log_model(self, model, condition):
        """
        save model to data dict "last model".
        if condition function returns True, also save to "best model"
        :param model: pytorch model
        :param condition: a function of data_dict, return bool
        :return: condition function result
        """
        c = condition(self.data_dict)
        self.data_dict[LAST] = model.state_dict()
        if c:
            state = copy.deepcopy(model.state_dict())
            self.data_dict[BEST] = state
        return c

    def load_model(self, model, id=LAST):
        """
        load model from data dict. If loading from a directory, make sure load logger profile first
        :param model: pytorch model
        :param id: which model do you want? BEST or LAST
        :return: the loaded model
        """
        model.load_state_dict(self.data_dict[id])
        return model

    def save_profile(self, dir):
        """
        save the profile.pkl from dir. It should be a Logger object
        :param dir: the dir to save the log
        :return: None
        """
        filename = "profile.pkl"
        with open(osp.join(dir, filename), "wb") as f:
            pickle.dump(self, f)

    def save_data_dict(self, dir):
        with open(osp.join(dir, 'dict.pkl'), 'wb') as f:
            pickle.dump(self.data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            # load with
            # with open('filename.pkl', 'rb') as f:
            #    b = pickle.load(f)

    def save_matlab(self, dir):
        if BEST in self.data_dict:
            d = copy.deepcopy(self.data_dict[BEST])
            for k in d:
                d[k] = d[k].detach().cpu().numpy()
            sio.savemat(osp.join(dir, "weights.mat"), d)

    @staticmethod
    def load_profile(dir):
        """
        load the profile.pkl from dir. It should be a Logger object
        :param dir: the dir to load the log
        :return: the loaded Logger
        """
        filename = "profile.pkl"
        with open(osp.join(dir, filename), "rb") as f:
            logger = pickle.load(f)
        return logger
