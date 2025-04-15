import json
import os
import sys
import importlib
import pickle
import yaml
from easydict import EasyDict as edict
from typing import Any, IO
import traceback
import torch

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

class TextLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        with open(self.log_path, "w") as f:
            f.write("")
    def log(self, log):
        with open(self.log_path, "a+") as f:
            f.write(log + "\n")

class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)

def construct_include(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""

    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        elif extension in ('json', ):
            return json.load(f)
        else:
            return ''.join(f.readlines())

def get_config(config_path):
    yaml.add_constructor('!include', construct_include, Loader)
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    config = edict(config)
    _, config_filename = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_filename)
    config.name = config_name
    return config

def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
def read_pkl(data_url):
    file = open(data_url,'rb')
    content = pickle.load(file)
    file.close()
    return content

def import_class(class_name):
    mod_str, _sep, class_str = class_name.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def import_function(func_name=None):
    """
    动态导入指定的函数。
    
    参数:
    - func_name: 一个字符串，表示函数的全限定名，如 "mymodule.my_function"
    
    返回:
    - 导入的函数对象
    """    
    # 分割模块名和函数名
    module_name, func_name = func_name.rsplit('.', 1)
    
    # 动态导入模块
    module = importlib.import_module(module_name)
    
    # 获取函数对象
    func = getattr(module, func_name)
    
    return func

def save_checkpoint(chk_path, epoch, lr, optimizer, model, eval_dict, no_print=False):
    if not no_print:
        print('\tSaving checkpoint to', chk_path)
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'eval_dict' : eval_dict
    }, chk_path)