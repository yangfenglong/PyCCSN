#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Yangfenglong
# @Date: 2019-05-20
"""
set of useful functions
usage: import useful_functions as uf
       uf.create_dir(dir)
"""

import os
import hashlib
from datetime import datetime
import logging
import traceback
from functools import wraps

def create_dir(dir):
    if not os.path.exists(dir):
        assert not os.system('mkdir {}'.format(dir))

def get_md5(file):
    md5file = open(file,'rb')
    md5 = hashlib.md5(md5file.read()).hexdigest()
    md5file.close()
    return md5

def now():
    fmt = "%Y%m%d%H%M%S"
    return datetime.now().strftime(fmt)
    
def create_logger(logFile_name):
    """同时输出log到文件和屏幕"""
    logger = logging.getLogger()  # 不加名称设置root logger
    logger.setLevel(logging.INFO)

    sh = logging.StreamHandler() # 使用StreamHandler输出到屏幕
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(filename)s:%(lineno)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')    
    sh.setFormatter(formatter)
    logger.addHandler(sh) # 添加screen Handler

    fh = logging.FileHandler(logFile_name) # 使用FileHandler输出到文件
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh) #添加file Handle
    return logger

def robust(actual_do):
    """try_except 装饰器"""
    #print('running decorate', actual_do)
    @wraps(actual_do) #把原函数的元信息拷贝到装饰器函数中
    def add_robust(*args, **keyargs):
        try:
            return actual_do(*args, **keyargs)
        except:
            print ('Error execute: {}'.format(actual_do.__name__))
            traceback.print_exc()
    return add_robust
