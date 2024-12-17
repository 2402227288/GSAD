import logging
logger = logging.getLogger('base') ##获取名为base的日志


def create_model(opt):  ##实例化模型并输出创建日志
    from .model import DDPM as M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
