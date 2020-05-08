from __future__ import absolute_import, division, print_function, unicode_literals
from collections import namedtuple
from .observer import *
from .fake_quantize import *
import torch.nn as nn

class QConfig(namedtuple('QConfig', ['activation', 'weight'])):
    """
    Describes how to quantize a layer or a part of the network by providing
    settings (observer classes) for activations and weights respectively.


    Note that QConfig needs to contain observer **classes** (like MinMaxObserver) or a callable that returns
    instances on invocation, not the concrete observer instances themselves.
    Quantization preparation function will instantiate observers multiple times for each of the layers.


    Observer classes have usually reasonable default arguments, but they can be overwritten with `with_args`
    method (that behaves like functools.partial):

      my_qconfig = QConfig(activation=MinMaxObserver.with_args(dtype=torch.qint8), 
      weight=default_observer.with_args(dtype=torch.qint8))
    """
    def __new__(cls, activation, weight):
        # catch common mistakes
        if isinstance(activation, nn.Module) or isinstance(weight, nn.Module):
            raise ValueError("QConfig received observer instance, please pass observer class instead. " +
                             "Use MyObserver.with_args(x=1) to override arguments to constructor if needed")
        return super(QConfig, cls).__new__(cls, activation, weight)


default_qconfig = QConfig(activation=default_observer,
                          weight=default_weight_observer)

default_debug_qconfig = QConfig(weight=default_weight_observer,
                                activation=default_debug_observer)

default_per_channel_qconfig = QConfig(activation=default_observer,
                                      weight=default_per_channel_weight_observer)

class QConfigDynamic(namedtuple('QConfigDynamic', ['weight'])):
    """
    Describes how to dynamically quantize a layer or a part of the network by providing
    settings (observer classe) for weights.

    It's like QConfig, but for dynamic quantization.

    Note that QConfigDynamic needs to contain observer **classes** (like MinMaxObserver) or a callable that returns
    instances on invocation, not the concrete observer instances themselves.
    Quantization function will instantiate observers multiple times for each of the layers.

    Observer classes have usually reasonable default arguments, but they can be overwritten with `with_args`
    method (that behaves like functools.partial):

      my_qconfig = QConfigDynamic(weight=default_observer.with_args(dtype=torch.qint8))
    """
    def __new__(cls, weight):
        # catch common mistakes
        if isinstance(weight, nn.Module):
            raise ValueError("QConfigDynamic received observer instance, please pass observer class instead. " +
                             "Use MyObserver.with_args(x=1) to override arguments to constructor if needed")
        return super(QConfigDynamic, cls).__new__(cls, weight)

default_dynamic_qconfig = QConfigDynamic(weight=default_weight_observer)
float16_dynamic_qconfig = QConfigDynamic(weight=NoopObserver.with_args(dtype=torch.float16))
per_channel_dynamic_qconfig = QConfigDynamic(weight=default_per_channel_weight_observer)

default_qat_qconfig = QConfig(activation=default_fake_quant,
                              weight=default_weight_fake_quant)

default_weight_only_qconfig = QConfig(activation=torch.nn.Identity,
                                      weight=default_weight_fake_quant)
default_activation_only_qconfig = QConfig(activation=default_fake_quant,
                                          weight=torch.nn.Identity)

def get_default_qconfig(backend='fbgemm'):
    if backend == 'fbgemm':
        qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=True),
                          weight=default_per_channel_weight_observer)
    elif backend == 'qnnpack':
        qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=False),
                          weight=default_weight_observer)
    else:
        raise ValueError("Unknown backend, please specify qconfig manually")
    return qconfig

# Flab by Y. Tamiya
#def get_default_qat_qconfig(backend='fbgemm'):
def get_default_qat_qconfig(backend='fbgemm', grad_observer=None):
    # Histogram observer is too slow for quantization aware training
    if backend == 'fbgemm':
        qconfig = QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                            quant_min=0,
                                                            quant_max=255,
                                                            grad_observer=grad_observer, #Flab by Y.Tamiya
                                                            reduce_range=True),
                          # Flab by Y. Tamiya
                          #weight=default_per_channel_weight_fake_quant)
                          weight=default_per_channel_weight_fake_quant.with_args(grad_observer=grad_observer))
    elif backend == 'qnnpack':
        qconfig = QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                            quant_min=0,
                                                            quant_max=255,
                                                            grad_observer=grad_observer, #Flab by Y.Tamiya
                                                            reduce_range=False),
                          # Flab by Y. Tamiya
                          #weight=default_weight_fake_quant)
                          weight=default_weight_fake_quant.with_args(grad_observer=grad_observer))
    else:
        raise ValueError("Unknown backend, please specify qconfig manually")

    return qconfig

# Added by Flab (Y. Tamiya) #
def get_flexfp_qat_qconfig(fpfmt, grad_fpfmt=None):
    observer = (FlexFpDynBiasObserver if len(fpfmt) < 3 else
                FlexFpObserver)
    return QConfig(activation=FakeQuantize.with_args(observer=observer,
                         quant_min=torch.iinfo(torch.int32).min,
                         quant_max=torch.iinfo(torch.int32).max,
                         dtype=torch.int32,
                         fpfmt=fpfmt, grad_fpfmt=grad_fpfmt),
                   weight=FakeQuantize.with_args(observer=observer,
                         quant_min=torch.iinfo(torch.int32).min,
                         quant_max=torch.iinfo(torch.int32).max,
                         dtype=torch.int32,
                         fpfmt=fpfmt, grad_fpfmt=grad_fpfmt))

# Added by Flab (Y. Tamiya) #
def get_flexfp_dynbias_qat_qconfig(fpfmt, grad_fpfmt=None):
    '''Obsoleted by get_flexfp_qat_qconfig()'''
    return get_flexfp_qat_qconfig(fpfmt, grad_fpfmt)

# Added by Flab (Y. Tamiya) #
def get_qint_grad_flexfp_qat_qconfig(grad_fpfmt):
    grad_observer = (FlexFpDynBiasObserver if len(grad_fpfmt) < 3 else
                     FlexFpObserver)
    return QConfig(activation=FakeQuantize.with_args(
                       observer=MovingAverageMinMaxObserver,
                       quant_min=0,
                       quant_max=255,
                       dtype=torch.quint8,
                       grad_observer=grad_observer,
                       grad_fpfmt=grad_fpfmt),
                   weight=FakeQuantize.with_args(
                       observer=MovingAverageMinMaxObserver,
                       quant_min=-128,
                       quant_max= 127,
                       dtype=torch.qint8,
                       grad_observer=grad_observer,
                       grad_fpfmt=grad_fpfmt))
