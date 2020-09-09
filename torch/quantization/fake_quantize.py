from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch.nn import Module
from .observer import MovingAverageMinMaxObserver, HistogramObserver, MovingAveragePerChannelMinMaxObserver, _with_args

class FakeQuantize(Module):
    r""" Simulate the quantize and dequantize operations in training time.
    The output of this module is given by

    x_out = (clamp(round(x/scale + zero_point), quant_min, quant_max)-zero_point)*scale



    * :attr:`scale` defines the scale factor used for quantization.

    * :attr:`zero_point` specifies the quantized value to which 0 in floating point maps to

    * :attr:`quant_min` specifies the minimum allowable quantized value.

    * :attr:`quant_max` specifies the maximum allowable quantized value.

    * :attr:`fake_quant_enable` controls the application of fake quantization on tensors, note that
      statistics can still be updated.

    * :attr:`observer_enable` controls statistics collection on tensors

    * :attr:`dtype` specifies the quantized dtype that is being emulated with fake-quantization,
                    allowable values are torch.qint8 and torch.quint8. The values of quant_min and
                    quant_max should be chosen to be consistent with the dtype


    Args:
        observer (module): Module for observing statistics on input tensors and calculating scale
                           and zero-point.
        quant_min (int): The minimum allowable quantized value.
        quant_max (int): The maximum allowable quantized value.
        observer_kwargs (optional): Arguments for the observer module

    Attributes:
        observer (Module): User provided module that collects statistics on the input tensor and
                           provides a method to calculate scale and zero-point.

    """
    # Flab by Y. Tamiya
    #def __init__(self, observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, **observer_kwargs):
    def __init__(self, observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, grad_observer=None, **observer_kwargs):
        super(FakeQuantize, self).__init__()
        assert quant_min <= quant_max, \
            'quant_min must be less than or equal to quant_max'
        self.quant_min = quant_min
        self.quant_max = quant_max
        # fake_quant_enabled and observer_enabled are buffers to support their
        # replication in DDP. Data type is uint8 because NCCL does not support
        # bool tensors.
        self.register_buffer('fake_quant_enabled', torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('observer_enabled', torch.tensor([1], dtype=torch.uint8))
        # Flab by Y. Tamiya
        grad_fpfmt_is_None = 'grad_fpfmt' in observer_kwargs \
                             and observer_kwargs['grad_fpfmt'] == None
        grad_fpfmt = observer_kwargs.pop('grad_fpfmt', None)
        if grad_observer or grad_fpfmt:
            grad_obs_kwargs = observer_kwargs.copy()
            if grad_fpfmt_is_None: # specified: grad_fpfmt=None
                grad_obs_kwargs.pop('fpfmt', None)
            if grad_fpfmt:
                grad_obs_kwargs['fpfmt'] = grad_fpfmt
            if grad_observer:
                self.grad_quant = grad_observer(**grad_obs_kwargs)
            else:
                self.grad_quant = observer(**grad_obs_kwargs)
            self.grad_quant_min = torch.iinfo(self.grad_quant.dtype).min
            self.grad_quant_max = torch.iinfo(self.grad_quant.dtype).max
            self.register_backward_hook(FakeQuantize.backward_hook)
        self.activation_post_process = observer(**observer_kwargs)
        assert torch.iinfo(self.activation_post_process.dtype).min <= quant_min, 'quant_min out of bound'
        assert quant_max <= torch.iinfo(self.activation_post_process.dtype).max, 'quant_max out of bound'
        self.register_buffer('scale', torch.tensor([1.0]))
        self.register_buffer('zero_point', torch.tensor([0]))
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = self.activation_post_process.ch_axis \
            if hasattr(self.activation_post_process, 'ch_axis') else -1

    @torch.jit.export
    def enable_fake_quant(self, enabled=True):
        # type: (bool) -> FakeQuantize
        self.fake_quant_enabled[0] = 1 if enabled else 0
        return self

    @torch.jit.export
    def disable_fake_quant(self):
        return self.enable_fake_quant(False)

    @torch.jit.export
    def enable_observer(self, enabled=True):
        # type: (bool) -> FakeQuantize
        self.observer_enabled[0] = 1 if enabled else 0
        return self

    @torch.jit.export
    def disable_observer(self):
        return self.enable_observer(False)

    @torch.jit.export
    def calculate_qparams(self):
        return self.activation_post_process.calculate_qparams()

    def forward(self, X):
        if self.observer_enabled[0] == 1:
            # Flab by Y. Tamiya
            if hasattr(self, 'dbg_level') \
               and (not hasattr(self, 'dbg_device') or self.dbg_device == X.device):
                if abs(self.dbg_level) == 1:
                    print(X.device, self.fullname, 'X', torch.min(X).item(), torch.max(X).item())
                elif abs(self.dbg_level) == 2:
                    print(X.device, self.fullname, 'X', X.cpu())
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            self.scale.resize_(_scale.shape)
            self.scale.copy_(_scale)
            self.zero_point.resize_(_zero_point.shape)
            self.zero_point.copy_(_zero_point)

        if self.fake_quant_enabled[0] == 1:
            #print('fake_quantize.forward: scale={}, zero_point={:x}'.format(self.scale, self.zero_point[0]))
            if self.qscheme == torch.per_channel_symmetric or self.qscheme == torch.per_channel_affine:
                X = torch.fake_quantize_per_channel_affine(X, self.scale, self.zero_point,
                                                           self.activation_post_process.ch_axis, self.quant_min, self.quant_max, self.training)
            else:
                X = torch.fake_quantize_per_tensor_affine(X, float(self.scale),
                                                          int(self.zero_point), self.quant_min,
                                                          self.quant_max, self.training)
            # Flab by Y. Tamiya
            if hasattr(self, 'dbg_level') \
               and (not hasattr(self, 'dbg_device') or self.dbg_device == X.device):
                if self.dbg_level == -1:
                    print(X.device, self.fullname, 'Y', torch.min(X).item(), torch.max(X).item())
                elif self.dbg_level == -2:
                    print(X.device, self.fullname, 'Y', X.cpu())
        return X

    # Flab by Y. Tamiya
    @staticmethod
    @torch.jit.export
    def backward_hook(self, dX, dY):
        assert not self.fake_quant_enabled or len(dY)==1, \
            'FakeQuantize with more than one inputs: {}'.format(len(dY))
        if self.observer_enabled:
            # Flab by Y. Tamiya
            if hasattr(self, 'dbg_level') \
               and (not hasattr(self, 'dbg_device') or self.dbg_device == dY[0].device):
                if abs(self.dbg_level) == 1:
                    print(dY[0].device, self.fullname, 'dY', torch.min(dY[0]).item(), torch.max(dY[0]).item())
                elif abs(self.dbg_level) == 2:
                    print(dY[0].device, self.fullname, 'dY', dY[0].cpu())
            self.grad_quant(dY[0])
            _scale, _zero_point = self.grad_quant.calculate_qparams()
            scale = _scale.to(self.scale.device)
            zero_point = _zero_point.to(self.zero_point.device, torch.int64)
            #print('fake_quantize.grad_hook: scale={}, zero_point={:x}'.format(self.scale, self.zero_point[0]))
        if self.fake_quant_enabled:
            if self.grad_quant.qscheme == torch.per_channel_symmetric \
               or self.grad_quant.qscheme == torch.per_channel_affine:
                dx = torch.fake_quantize_per_channel_affine(dY[0],
                                scale, zero_point,
                                self.grad_quant.ch_axis, self.grad_quant_min, self.grad_quant_max, self.training)
            else:
                dx = torch.fake_quantize_per_tensor_affine(dY[0],
                                float(scale), int(zero_point),
                                self.grad_quant_min, self.grad_quant_max, self.training)
            # Flab by Y. Tamiya
            if hasattr(self, 'dbg_level') \
               and (not hasattr(self, 'dbg_device') or self.dbg_device == dx.device):
                if self.dbg_level == -1:
                    print(dx.device, self.fullname, 'dX', torch.min(dx).item(), torch.max(dx).item())
                elif self.dbg_level == -2:
                    print(dx.device, self.fullname, 'dX', dx.cpu())
            return (dx,)

    with_args = classmethod(_with_args)

    @torch.jit.export
    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={},\
            scale={}, zero_point={}'.format(
            self.fake_quant_enabled, self.observer_enabled,
            self.scale, self.zero_point)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super(FakeQuantize, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = self.scale
        destination[prefix + 'zero_point'] = self.zero_point

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Removing this function throws an error that the the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                device = getattr(self, name).device
                setattr(self, name, val.to(device))
            elif strict:
                missing_keys.append(key)
        super(FakeQuantize, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                        missing_keys, unexpected_keys, error_msgs)

default_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255,
                                            dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=True)
default_weight_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=-128, quant_max=127,
                                                   dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False)

default_per_channel_weight_fake_quant = FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
                                                               quant_min=-128,
                                                               quant_max=127,
                                                               dtype=torch.qint8,
                                                               qscheme=torch.per_channel_symmetric,
                                                               reduce_range=False,
                                                               ch_axis=0)
default_histogram_fake_quant = FakeQuantize.with_args(observer=HistogramObserver,
                                                      quant_min=0,
                                                      quant_max=255,
                                                      dtype=torch.quint8,
                                                      qscheme=torch.per_tensor_affine,
                                                      reduce_range=True)
def disable_fake_quant(mod):
    if type(mod) == FakeQuantize:
        mod.disable_fake_quant()

def enable_fake_quant(mod):
    if type(mod) == FakeQuantize:
        mod.enable_fake_quant()

def disable_observer(mod):
    if type(mod) == FakeQuantize:
        mod.disable_observer()

def enable_observer(mod):
    if type(mod) == FakeQuantize:
        mod.enable_observer()

# Flab by Y. Tamiya
def debug_fake_quant(module, dbg_level=0, device=None):
    '''Set QAT debug level
    Call this func from the script or debugger like:
      torch.quantizatin.debug_fake_quant(model, 1, torch.device('cuda',0))
    dgb_level means:
     0: nop
     1: print min/max of Tensors before quantized.
    -1: print min/max of Tensors before & after quantized.
     2: print contents of Tensors before quantized.
    -2: print contents of Tensors before & after quantized.
    If device is given, print Tensors only on the specific device. 
    '''
    def set_fullname(mod, path_name=''):
        '''Set hierarchical names (fullnames) to submodules.'''
        for m in mod.named_children():
            fullname = path_name + '/' + m[0]
            m[1].fullname = fullname
            set_fullname(m[1], path_name=fullname) #recursive call

    # Set fullname to all submodules
    if dbg_level and not hasattr(module, 'fullname'):
        set_fullname(module)
    # print() will print all contents of Tensors
    if abs(dbg_level) == 2:
        torch.set_printoptions(profile='full')
    else:
        torch.set_printoptions(profile='default')
        
    # Set debug level to all FakeQuantize instances
    for m in module.modules():
        if type(m) == FakeQuantize:
            m.dbg_level = dbg_level
            if dbg_level:
                m.enable_observer()
            else:
                m.disable_observer()
            if device:
                m.dbg_device = device
