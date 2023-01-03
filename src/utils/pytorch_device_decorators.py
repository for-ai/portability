from functools import wraps
import unittest

NATIVE_DEVICES = {'cpu', 'cuda', 'xla'}
ACCELERATED_DEVICES = {'cuda', 'xla'}


def onlyAcceleratedDeviceTypes(fn):
    @wraps(fn)
    def only_fn(self, *args, **kwargs):
        if self.device_type not in ACCELERATED_DEVICES:
            reason = "onlyAcceleratedDeviceTypes: doesn't run on {0}".format(
                self.device_type)
            raise unittest.SkipTest(reason)

        return fn(self, *args, **kwargs)

    return only_fn


def onlyNativeDeviceTypes(fn):
    @wraps(fn)
    def only_fn(self, *args, **kwargs):
        if self.device_type not in NATIVE_DEVICES:
            reason = "onlyNativeDeviceTypes: doesn't run on {0}".format(
                self.device_type)
            raise unittest.SkipTest(reason)

        return fn(self, *args, **kwargs)

    return only_fn
