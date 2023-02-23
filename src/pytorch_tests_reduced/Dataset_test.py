import torch
import unittest
from torch.utils.data import (
    DataLoader,
    Dataset,
)
from torch.testing._internal.common_utils import (TestCase, run_tests)


from torch.testing._internal.common_device_type import onlyCUDA
from ..utils.pytorch_device_decorators import onlyAcceleratedDeviceTypes, instantiate_device_type_tests
from ..utils.timer_wrapper import pytorch_op_timer


class DictDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, ndx):
        return {
            'a_tensor': torch.empty(4, 2).fill_(ndx),
            'another_dict': {
                'a_number': ndx,
            },
        }


class TestDictDataLoader(TestCase):
    def setUp(self):
        with pytorch_op_timer():
            self.dataset = DictDataset()

    def test_sequential_batch(self):
        for persistent_workers in (False, True):
            if persistent_workers:
                loader = DataLoader(self.dataset, batch_size=2, shuffle=False,
                                    persistent_workers=persistent_workers, num_workers=1)
            else:
                loader = DataLoader(self.dataset, batch_size=2, shuffle=False,
                                    persistent_workers=persistent_workers)
            batch_size = loader.batch_size
            for i, sample in enumerate(loader):
                idx = i * batch_size
                self.assertEqual(set(sample.keys()), {
                                 'a_tensor', 'another_dict'})
                self.assertEqual(
                    set(sample['another_dict'].keys()), {'a_number'})

                t = sample['a_tensor']
                self.assertEqual(t.size(), torch.Size([batch_size, 4, 2]))
                self.assertTrue((t[0] == idx).all())
                self.assertTrue((t[1] == idx + 1).all())

                n = sample['another_dict']['a_number']
                self.assertEqual(n.size(), torch.Size([batch_size]))
                self.assertEqual(n[0], idx)
                self.assertEqual(n[1], idx + 1)

    def test_pin_memory(self, device):
        loader = DataLoader(self.dataset, batch_size=2, pin_memory=True)
        for sample in loader:
            self.assertTrue(sample['a_tensor'].is_pinned())
            self.assertTrue(sample['another_dict']['a_number'].is_pinned())

    def test_pin_memory_device(self, device):
        loader = DataLoader(self.dataset, batch_size=2,
                            pin_memory=True, pin_memory_device='cuda')
        for sample in loader:
            self.assertTrue(sample['a_tensor'].is_pinned(device='cuda'))
            self.assertTrue(sample['another_dict']
                            ['a_number'].is_pinned(device='cuda'))

    def test_pin_memory_with_only_device(self, device):
        loader = DataLoader(self.dataset, batch_size=2,
                            pin_memory_device=device)
        for sample in loader:
            self.assertFalse(sample['a_tensor'].is_pinned(device=device))
            self.assertFalse(sample['another_dict']
                             ['a_number'].is_pinned(device=device))


class StringDataset(Dataset):
    def __init__(self):
        self.s = '12345'

    def __len__(self):
        return len(self.s)

    def __getitem__(self, ndx):
        return (self.s[ndx], ndx)


class TestStringDataLoader(TestCase):
    def setUp(self):
        self.dataset = StringDataset()

    def test_shuffle_pin_memory(self, device):
        loader = DataLoader(self.dataset, batch_size=2,
                            shuffle=True, num_workers=4, pin_memory=True)
        for (s, n) in loader:
            self.assertIsInstance(s[0], str)
            self.assertTrue(n.is_pinned())


class DictDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, ndx):
        return {
            'a_tensor': torch.empty(4, 2).fill_(ndx),
            'another_dict': {
                'a_number': ndx,
            },
        }


instantiate_device_type_tests(TestDictDataLoader, globals())
instantiate_device_type_tests(TestStringDataLoader, globals())

if __name__ == '__main__':
    run_tests()
