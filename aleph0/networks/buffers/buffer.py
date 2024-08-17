import torch, os, shutil, pickle


class ReplayBuffer:
    storage_dir = None

    def set_storage_dir(self, storage_dir, reset=False):
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        if reset:
            self.reset_storage()

    def reset_storage(self):
        """
        resets internal buffer
        """
        raise NotImplementedError

    def extend(self, items):
        for item in items:
            self.push(item)

    def push(self, item):
        """
        pushes an item into replay buffer
        Args:
            item: item
        Returns: item that is displaced, or None if no such item
        """
        raise NotImplementedError

    def sample_one(self):
        raise NotImplementedError

    def clear(self):
        pass

    def save(self, save_dir):
        pass

    def load(self, save_dir):
        pass

    def sample(self, batch, **kwargs):
        for _ in range(batch):
            yield self.sample_one()

    def __len__(self):
        raise NotImplementedError


class ReplayBufferDiskStorage(ReplayBuffer):
    def __init__(self,
                 storage_dir=None,
                 capacity=1e6,
                 device=None,
                 ):
        self.idx = 0
        self.size = 0
        self.capacity = capacity
        self.device = device
        if storage_dir is not None:
            self.set_storage_dir(storage_dir=storage_dir)

    def clear(self):
        super().clear()
        if self.storage_dir is not None:
            if os.path.exists(self.storage_dir):
                shutil.rmtree(self.storage_dir)

    def save(self, save_dir):
        super().save(save_dir=save_dir)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        shutil.copytree(src=self.storage_dir, dst=save_dir)

    def load(self, save_dir):
        super().load(save_dir=save_dir)
        self.clear()
        shutil.copytree(src=save_dir, dst=self.storage_dir)
        self.load_place(force=False)

    def reset_storage(self):
        self.clear()
        os.makedirs(self.storage_dir)
        self.size = 0
        self.idx = 0
        self.save_place()

    def save_place(self):
        """
        saves idx and size to files as well
        """
        pickle.dump(
            {
                'size': self.size,
                'idx': self.idx,
            },
            open(self._get_file('info'), 'wb')
        )

    def load_place(self, force=False):
        info_file = self._get_file(name='info')
        if os.path.exists(info_file):
            dic = pickle.load(open(info_file, 'rb'))
            self.size = dic['size']
            self.idx = dic['idx']
        else:
            if force:
                print('failed to load file:', info_file)
                print('resetting storage')
                self.reset_storage()
            else:
                raise Exception('failed to load file: ' + info_file)

    def _get_file(self, name):
        return os.path.join(self.storage_dir, str(name) + '.pkl')

    def push(self, item):
        if self.size == self.capacity:
            disp = self.__getitem__(self.idx)
        else:
            disp = None
        pickle.dump(item, open(self._get_file(self.idx), 'wb'))

        self.size = max(self.idx + 1, self.size)
        self.idx = int((self.idx + 1)%self.capacity)

        self.save_place()
        return disp

    def _grab_item_by_idx(self, idx, change_device=True):
        item = pickle.load(open(self._get_file(name=idx), 'rb'))
        return self._convert_device(item=item, change_device=change_device)

    def _convert_device(self, item, change_device):
        if change_device:
            if type(item) == tuple:
                item = tuple(self._convert_device(t, change_device=change_device)
                             for t in item)
            elif torch.is_tensor(item):
                item = item.to(self.device)
        return item

    def sample_one(self):
        return self[torch.randint(0, self.size, (1,))]

    def __getitem__(self, item):
        if item >= self.size:
            raise IndexError
        return self._grab_item_by_idx(idx=int((self.idx + item)%self.size))

    def __len__(self):
        return self.size


if __name__ == '__main__':
    test = ReplayBufferDiskStorage(capacity=3, storage_dir=os.path.join('replay_buffer_test'))
    test.extend('help')
    print([test[i] for i in range(len(test))])
    test.clear()
