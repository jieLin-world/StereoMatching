import os

from data.reader.base_reader import BaseReader

class KittiReader(BaseReader):
    def __init__(self, root, list_file, image_reader='PIL', disp_reader='PIL', right_disp=False, use_noc=False):
        super().__init__(root, list_file, image_reader, disp_reader, right_disp)
        self.use_noc = use_noc
        assert disp_reader == 'PIL', 'Kitti Disp only support PIL format'

    def item_loader(self, item):
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path, disp_img_path = full_paths
        if self.use_noc:
            disp_img_path = disp_img_path.replace('disp_occ', 'disp_noc')
        left_img = self.image_loader(left_img_path)
        right_img = self.image_loader(right_img_path)
        disp_img = self.disp_loader(disp_img_path) / 256.0
        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
        }
        if self.return_right_disp:
            disp_img_right_path = disp_img_path.replace('c_0', 'c_1')
            disp_img_right = self.disp_loader(disp_img_right_path) / 256.0
            sample['disp_right'] = disp_img_right
        return sample


class KittiTestReader(KittiReader):
    def __init__(self, root, list_file, image_reader='PIL', disp_reader='PIL', right_disp=False, use_noc=False):
        super().__init__(root, list_file, image_reader, disp_reader, right_disp, use_noc)

    def item_loader(self, item):
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path = full_paths[:2]
        left_img = self.image_loader(left_img_path)
        right_img = self.image_loader(right_img_path)
        sample = {
            'left': left_img,
            'right': right_img,
            'name': left_img_path.split('/')[-1],
        }
        return sample


if __name__ == '__main__':
    dataset = KittiReader(root='/mnt/cephfs/dataset/stereo_matching/kitti2015', list_file='/mnt/cephfs/home/zhihongyan/linjie/sm/filenames/KITTI15/kitti15_train200.txt')
    print(dataset)
    sample = dataset[0]
    print(sample['left'].shape, sample['right'].shape, sample['disp'].shape)


#text:data_list, full_paths, left_img_path, left_img, sample
""" full_paths = [os.path.join(dataset.root, x) for x in dataset.data_list[0]]
left_img_path, right_img_path, disp_img_path = full_paths
left_img = dataset.image_loader(left_img_path)
right_img = dataset.image_loader(right_img_path)
disp_img = dataset.disp_loader(disp_img_path) / 256.0
sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,}
print(dataset.data_list[0])
print(full_paths)
print(left_img_path)
print(right_img_path)
print(disp_img_path)
print(left_img)
print(right_img)
print(disp_img)
print(sample)
 """