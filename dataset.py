import torch
from torch.utils.data import Dataset
from utils import is_dict, plot_boxes_cv2
from image import *
from old_cfg import cfg
import pdb
from torchvision import transforms


def build_dataset(dataopt):
    # Base training dataset
    if not cfg.tuning:
        return load_paths(dataopt['train'])

    # Meta tuning dataset
    if cfg.repeat == 1:
        return load_paths(dataopt['meta'])
    else:
        if 'dynamic' not in dataopt or int(dataopt['dynamic']) == 0:
            return load_paths(dataopt['meta']) * cfg.repeat
        else:
            # dynamic == 1
            metalist, metacnt = load_metadict(dataopt['meta'], cfg.repeat)
            return build_fewset(dataopt['train'], metalist, metacnt, cfg.shot * cfg.repeat)


def load_metadict(metapath, repeat=1):
    with open(metapath, 'r') as f:
        files = []
        for line in f.readlines():
            pair = line.rstrip().split()
            if len(pair) == 2:
                pass
            elif len(pair) == 4:
                pair = [pair[0] + ' ' + pair[1], pair[2] + ' ' + pair[3]]
            else:
                raise NotImplementedError('{} not recognized'.format(pair))
            files.append(pair)
        # files = [line.rstrip().split() for line in f.readlines()]

        metadict = {line[0]: load_paths(line[1]) for line in files}

    pdb.set_trace()
    # Remove base-class images
    for k in metadict.keys():
        if k not in cfg.novel_classes:
            metadict[k] = []
    metalist = set(sum(metadict.values(), []))

    # Count bboxes
    metacnt = {c: 0 for c in metadict.keys()}
    for imgpath in metalist:
        labpath = get_label_path(imgpath.strip())
        # Load converted annotations
        bs = np.loadtxt(labpath)
        bs = np.reshape(bs, (-1, 5))
        bcls = bs[:, 0].astype(np.int).tolist()
        for ci in set(bcls):
            metacnt[cfg.classes[ci]] += bcls.count(ci)

    for c in metacnt.keys():
        metacnt[c] *= repeat

    metalist = list(metalist) * repeat
    return metalist, metacnt


def build_fewset(imglist, metalist, metacnt, shot, replace=True):
    # Random sample bboxes for base classes
    if isinstance(imglist, str):
        with open(imglist) as f:
            names = f.readlines()
    elif isinstance(imglist, list):
        names = imglist.copy()
    else:
        raise NotImplementedError('imglist type not recognized')

    while min(metacnt.values()) < shot:
        imgpath = random.sample(names, 1)[0]
        labpath = get_label_path(imgpath.strip())
        # Remove empty annotation
        if not os.path.getsize(labpath):
            names.remove(imgpath)
            continue

        # Load converted annotations
        bs = np.loadtxt(labpath)
        bs = np.reshape(bs, (-1, 5))
        bcls = bs[:, 0].astype(np.int).tolist()

        if bs.shape[0] > 3:
            continue

        # Remove images contatining novel objects
        if not set(bcls).isdisjoint(set(cfg.novel_ids)):
            names.remove(imgpath)
            continue

        # Check total number of bbox per class so far
        overflow = False
        for ci in set(bcls):
            if metacnt[cfg.classes[ci]] + bcls.count(ci) > shot:
                overflow = True
                break
        if overflow:
            names.remove(imgpath)
            continue

        # Add current imagepath to the file lists
        for ci in set(bcls):
            metacnt[cfg.classes[ci]] += bcls.count(ci)
        metalist.append(imgpath)

        # To avoid duplication
        if not replace:
            names.remove(imgpath)

    random.shuffle(metalist)
    return metalist


def load_paths(root, checkvalid=True):
    if is_dict(root):
        lines = []
        with open(root, 'r') as f:
            # files = [line.rstrip().split()[-1] for line in f.readlines()]
            files = [line.rstrip().split() for line in f.readlines()]
            if checkvalid:
                files = [line[-1] for line in files if line[0] in cfg.base_classes]
            else:
                files = [line[-1] for line in files if line[0] in cfg.classes]
        for file in files:
            with open(file, 'r') as f:
                lines.extend(f.readlines())
        lines = sorted(list(set(lines)))
    else:
        with open(root, 'r') as file:
            lines = file.readlines()
    if checkvalid:
        # check whether the classes of the images contains meta classes.
        lines = [l for l in lines if is_valid(l)]
    return lines


def get_label_path(img_path):
    # VOC only
    subdir = 'labels'
    labpath = img_path.replace('images', subdir) \
        .replace('JPEGImages', subdir) \
        .replace('.jpg', '.txt').replace('.png', '.txt')
    return labpath


def get_label_path_cls(imgpath, cls_name):
    # if cfg.data == 'voc':
    #     labpath = imgpath.replace('images', 'labels_1c/{}'.format(cls_name)) \
    #         .replace('JPEGImages', 'labels_1c/{}'.format(cls_name)) \
    #         .replace('.jpg', '.txt').replace('.png', '.txt')
    # else:
    #     if 'train2014' in imgpath:
    #         labpath = imgpath.replace('images/train2014', 'labels_1c/train2014/{}'.format(cls_name)) \
    #             .replace('.jpg', '.txt').replace('.png', '.txt')
    #     elif 'val2014' in imgpath:
    #         labpath = imgpath.replace('images/val2014', 'labels_1c/val2014/{}'.format(cls_name)) \
    #             .replace('.jpg', '.txt').replace('.png', '.txt')
    #     else:
    #         raise NotImplementedError("Image path note recognized!")

    label_path = imgpath.replace('images', 'labels_1c/{}'.format(cls_name)) \
        .replace('JPEGImages', 'labels_1c/{}'.format(cls_name)) \
        .replace('.jpg', '.txt').replace('.png', '.txt')

    return label_path


def is_valid(img_path):
    label_path = get_label_path(img_path.rstrip())
    if os.path.getsize(label_path):
        bs = np.loadtxt(label_path)
        if bs is not None:
            bs = np.reshape(bs, (-1, 5))
            clsset = set(bs[:, 0].astype(np.int).tolist())
            if not clsset.isdisjoint(set(cfg.base_ids)):
                # whether current image contains classes of base classes
                return True
    return False


class ListDataset(Dataset):
    def __init__(self, data, shape=None, shuffle=True,
                 transform=None, target_transform=None,
                 train=False, seen_batch=0, batch_size=24, num_workers=4):
        self.train = train

        if isinstance(data, list):
            # a list of txt files containing labels of training image.
            self.img_paths = data
        # don't get the usage of this code
        # elif is_dict(directory):
        #     lines = []
        #     with open(directory, 'r') as f:
        #         files = [line.rstrip().split()[-1] for line in f.readlines()]
        #     for file in files:
        #         with open(file, 'r') as f:
        #             lines.extend(f.readlines())
        #     self.lines = sorted(list(set(lines)))
        elif isinstance(data, str):
            with open(data, 'r') as file:
                self.img_paths = [line for line in file.readlines()]
        else:
            raise TypeError('Not supported data type')

        # Filter out images not in base classes
        if self.train and not isinstance(data, list):
            print("===> Number of samples (before filtring): %d" % len(self.img_paths))
            self.img_paths = [line for line in self.img_paths if is_valid(line)]
            print("===> Number of samples (after filtring): %d" % len(self.img_paths))

        if shuffle:
            random.shuffle(self.img_paths)

        self.num_samples = len(self.img_paths)

        self.transform = transform if transform is not None else transforms.ToTensor()
        self.target_transform = target_transform

        self.shape = shape
        self.seen_batch = seen_batch
        # seems not using
        # self.batch_size = batch_size
        self.num_workers = num_workers
        # it seems this line of code has no effect
        # self.first_batch = False

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img_path = self.img_paths[index].rstrip()

        jitter = 0.2
        hue = 0.1
        saturation = 1.5
        exposure = 1.5

        label_path = get_label_path(img_path)
        # returned labels already filter out those not in base_ids
        img, label = load_data_detection(img_path, label_path, self.shape,
                                         jitter, hue, saturation, exposure,
                                         data_aug=self.train)

        label = torch.from_numpy(label)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        # every worker will fetch a batch of data
        self.seen_batch = self.seen_batch + self.num_workers

        return img, label


class MetaDataset(Dataset):
    def __init__(self, meta_files, transform=None, target_transform=None,
                 train=False, num_workers=4, evaluate=False, with_ids=False):

        # Backup labeled image paths (for meta-model)
        if train:
            self.classes = cfg.base_classes
            factor = 1
            if cfg.data == 'coco':
                factor = 4
        else:
            # self.classes = cfg.base_classes
            if cfg.data == 'coco':
                self.classes = cfg.base_classes
            else:
                self.classes = cfg.classes
            factor = 10
        # print('num classes: ', len(self.classes))

        # use the number of dataset size might be better
        # num_batch = factor * 500 * 64 * cfg.num_gpus // cfg.batch_size
        dataset_size = factor * 500 * 64
        num_batch = dataset_size // cfg.batch_size

        meta_indexes = [[]] * len(self.classes)
        with open(meta_files, 'r') as f:
            meta_files = []
            for line in f.readlines():
                pair = line.strip().split()
                if len(pair) == 2:
                    pass
                # what is this used for
                # elif len(pair) == 4:
                #     pair = [pair[0] + ' ' + pair[1], pair[2] + ' ' + pair[3]]
                else:
                    raise NotImplementedError('{} not recognized'.format(pair))
                meta_files.append(pair)
            meta_files = {k: v for k, v in meta_files}

            self.meta_paths = [[]] * len(self.classes)
            for i, cls_name in enumerate(self.classes):
                with open(meta_files[cls_name], 'r') as img_path_file:
                    img_paths = img_path_file.readlines()
                    self.meta_paths[i] = img_paths
                    if evaluate:
                        meta_indexes[i] = list(zip([i] * len(img_paths), list(range(len(img_paths)))))
                    else:
                        # for every class, randomly sample img num_batch times.
                        indexes = np.random.choice(range(len(img_paths)), num_batch).tolist()
                        meta_indexes[i] = list(zip([i] * num_batch, indexes))

        # sum([[1,2],[3,4]],[])-->[1,2,3,4]
        # sum(((1,2),(3,4)),())-->(1,2,3,4)
        self.indexes = sum(meta_indexes, []) if evaluate \
            else sum(list(zip(*meta_indexes)), ())
        self.meta_counts = [len(paths) for paths in self.meta_paths]

        # if cfg.rand_meta:
        #     self.indexes = list(self.indexes)
        #     random.shuffle(self.indexes)
        #     self.indexes = tuple(self.indexes)

        self.get_with_id = with_ids
        self.evaluate = evaluate
        # why need to multiply num gpus here?
        # self.batch_size = len(self.classes) * cfg.num_gpus
        self.batch_size = len(self.classes)
        self.meta_shape = (cfg.meta_height, cfg.meta_width)
        self.mask_shape = (cfg.meta_height, cfg.meta_width)

        self.transform = transform if transform is not None \
            else transforms.ToTensor()
        self.target_transform = target_transform

        self.train = train
        self.num_workers = num_workers

        # it seems these lines of code could be replaced by transform
        # self.meta_transform = transforms.Compose([
        #     transforms.ToTensor(),
        # ])

        if evaluate:
            self.indexes = self.build_mask(self.indexes)

        self.num_samples = len(self.indexes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        cls_id, meta_ind = self.indexes[index]

        img, mask = self.get_meta_image_resampling(cls_id, meta_ind)

        if self.get_with_id:
            return img, mask, cls_id
        else:
            return img, mask

    def get_meta_img(self, clsid, imgpath):
        jitter = 0.2
        hue = 0.1
        saturation = 1.5
        exposure = 1.5

        if isinstance(imgpath, int):
            imgpath = self.meta_paths[clsid][imgpath].rstrip()
        elif isinstance(imgpath, str):
            pass
        else:
            raise NotImplementedError("{}: img path not recognized")

        label_path = get_label_path_cls(imgpath, self.classes[clsid])
        img, lab = load_data_with_label(
            imgpath, label_path, self.meta_shape, jitter, hue, saturation, exposure, data_aug=self.train)
        return img, lab

    def get_img_mask(self, img, boxes, merge=True):
        w, h = self.mask_shape
        mask = None

        for box in boxes:
            x1 = int(max(0, round((box[0] - box[2] / 2) * w)))
            y1 = int(max(0, round((box[1] - box[3] / 2) * h)))
            x2 = int(min(w, round((box[0] + box[2] / 2) * w)))
            y2 = int(min(h, round((box[1] + box[3] / 2) * h)))

            # never use metaintype 3 and 4
            # if cfg.metain_type in [3, 4]:
            #     croped = img.crop((x1, y1, x2, y2)).resize(img.size)
            #     croped = self.transform(croped)
            #     img = self.transform(img)
            #     img = torch.cat([img, croped])
            # else:
            #     img = self.transform(img)

            if x1 == x2 or y1 == y2:
                continue
            else:
                mask = torch.zeros((1, h, w)) if mask is None else mask
                mask[:, y1:y2, x1:x2] = 1

        img = self.transform(img)

        if merge:
            return torch.cat([img, mask])
        else:
            return img, mask

    def get_meta_image_resampling(self, clsid, metaind):
        """automatically resample the asked image if label is difficult,
        since difficult label have been deleted in voc label.
        """
        meta_img, meta_lab = self.get_meta_img(clsid, metaind)
        if meta_lab:
            # for lab in meta_lab:
            #     # if this image has more than one label, pick one of them, because only need one label to build tha
            #     # mask.
            #     img, mask = self.get_img_mask(meta_img, lab, merge=False)
            #     if mask is None:
            #         continue
            #     return img, mask
            img, mask = self.get_img_mask(meta_img, meta_lab, merge=False)
            if mask is not None:
                return img, mask

        # In case the selected meta image has only difficult objects
        # then random sample another image
        # if evaluate, skip this image
        while True and not self.evaluate:
            meta_imgpath = random.sample(self.meta_paths[clsid], 1)[0].rstrip()
            meta_img, meta_lab = self.get_meta_img(clsid, meta_imgpath)
            if not meta_lab:
                continue
            # for lab in meta_lab:
            #     img, mask = self.get_img_mask(meta_img, lab, merge=False)
            #     if mask is None:
            #         continue
            #     return img, mask
            img, mask = self.get_img_mask(meta_img, meta_lab, merge=False)
            if mask is None:
                continue
            return img, mask
        return None, None

    def build_mask(self, indexes):
        new_inds = []
        print('===> buliding mask...')
        _cnt = 0
        for clsid, metaind in indexes:
            print('buliding mask | {}/{}'.format(_cnt, len(indexes)))
            _cnt += 1
            img, mask = self.get_meta_image_resampling(clsid, metaind)
            if img is not None:
                new_inds.append((clsid, metaind))
        return new_inds


def debug_image_label(image, box, shape):
    import cv2
    import numpy as np
    height, width, = shape
    # convert from CWH to HWC
    image = image.view(3, -1).transpose(1, 0).view(height, width, 3)

    image = np.array(image) * 255
    image = image.astype(np.uint8)

    # cv2.imshow('image', image)
    # cv2.moveWindow("image", 100, 100)
    # cv2.waitKey()

    x1 = int(round((box[0] - box[2] / 2.0) * width))
    y1 = int(round((box[1] - box[3] / 2.0) * height))
    x2 = int(round((box[0] + box[2] / 2.0) * width))
    y2 = int(round((box[1] + box[3] / 2.0) * height))

    image_temp = np.zeros((height, width, 3), np.uint8)
    image_temp[:] = image[:]

    # print(image.shape, label)

    cv2.rectangle(image_temp, (x1, y1), (x2, y2), (255, 0, 0))
    cv2.imshow('image', image_temp)
    cv2.moveWindow("image", 100, 100)
    cv2.waitKey()


def debug_img_mask(image, mask):
    import cv2
    import numpy as np

    width, height = image.shape[1:]
    # convert from CWH to HWC
    image = image.view(3, -1).transpose(1, 0).view(height, width, 3)
    maks = mask.view(height, width)

    image = np.array(image) * 255
    image = image.astype(np.uint8)
    maks = np.array(maks) * 255
    maks = maks.astype(np.uint8)

    image_temp = np.zeros((height, width, 3), np.uint8)
    image_temp[:] = image[:]
    mask_temp = np.zeros((height, width), np.uint8)
    mask_temp[:] = maks[:]

    # print(image.shape, label)

    # cv2.rectangle(image_temp, (x1, y1), (x2, y2), (255, 0, 0))
    cv2.imshow('image', image_temp)
    cv2.moveWindow("image", 100, 100)

    cv2.imshow('mask', mask_temp)
    cv2.moveWindow('mask', 300, 300)
    cv2.waitKey()


if __name__ == '__main__':
    from utils import read_data_cfg
    from old_cfg import parse_cfg
    from PIL import Image, ImageDraw

    datacfg = 'cfg/metayolo.data'
    netcfg = 'cfg/darknet_dynamic.cfg'
    metacfg = 'cfg/reweighting_net.cfg'

    data_options = read_data_cfg(datacfg)
    net_options = parse_cfg(netcfg)[0]
    meta_options = parse_cfg(metacfg)[0]

    cfg.config_data(data_options)
    cfg.config_meta(meta_options)
    cfg.config_net(net_options)
    cfg.num_gpus = 1
    cfg.batch_size = 64

    # listdataset
    # data = r'D:\Code\Python\data\voc_few_shot/voclist/aeroplane_train.txt'
    # dataset = ListDataset(data, shape=(300, 300), num_workers=0)
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    # for img, label in data_loader:
    #     img = img[0]
    #
    #     label = label[0]
    #     box = label[0][1:5].numpy()
    #
    #     debug_image_label(img, box, (300, 300))
    #
    #     print(img.shape, label)

    # metadataset
    data = None
    meta_files = 'data/voc_traindict_full.txt'
    metaset = MetaDataset(meta_files, train=True)
    metaloader = torch.utils.data.DataLoader(
        metaset,
        batch_size=metaset.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    for img, mask in metaloader:
        print(img.shape, mask.shape)
        for i in range(len(img)):
            debug_img_mask(img[i], mask[i])
