import os.path as osp
import tempfile
from .builder import DATASETS
from .custom import CustomDataset
import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image


PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]]

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

@DATASETS.register_module()
class PascalContextDataset(CustomDataset):
    """PascalContext dataset.
    In segmentation map annotation for PascalContext, 0 stands for background,
    which is included in 60 categories. ``reduce_zero_label`` is fixed to
    False. The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png'.
    Args:
        split (str): Split txt file for PascalContext.
    """

    CLASSES = ('background', 'aeroplane', 'bag', 'bed', 'bedclothes', 'bench',
               'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus',
               'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth',
               'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence',
               'floor', 'flower', 'food', 'grass', 'ground', 'horse',
               'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person',
               'plate', 'platform', 'pottedplant', 'road', 'rock', 'sheep',
               'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table',
               'track', 'train', 'tree', 'truck', 'tvmonitor', 'wall', 'water',
               'window', 'wood')

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]]

    def __init__(self, split, **kwargs):
        super(PascalContextDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            split=split,
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None


@DATASETS.register_module()
class PascalContextDataset59(CustomDataset):
    """PascalContext dataset.
    In segmentation map annotation for PascalContext, 0 stands for background,
    which is included in 60 categories. ``reduce_zero_label`` is fixed to
    False. The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is
    fixed to '.png'.
    Args:
        split (str): Split txt file for PascalContext.
    """

    CLASSES = ('aeroplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle',
               'bird', 'boat', 'book', 'bottle', 'building', 'bus', 'cabinet',
               'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow',
               'cup', 'curtain', 'dog', 'door', 'fence', 'floor', 'flower',
               'food', 'grass', 'ground', 'horse', 'keyboard', 'light',
               'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform',
               'pottedplant', 'road', 'rock', 'sheep', 'shelves', 'sidewalk',
               'sign', 'sky', 'snow', 'sofa', 'table', 'track', 'train',
               'tree', 'truck', 'tvmonitor', 'wall', 'water', 'window', 'wood')

    PALETTE = [[180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
               [120, 120, 80], [140, 140, 140], [204, 5, 255], [230, 230, 230],
               [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61],
               [120, 120, 70], [8, 255, 51], [255, 6, 82], [143, 255, 140],
               [204, 255, 4], [255, 51, 7], [204, 70, 3], [0, 102, 200],
               [61, 230, 250], [255, 6, 51], [11, 102, 255], [255, 7, 71],
               [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92],
               [112, 9, 255], [8, 255, 214], [7, 255, 224], [255, 184, 6],
               [10, 255, 71], [255, 41, 10], [7, 255, 255], [224, 255, 8],
               [102, 8, 255], [255, 61, 6], [255, 194, 7], [255, 122, 8],
               [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255],
               [235, 12, 255], [160, 150, 20], [0, 163, 255], [140, 140, 140],
               [250, 10, 15], [20, 255, 0], [31, 255, 0], [255, 31, 0],
               [255, 224, 0], [153, 255, 0], [0, 0, 255], [255, 71, 0],
               [0, 235, 255], [0, 173, 255], [31, 0, 255]]

    def __init__(self, split, **kwargs):
        super(PascalContextDataset59, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            split=split,
            reduce_zero_label=True,
            **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
    def results2img(self, results, imgfile_prefix):
        """Write the segmentation results to images.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        prog_bar = mmcv.ProgressBar(len(self))
        palette = get_palette(len(PALETTE))
        for idx in range(len(self)):
            result = results[idx]
            #if to_label_id:
            #    result = self._convert_to_label_id(result)
            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            output = Image.fromarray(result.astype(np.uint8)).convert('P')
            #import cityscapesscripts.helpers.labels as CSLabels
            #palette = np.zeros((len(CSLabels.id2label), 3), dtype=np.uint8)
            #for label_id, label in CSLabels.id2label.items():
            #    palette[label_id] = label.color

            output.putpalette(palette)
            output.save(png_filename)
            result_files.append(png_filename)
            prog_bar.update()

        return result_files

    def format_results(self, results, imgfile_prefix=None):
        """Format the results into dir (standard format for Cityscapes
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        if imgfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            imgfile_prefix = tmp_dir.name
        else:
            tmp_dir = None
        result_files = self.results2img(results, imgfile_prefix)

        return result_files, tmp_dir