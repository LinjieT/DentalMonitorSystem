import os.path as osp
from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()  # 注册   不要忘记在__init__.py作显示导入


class cbctdataset(CustomDataset):
    CLASSES = ('background', 'down','up')   # 类别名称设置
    PALETTE = [[0, 0, 255], [255, 0, 0],[0, 255, 0]]  # 调色板设置
    
    #CLASSES = ('background', 'tooth')   # 类别名称设置
    #PALETTE = [[0, 0, 255], [255, 0, 0]]  # 调色板设置

    def __init__(self,**kwargs):
        super(cbctdataset, self).__init__(
            img_suffix='.png',  # img文件‘后缀’
            seg_map_suffix='.png',  # gt文件‘后缀’
            reduce_zero_label=True,
            **kwargs)
        assert osp.exists(self.img_dir)