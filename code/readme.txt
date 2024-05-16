共包含两个功能代码：

1、红外-可见光图像融合：在/code/reg_hw_rgb目录下：
	运行merge_loftr.py和merg_sg.py，在完整数据集上对比多种融合方法，并保存性能统计数据；
	运行plot.py和plot2.py画出统计图，统计各种指标
	运行merge_demo.py，融合单对图像

2、图像拼接：在/code/stitch目录下：
	运行stitcher_pano_all.py和stitcher_cross_all.py，在同模态和跨模态数据集上进行拼接，得到若干拼接结果；
	运行stitcher_eval.py和stitcher_cross_eval.py，统计数据指标；
	运行stitcher_two_draw.py和stitcher_cross_two_draw.py，对图像点匹配可视化；
	运行stitcher_demo.py，拼接特定图像序列

注意修改数据读入，和输出路径，默认目录为/home/fsh/stitching_and_registration/input_data
和/home/fsh/stitching_and_registration/output_data

环境配置：
参考https://github.com/zju3dv/LoFTR，https://github.com/magicleap/SuperGluePretrainedNetwork，https://github.com/magicleap/SuperPointPretrainedNetwork