avod:			3D object detectie op basis van LIDAR. 
			Github pagina: https://github.com/kujason/avod
			evaluatie in avod/avod/data/outputs/.../predicions/kitti_native_eval: 
				./evaluate_object_3d_offline ~/Kitti/object/training/label_2 0.1/120000
darknet:		YOLOv3 detector
disparity_test:		zelf gemaakt, om .npy disparity maps te visualiseren, tests met opencv stereo estimators
eval_kitti:		officiële kitti evaluation toolkit, zelfde als in avod/data/outputs/.../predictions/kitti_native_eval
kitti_object_vis:	visualisatie van lidar beelden + labels
pseudo_lidar:		bevat de code om pseudo lidar te genereren + PSMNet voor disparity map.
			Github pagina voor pseudo-LIDAR: https://github.com/mileyan/pseudo_lidar
			Github pagina voor PSMNet: https://github.com/JiaRenChang/PSMNet
thesis:			bevat de thesis in pdf-vorm

De KITTI dataset kan hier gedownload worden: http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
