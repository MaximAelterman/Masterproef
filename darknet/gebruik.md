VOOR GEBRUIK: YOLOv3-608 weights downloaden en in /darknet plaatsen.
download hier: https://pjreddie.com/darknet/yolo/

C:
./darknet detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights [path_to_img] (img path optioneel)

python:
python2 python/darknet.py

python op kitti:
python2 python/darknet_kitti.py --data_path ~/Kitti/object/training/ --split_file ~/Kitti/object/trainval.txt

darknet.py aanpassen voor andere modellen/afbeeldingen

