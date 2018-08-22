#!/bin/bash

# set paths
deepglobe_roads_root=/tmp/suriya/datasets/deepglobe_roads/
deepglobe_roads_splits=/tmp/suriya/datasets/deepglobe_roads/splits/
deepglobe_roads_img_root=/tmp/suriya/datasets/deepglobe_roads/train/
deepglobe_roads_gt_root=/tmp/suriya/datasets/deepglobe_roads/train/
deepglobe_roads_stride=256

echo 'creating directories for images and gt crops for train and validation splits'
mkdir -p $deepglobe_roads_root/processed/train/images/ $deepglobe_roads_root/processed/train/gt/ $deepglobe_roads_root/processed/val/images/ $deepglobe_roads_root/processed/val/gt/

echo 'creating train crops with stride of' ${deepglobe_roads_stride} x ${deepglobe_roads_stride}
while read -r line; do 
	echo 'processing ' $line;
	for row in {0..512..256}; do
		pids="";
		for col in {0..512..256}; do 
			echo 'creating ' ${line}_${row}_${col};
			convert ${deepglobe_roads_img_root}/${line}_sat.jpg -crop 512x512+${col}+${row} ${deepglobe_roads_root}/processed/train/images/${line}_${row}_${col}.jpg 2>/dev/null &
			convert ${deepglobe_roads_gt_root}/${line}_mask.png -crop 512x512+${col}+${row} ${deepglobe_roads_root}/processed/train/gt/${line}_${row}_${col}.png 2>/dev/null &
			pids="$pids $!";
		done;
		wait $pids;
	done;
done < ${deepglobe_roads_splits}/train.txt

deepglobe_roads_stride=512
echo 'creating val crops with stride of' ${deepglobe_roads_stride} x ${deepglobe_roads_stride}
while read -r line; do 
	echo 'processing ' $line;
	python utils/color_map_to_class_index.py ${deepglobe_roads_gt_root}/${line}_mask.png ${deepglobe_roads_gt_root}/${line}_mask_.png deepglobe_roads
	for row in {0..512..512}; do
		pids="";
		for col in {0..512..512}; do 
			echo 'creating ' ${line}_${row}_${col};
			convert ${deepglobe_roads_img_root}/${line}_RGB.tif -crop 512x512+${col}+${row} ${deepglobe_roads_root}/processed/val/images/${line}_${row}_${col}.jpg 2>/dev/null &
			convert ${deepglobe_roads_gt_root}/${line}_label.png -crop 512x512+${col}+${row} ${deepglobe_roads_root}/processed/val/gt/${line}_${row}_${col}.png 2>/dev/null &
			pids="$pids $!";
		done;
		wait $pids;
	done;
done < ${deepglobe_roads_splits}/val.txt

