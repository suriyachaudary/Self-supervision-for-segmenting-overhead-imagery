#!/bin/bash

# set paths
deepglobe_lands_root=/tmp/suriya/datasets/deepglobe_lands/
deepglobe_lands_splits=/tmp/suriya/datasets/deepglobe_lands/splits/
deepglobe_lands_img_root=/tmp/suriya/datasets/deepglobe_lands/land-train/
deepglobe_lands_gt_root=/tmp/suriya/datasets/deepglobe_lands/land-train/
deepglobe_lands_stride=228

echo 'creating directories for images and gt crops for train and validation splits'
mkdir -p $deepglobe_lands_root/processed/train/images/ $deepglobe_lands_root/processed/train/gt/ $deepglobe_lands_root/processed/val/images/ $deepglobe_lands_root/processed/val/gt/

echo 'creating train crops with stride of' ${deepglobe_lands_stride} x ${deepglobe_lands_stride}
while read -r line; do 
	echo 'processing ' $line;
	python utils/color_map_to_class_index.py ${deepglobe_lands_gt_root}/${line}_mask.png ${deepglobe_lands_gt_root}/${line}_mask_.png deepglobe_lands
	for row in {0..1824..228}; do
		pids="";
		for col in {0..1824..228}; do 
			echo 'creating ' ${line}_${row}_${col};
			convert ${deepglobe_lands_img_root}/${line}_sat.jpg -crop 612x612+${col}+${row} ${deepglobe_lands_root}/processed/train/images/${line}_${row}_${col}.jpg 2>/dev/null &
			convert ${deepglobe_lands_gt_root}/${line}_mask_.png -crop 612x612+${col}+${row} ${deepglobe_lands_root}/processed/train/gt/${line}_${row}_${col}.png 2>/dev/null &
			pids="$pids $!";
		done;
		wait $pids;
	done;
done < ${deepglobe_lands_splits}/train.txt

deepglobe_lands_stride=612
echo 'creating val crops with stride of' ${deepglobe_lands_stride} x ${deepglobe_lands_stride}
while read -r line; do 
	echo 'processing ' $line;
	python utils/color_map_to_class_index.py ${deepglobe_lands_gt_root}/${line}_mask.png ${deepglobe_lands_gt_root}/${line}_mask_.png deepglobe_lands
	for row in {0..1824..612}; do
		pids="";
		for col in {0..1824..612}; do 
			echo 'creating ' ${line}_${row}_${col};
			convert ${deepglobe_lands_img_root}/${line}_sat.jpg -crop 612x612+${col}+${row} ${deepglobe_lands_root}/processed/val/images/${line}_${row}_${col}.jpg 2>/dev/null &
			convert ${deepglobe_lands_gt_root}/${line}_mask_.png -crop 612x612+${col}+${row} ${deepglobe_lands_root}/processed/val/gt/${line}_${row}_${col}.png 2>/dev/null &
			pids="$pids $!";
		done;
		wait $pids;
	done;
done < ${deepglobe_lands_splits}/val.txt

