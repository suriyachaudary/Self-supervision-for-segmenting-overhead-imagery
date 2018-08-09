#!/bin/bash

# set paths
spacenet_root=/tmp/suriya/datasets/spacenet/
spacenet_splits=/tmp/suriya/datasets/spacenet/splits/
spacenet_individual_city_RGB_root=/tmp/suriya/datasets/spacenet/
spacenet_all_city_RGB_root=/tmp/suriya/datasets/spacenet/spacenet_processed/

mkdir -p $spacenet_all_city_RGB_root/tif/ $spacenet_all_city_RGB_root/train/{images,gt} \
										  $spacenet_all_city_RGB_root/val/{images,gt}

for f in $spacenet_individual_city_RGB_root/AOI*; do
	echo 'copying ' $f to $spacenet_all_city_RGB_root/tif/;
	cp $f/RGB-PanSharpen/*.tif $spacenet_all_city_RGB_root/tif/
done

echo "Converting SpaceNet RGB-PanSharpen 16-bit image to 8-bit RGB image"
python utils/spacenet_utils.py $spacenet_splits/train.txt $spacenet_all_city_RGB_root/tif/ \
							   $spacenet_all_city_RGB_root/train/images/ $spacenet_all_city_RGB_root/train/gt/ \
							   $spacenet_root/linestrings/Vegas_Paris_Shanghai_Khartoum_linestrings.txt 215

python utils/spacenet_utils.py $spacenet_splits/val.txt $spacenet_all_city_RGB_root/tif/ \
							   $spacenet_all_city_RGB_root/val/images/ $spacenet_all_city_RGB_root/val/gt/ \
							   $spacenet_root/linestrings/Vegas_Paris_Shanghai_Khartoum_linestrings.txt	650					   