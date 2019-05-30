#### Self-Supervised Feature Learning for Semantic Segmentation of Overhead Imagery

If you find our work useful in your research, please cite:

    @inproceedings{singhBMVC18overhead,
        Author = {Singh, Suriya; Batra, Anil; Pang, Guan; Torresani, Lorenzo; Basu, Saikat; Paluri, Manohar; Jawahar, C. V.},
        Title = {Self-supervised Feature Learning for Semantic Segmentation of Overhead Imagery},
        Booktitle = {BMVC},
        Year = {2018}
    }

#### Dependencies
* [PyTorch 0.3](https://pytorch.org/)
* [CUDA-8.0](https://developer.nvidia.com/cuda-80-ga2-download-archive)
* [OpenCV](https://opencv.org/)
* [tifffile](https://www.lfd.uci.edu/~gohlke/code/tifffile.py.html)


#### Datasets
* [ISPRS Potsdam](http://www2.isprs.org/commissions/comm3/wg4/data-request-form2.html)
* [SpaceNet Roads](https://spacenetchallenge.github.io/Challenges/Challenge-3.html)
* [DeepGlobe Lands](http://deepglobe.org/index.html)
* [DeepGlobe Roads](http://deepglobe.org/index.html)
* **Note** Results reported are on our splits.

##### Preparing datasets for training
* Potsdam
```bash 
bash prepare_potsdam.sh
```
* SpaceNet
```bash 
bash prepare_spacenet.sh
```
* DeepGlobe Lands
```bash 
bash prepare_deepglobe_lands.sh
```
* DeepGlobe Roads
```bash 
bash prepare_deepglobe_roads.sh
```

##### Please refer to jupyter notebook for semantic inpainting (Pathak et al, CVPR 2016), adversarial mask prediction, and semantic segmentation codes:
[Notebook](https://github.com/suriyasingh/Self-supervision-for-segmenting-overhead-imagery/blob/master/Self_supervised_Feature_Learning_for_Semantic_Segmentation_of_Overhead_Imagery.ipynb)

Toggling `use_coach` flag in the notebook is sufficient to switch between our method and Context Encoders. Random patches from the image are erased when `use_coach = False` whereas the mask values are sampled from uniform distribution in `iteration 0` or predicted by the coach network in `iteration > 0` when `use_coach = True`.

##### For other ResNet-18 baselines in Table 2,

###### scratch :
`from models import resnet18, FCNify
net = resnet18().cuda()
net_segmentation = FCNify(net, n_class = nClasses)`

###### ImageNet :
`from models import resnet18, FCNify` 
`net = resnet18(pretrained=True).cuda()` 
`net_segmentation = FCNify(net, n_class = nClasses)`

###### autoencoder with bottleneck and retaining pre-trained deocder:
`net = resnet18_encoderdecoder_wbottleneck().cuda()`
`erase_count = 0 ### number of blocks to erase from image`
`net_segmentation = FCNify_v2(net, n_class = nClasses)`

###### autoencoder without bottleneck and retaining pre-trained deocder:
`net = resnet18_encoderdecoder_wbottleneck().cuda()`
`erase_count = 0 ### number of blocks to erase from image`
`net_segmentation = FCNify_v2(net, n_class = nClasses)`
