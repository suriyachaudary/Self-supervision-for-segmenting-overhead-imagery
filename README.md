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
* [SpaceNet Roads](https://spacenetchallenge.github.io/Competitions/Competition3.html)
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

##### Please refer to jupyter notebook for reproducing our results
[Notebook](https://github.com/suriyasingh/Self-supervision-for-segmenting-overhead-imagery/blob/master/Self_supervised_Feature_Learning_for_Semantic_Segmentation_of_Overhead_Imagery.ipynb)
