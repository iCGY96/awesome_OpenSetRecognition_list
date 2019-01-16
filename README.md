
# Awesome Open Set Recognition list [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

> A curated list of papers & ressources linked to open set recognition and open world recognition

**Note that:**
- **This list is not exhaustive.**
- **Tables use alphabetical order for fairness.**


## Contents

- [Tutorials](#tutorials)

- [Papers](#papers)
	- [Open-Set-Recognition](#papers-osr)
    	- [Traditional Machine Learning Methods-based](#papers-osr-ml)
    	- [Deep Neural Network-based](#papers-osr-dnn)
    	- [Adversarial Learning-based](#papers-osr-al)
    	- [EVT-based](#papers-osr-evt)
	- [Open-World-Recognition](#papers-owr)

- [OpenSource software resources](#opensource)
	- [Open-Set-Recognition](#opensource-osr)
	- [Open-World-Recognition](#opensource-owr)

- [Datasets with ground truth - Reproducible research](#dataset)

- [License](#license)

- [Contributing](#contributing)

<a name="tutorials"></a>
# Tutorials

## Tutorial & survey

[Toward Open Set Recognition](https://ieeexplore.ieee.org/abstract/document/6365193), Scheirer W J, de Rezende Rocha A, Sapkota A, et al. PAMI, 2013

[Towards Open World Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Bendale_Towards_Open_World_2015_CVPR_paper.html), Bendale A, Boult T. CVPR, 2015.

[Recent Advances in Open Set Recognition: A Survey](https://arxiv.org/abs/1811.08581), Geng C, Huang S, Chen S. arXiv, 2018.

<a name="papers"></a>
# Papers

<a name="papers-osr"></a>
## Open Set Recognition

<a name="papers-osr-ml"></a>
### Traditional Machine Learning Methods-based

[Toward Open Set Recognition](https://ieeexplore.ieee.org/abstract/document/6365193), Scheirer W J, de Rezende Rocha A, Sapkota A, et al. PAMI, 2013.

[Probability models for open set recognition](https://ieeexplore.ieee.org/abstract/document/6809169/), Scheirer W J, Jain L P, Boult T E. PAMI, 2014.

[Multi-class open set recognition using probability of inclusion](https://link.springer.com/chapter/10.1007/978-3-319-10578-9_26), Jain L P, Scheirer W J, Boult T E. ECCV, 2014.

[Breaking the closed world assumption in text classification](http://www.aclweb.org/anthology/N16-1061), Fei G, Liu B. NAACL, 2016.

[Sparse representation-based open set recognition](https://ieeexplore.ieee.org/abstract/document/7577876/), Zhang H, Patel V M. PAMI, 2017.

[Best fitting hyperplanes for classification](https://ieeexplore.ieee.org/abstract/document/7506010/), Cevikalp H. PAMI, 2017.

[Polyhedral conic classifiers for visual object detection and classification](http://openaccess.thecvf.com/content_cvpr_2017/papers/Cevikalp_Polyhedral_Conic_Classifiers_CVPR_2017_paper.pdf), Cevikalp H, Triggs B. Rigling B D. CVPR, 2017.

[Fast and Accurate Face Recognition with Image Sets](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w23/Cevikalp_Fast_and_Accurate_ICCV_2017_paper.pdf), Cevikalp H, Yavuz H S. ICCVW, 2017.

[Nearest neighbors distance ratio open-set classifier](https://link.springer.com/article/10.1007/s10994-016-5610-8), Júnior P R M, de Souza R M, Werneck R O, et al. Machine Learning, 2017.

[Data-Fusion Techniques for Open-Set Recognition Problems](https://ieeexplore.ieee.org/abstract/document/8332940/), Neira M A C, Júnior P R M, Rocha A, et al. IEEE Access, 2018.

[Towards open-set face recognition using hashing functions](https://ieeexplore.ieee.org/abstract/document/8272751/), Vareto R, Silva S, Costa F, et al. IJCB, 2018.


<a name="papers-osr-dnn"></a>
### Deep Neural Network-based

[A bounded neural network for open set recognition](https://ieeexplore.ieee.org/abstract/document/7280680/), Cardoso D O, França F, Gama J. IJCNN, 2015.

[Weightless neural networks for open set recognition](https://link.springer.com/article/10.1007/s10994-017-5646-4), Cardoso D O, Gama J, França F M G. Machine Learning, 2017.

[Towards open set deep networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html), Bendale A, Boult T E. CVPR, 2016.

[*Adversarial Robustness: Softmax versus Openmax](https://arxiv.org/abs/1708.01697), Rozsa A, Günther M, Boult T E. arXiv, 2017.

[*Deep open classification of text documents](https://arxiv.org/abs/1709.08716), Shu L, Xu H, Liu B. Doc. arXiv, 2017.

[Open Set Text Classification using Convolutional Neural Networks](http://www.cs.uccs.edu/~jkalita/papers/2017/SridhamaPrakhyaICON2017.pdf), Prakhya S, Venkataram V, Kalita J. NLPIR, 2018.

[*Learning a Neural-network-based Representation for Open Set Recognition](https://arxiv.org/abs/1802.04365), Hassen M, Chan P K. arXiv, 2018.

[*Unseen Class Discovery in Open-world Classification](https://arxiv.org/abs/1802.04365), Shu L, Xu H, Liu B. arXiv, 2018.


<a name="papers-osr-al"></a>
### Adversarial Learning-based

[*Generative openmax for multi-class open set classification](https://arxiv.org/abs/1707.07418), Ge Z Y, Demyanov S, Chen Z, et al. arXiv, 2017.

[Open-category classification by adversarial sample generation](https://arxiv.org/abs/1705.08722), Yu Y, Qu W Y, Li N, et al. IJCAI, 2017.

[*Open Set Adversarial Examples](https://arxiv.org/abs/1809.02681), Zhedong Z, Liang Z, Zhilan H, et al. arXiv, 2018.

[Open Set Learning with Counterfactual Images](https://link.springer.com/chapter/10.1007/978-3-030-01231-1_38), Neal L, Olson M, Fern X, et al. ECCV, 2018.

[Open-set human activity recognition based on micro-Doppler signatures](https://www.sciencedirect.com/science/article/pii/S003132031830270X), Yang Y, Hou C, Lang Y, et al. Pattern Recognition, 2019.


<a name="papers-osr-evt"></a>
### Extreme Value Theory-based

[The extreme value machine](https://ieeexplore.ieee.org/abstract/document/7932895/), Rudd E M, Jain L P, Scheirer W J, et al. PAMI, 2018.

[*Extreme Value Theory for Open Set Classification-GPD and GEV Classifiers](https://arxiv.org/abs/1808.09902), Vignotto E, Engelke S. arXiv, 2018.



<a name="papers-owr"></a>
## Open World Recognition

[Towards Open World Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Bendale_Towards_Open_World_2015_CVPR_paper.html), Bendale A, Boult T. CVPR, 2015.

[*Online open world recognition](https://arxiv.org/abs/1604.02275), De Rosa R, Mensink T, Caputo B. arXiv, 2016.

[*Open-World Visual Recognition Using Knowledge Graphs](https://arxiv.org/abs/1708.08310), Lonij V, Rawat A, Nicolae M I. arXiv, 2017.

[*Unseen Class Discovery in Open-world Classification](https://arxiv.org/abs/1801.05609), Shu L, Xu H, Liu B. arXiv, 2018.

[The extreme value machine](https://ieeexplore.ieee.org/abstract/document/7932895/), Rudd E M, Jain L P, Scheirer W J, et al. PAMI, 2018.

[*Learning to Accept New Classes without Training](https://arxiv.org/abs/1801.05609), Xu H, Liu B, Shu L, et al. arXiv, 2018.

[ODN: Opening the Deep Network for Open-Set Action Recognition](https://ieeexplore.ieee.org/abstract/document/8486601/), Shi Y, Wang Y, Zou Y, et al. ICME, 2018.

<a name="opensource"></a>
# OpenSource resources

<a name="opensource-osr"></a>
## OpenSource Open Set Recognition

### Traditional Machine Learning Methods-based

| Project |  Language | Model |
| ---  | --- | --- |
|[libsvm-openset](https://github.com/ljain2/libsvm-openset) | C/C++ | [1-vs-Set](https://ieeexplore.ieee.org/abstract/document/6365193), [W-SVM](https://ieeexplore.ieee.org/abstract/document/6809169/), [PI-SVM](https://link.springer.com/chapter/10.1007/978-3-319-10578-9_26) |
|[FittingHyperplaneClassifiers](http://mlcv.ogu.edu.tr/softwarefh.html) | Matlab | [BFHC](https://ieeexplore.ieee.org/abstract/document/7506010/) |
|[EPCC](http://mlcv.ogu.edu.tr/softwareepccisr.html) | Matlab | [EPCC](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w23/Cevikalp_Fast_and_Accurate_ICCV_2017_paper.pdf) |
|[SROSR](https://github.com/hezhangsprinter/SROSR) | Matlab | [SROSR](https://ieeexplore.ieee.org/abstract/document/7577876/)|
|[HPLS-HFCN-openset](https://github.com/rafaelvareto/HPLS-HFCN-openset) | Matlab | [HPLS, HFCN](https://ieeexplore.ieee.org/abstract/document/8272751/)|


### Deep Neural Network-based
| Project |  Language | License |
| ---  | --- | --- |
|[OpenMax](https://github.com/abhijitbendale/OSDN) | Python | [OpenMax](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html)|
|[CNN-for-Sentence-Classification-in-Keras](https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras) | Python | [DOC](https://arxiv.org/abs/1709.08716)|

### Adversarial Learning-based

| Project |  Language | License |
| ---  | --- | --- |
|[counterfactual-open-set](https://github.com/lwneal/counterfactual-open-set) | Python | [Neal](https://link.springer.com/chapter/10.1007/978-3-030-01231-1_38)|
|[ASG](https://github.com/eyounx/ASG) | Python | [ASG](https://arxiv.org/abs/1705.08722)|

### Extreme Value Theory-based
| Project |  Language | License |
| ---  | --- | --- |
|[ExtremeValueMachine](https://github.com/EMRResearch/ExtremeValueMachine) | Python | [EVM](https://ieeexplore.ieee.org/abstract/document/7932895/)|

# License

License [![CCBY-SA](https://i.creativecommons.org/l/by-sa/4.0/80x15.png)]()

To the extent possible under law, [Guangyao Chen](https://github.com/iCGY96/) has waived all copyright and related or neighboring rights to this work.

# Contributing
Please see [CONTRIBUTING](https://github.com/iCGY96/awesome_OpenSet_list/blob/master/contributing.md) for details.
