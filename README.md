
# Awesome Open Set Recognition list [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

> A curated list of papers & ressources linked to open set recognition, out-of-distribution, open set domain adaptation, and open world recognition

**Note that:**
- **This list is not exhaustive.**
- **Tables use alphabetical order for fairness.**


## Contents

- [Tutorials](#tutorials)

- [Papers](#papers)
	- [Open Set Recognition](#papers-osr)
    	- [Traditional Machine Learning Methods-based](#papers-osr-ml)
    	- [Deep Neural Network-based](#papers-osr-dnn)
    	- [Adversarial Learning-based](#papers-osr-al)
    	- [EVT-based](#papers-osr-evt)
  	- [Out-of-Distribution](#papers-odd)
  	- [Open Set Domain Adaptation](#papers-osda)
	- [Open World Recognition](#papers-owr)

- [License](#license)

- [Contributing](#contributing)

<a name="tutorials"></a>
# Tutorials

## Tutorial & survey

- [Toward Open Set Recognition](https://ieeexplore.ieee.org/abstract/document/6365193), Scheirer W J, de Rezende Rocha A, Sapkota A, et al. (**PAMI, 2013**).

- [Towards Open World Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Bendale_Towards_Open_World_2015_CVPR_paper.html), Bendale A, Boult T. (**CVPR, 2015**).

- [Recent Advances in Open Set Recognition: A Survey](https://arxiv.org/abs/1811.08581), Geng C, Huang S, Chen S. (**arXiv, 2018**).

- [Recent Advances in Open Set Recognition: A Survey v2](https://arxiv.org/abs/1811.08581v2), Chuanxing Geng, Sheng-jun Huang, Songcan Chen. (**arXiv, 2019**).

<a name="papers"></a>
# Papers

<a name="papers-osr"></a>
## Open Set Recognition

<a name="papers-osr-ml"></a>
### Traditional Machine Learning Methods-based

##### 2013
- [Toward Open Set Recognition](https://ieeexplore.ieee.org/abstract/document/6365193), Scheirer W J, de Rezende Rocha A, Sapkota A, et al. (**PAMI, 2013**).[**[code]**](https://github.com/ljain2/libsvm-openset).

##### 2014
- [Probability models for open set recognition](https://ieeexplore.ieee.org/abstract/document/6809169/), Scheirer W J, Jain L P, Boult T E. (**PAMI, 2014**). [**[code]**](https://github.com/ljain2/libsvm-openset).
- [Multi-class open set recognition using probability of inclusion](https://link.springer.com/chapter/10.1007/978-3-319-10578-9_26), Jain L P, Scheirer W J, Boult T E. (**ECCV, 2014**). [**[code]**](https://github.com/ljain2/libsvm-openset).

##### 2016
- [Breaking the closed world assumption in text classification](http://www.aclweb.org/anthology/N16-1061), Fei G, Liu B. (**NAACL, 2016**).

##### 2017
- [Sparse representation-based open set recognition](https://ieeexplore.ieee.org/abstract/document/7577876/), Zhang H, Patel V M. (**PAMI, 2017**).
- [Best fitting hyperplanes for classification](https://ieeexplore.ieee.org/abstract/document/7506010/), Cevikalp H. (**PAMI, 2017**). [**[code]**](http://mlcv.ogu.edu.tr/softwarefh.html).
- [Polyhedral conic classifiers for visual object detection and classification](http://openaccess.thecvf.com/content_cvpr_2017/papers/Cevikalp_Polyhedral_Conic_Classifiers_CVPR_2017_paper.pdf), Cevikalp H, Triggs B. Rigling B D. (**CVPR, 2017**).
- [Fast and Accurate Face Recognition with Image Sets](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w23/Cevikalp_Fast_and_Accurate_ICCV_2017_paper.pdf), Cevikalp H, Yavuz H S. (**ICCVW, 2017**). [**[code]**](https://github.com/hezhangsprinter/SROSR)
- [Nearest neighbors distance ratio open-set classifier](https://link.springer.com/article/10.1007/s10994-016-5610-8), Júnior P R M, de Souza R M, Werneck R O, et al. (**Machine Learning, 2017**).

##### 2018
- [Data-Fusion Techniques for Open-Set Recognition Problems](https://ieeexplore.ieee.org/abstract/document/8332940/), Neira M A C, Júnior P R M, Rocha A, et al. (**IEEE Access, 2018**).
- [Towards open-set face recognition using hashing functions](https://ieeexplore.ieee.org/abstract/document/8272751/), Vareto R, Silva S, Costa F, et al. (**IJCB, 2018**). [**[code]**](https://github.com/rafaelvareto/HPLS-HFCN-openset).
- [Learning to Separate Domains in Generalized Zero-Shot and Open Set Learning: a probabilistic perspective](https://arxiv.org/abs/1810.07368), Hanze Dong, Yanwei Fu, Leonid Sigal, Sung Ju Hwang, Yu-Gang Jiang, Xiangyang Xue. (**arXiv, 2018**).

<a name="papers-osr-dnn"></a>
### Deep Neural Network-based

##### 2015
- [A bounded neural network for open set recognition](https://ieeexplore.ieee.org/abstract/document/7280680/), Cardoso D O, França F, Gama J. (**IJCNN, 2015**).

##### 2016
- [Towards open set deep networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html), Bendale A, Boult T E. (**CVPR, 2016**). [**[code]**](https://github.com/abhijitbendale/OSDN).

##### 2017
- [Weightless neural networks for open set recognition](https://link.springer.com/article/10.1007/s10994-017-5646-4), Cardoso D O, Gama J, França F M G. (**Machine Learning, 2017**).
- [*Adversarial Robustness: Softmax versus Openmax](https://arxiv.org/abs/1708.01697), Rozsa A, Günther M, Boult T E. (**arXiv, 2017**).
- [*DOC: Deep open classification of text documents](https://arxiv.org/abs/1709.08716), Shu L, Xu H, Liu B. Doc. (**arXiv, 2017**). [**[code]**](https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras).

##### 2018
- [Open category detection with PAC guarantees](http://proceedings.mlr.press/v80/liu18e/liu18e.pdf), Si Liu, Risheek Garrepalli, Thomas G. Dietterich, Alan Fern, Dan Hendrycks. (**ICML, 2018**). [**[code]**](https://github.com/liusi2019/ocd).
- [Open Set Text Classification using Convolutional Neural Networks](http://www.cs.uccs.edu/~jkalita/papers/2017/SridhamaPrakhyaICON2017.pdf), Prakhya S, Venkataram V, Kalita J. (**NLPIR, 2018**).
- [*Learning a Neural-network-based Representation for Open Set Recognition](https://arxiv.org/abs/1802.04365), Hassen M, Chan P K. (**arXiv, 2018**).
- [*Unseen Class Discovery in Open-world Classification](https://arxiv.org/abs/1802.04365), Shu L, Xu H, Liu B. (**arXiv, 2018**).

##### 2019
- [The Importance of Metric Learning for Robotic Vision: Open Set Recognition and Active Learning](https://arxiv.org/abs/1902.10363), Benjamin J. Meyer, Tom Drummond. (**ICRA, 2019**).
- [*Deep CNN-based Multi-task Learning for Open-Set Recognition](https://arxiv.org/abs/1903.03161), Poojan Oza, Vishal M. Patel. (**arXiv, 2019, Under Review**).
- [Classification-Reconstruction Learning for Open-Set Recognition](https://arxiv.org/abs/1812.04246), Ryota Yoshihashi, Wen Shao, Rei Kawakami, Shaodi You, Makoto Iida, Takeshi Naemura. (**CVPR, 2019**).
- [*Alignment Based Matching Networks for One-Shot Classification and Open-Set Recognition](https://arxiv.org/abs/1903.06538), Paresh Malalur, Tommi Jaakkola. (**arXiv, 2019**).
- [*Open-Set Recognition Using Intra-Class Splitting](https://arxiv.org/abs/1903.04774), Patrick Schlachter, Yiwen Liao, Bin Yang. (**EUSIPCO, 2019**).
- [C2AE: Class Conditioned Auto-Encoder for Open-set Recognition](https://arxiv.org/abs/1904.01198v1), Poojan Oza, Vishal M Patel. (**CVPR, 2019, oral**).
- [*Experiments on Open-Set Speaker Identification with Discriminatively Trained Neural Networks](https://arxiv.org/abs/1904.01269v1), Stefano Imoscopi, Volodya Grancharov, Sigurdur Sverrisson, Erlendur Karlsson, Harald Pobloth. (**arXiv, 2019**).
- [Unsupervised Open Domain Recognition by Semantic Discrepancy Minimization](https://arxiv.org/abs/1904.08631v1), Junbao Zhuo, Shuhui Wang, Shuhao Cui, Qingming Huang (**CVPR, 2019**).
- [Large-Scale Long-Tailed Recognition in an Open World](https://arxiv.org/abs/1904.05160v1), ZiweiLiu, ZhongqiMiao, XiaohangZhan, et al. (**CVPR, Oral, 2019**).[[code]](https://github.com/zhmiao/OpenLongTailRecognition-OLTR)
- [Unified Probabilistic Deep Continual Learning through Generative Replay and Open Set Recognition](https://arxiv.org/abs/1905.12019v1), Martin Mundt, Sagnik Majumder, Iuliia Pliushch, Visvanathan Ramesh. (**arXiv, 2019**).
- [Specialized Support Vector Machines for Open-set Recognition](https://arxiv.org/abs/1606.03802v8), Pedro Ribeiro Mendes Júnior, Terrance E. Boult, Jacques Wainer, Anderson Rocha (**arXiv, 2019**).

<a name="papers-osr-al"></a>
### Adversarial Learning-based

##### 2017
- [*Generative openmax for multi-class open set classification](https://arxiv.org/abs/1707.07418), Ge Z Y, Demyanov S, Chen Z, et al. (**arXiv, 2017**).
- [Open-category classification by adversarial sample generation](https://arxiv.org/abs/1705.08722), Yu Y, Qu W Y, Li N, et al. (**IJCAI, 2017**). [**[code]**](https://github.com/eyounx/ASG)

##### 2018
- [*Open Set Adversarial Examples](https://arxiv.org/abs/1809.02681), Zhedong Z, Liang Z, Zhilan H, et al. (**arXiv, 2018**).
- [Open Set Learning with Counterfactual Images](https://link.springer.com/chapter/10.1007/978-3-030-01231-1_38), Neal L, Olson M, Fern X, et al. (**ECCV, 2018**). [**[code]**](https://github.com/lwneal/counterfactual-open-set)

##### 2019
- [Open-set human activity recognition based on micro-Doppler signatures](https://www.sciencedirect.com/science/article/pii/S003132031830270X), Yang Y, Hou C, Lang Y, et al. (**Pattern Recognition, 2019**).

<a name="papers-osr-evt"></a>
### Extreme Value Theory-based

##### 2018
- [The extreme value machine](https://ieeexplore.ieee.org/abstract/document/7932895/), Rudd E M, Jain L P, Scheirer W J, et al. (**PAMI, 2018**). [**[code]**](https://github.com/EMRResearch/ExtremeValueMachine)
- [*Extreme Value Theory for Open Set Classification-GPD and GEV Classifiers](https://arxiv.org/abs/1808.09902), Vignotto E, Engelke S. (**arXiv, 2018**).

##### 2019
- [Collective decision for open set recognition](https://arxiv.org/abs/1806.11258v4), Chuanxing Geng, Songcan Chen. (**arXiv, 2019**).


<a name="papers-odd"></a>
## Out-of-Distribution

##### 2017
- [A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks](https://arxiv.org/abs/1610.02136), Dan Hendrycks and Kevin Gimpel. (**ICLR, 2017**). [**[code]**](https://github.com/hendrycks/error-detection).

##### 2018
- [Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks](https://arxiv.org/abs/1706.02690), Shiyu Liang, Yixuan Li, R. Srikant. (**ICLR, 2018**). [**[code]**](https://github.com/facebookresearch/odin).
- [Training Confidence-calibrated Classifiers for Detecting Out-of-Distribution Samples](https://arxiv.org/abs/1711.09325), Kimin Lee, Honglak Lee, Kibok Lee, Jinwoo Shin. (**ICLR, 2018**). [**[code]**](https://github.com/alinlab/Confident_classifier)

##### 2019
- [Deep Anomaly Detection with Outlier Exposure](https://arxiv.org/abs/1812.04606), Dan Hendrycks, Mantas Mazeika, Thomas Dietterich. (**ICLR, 2019**). [**[code]**](https://github.com/hendrycks/outlier-exposure)
- [Likelihood Ratios for Out-of-Distribution Detection](https://arxiv.org/abs/1906.02845), Jie Ren, Peter J. Liu, Emily Fertig, Jasper Snoek, Ryan Poplin, Mark A. DePristo, Joshua V. Dillon, Balaji Lakshminarayanan. (**NeurIPS 2019**) 

<a name="papers-osda"></a>
## Open Set Domain Adaptation

##### 2017
- [Open Set Domain Adaptation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Busto_Open_Set_Domain_ICCV_2017_paper.pdf), Pau Panareda Busto, Juergen Gall. (**ICCV 2017**).
- [Label Efficient Learning of Transferable Representations across Domains and Tasks](https://arxiv.org/abs/1712.00123), Zelun Luo, Yuliang Zou, Judy Hoffman, Li Fei-Fei. (**NeurIPS 2017**).

##### 2018
- [Open set domain adaptation by backpropagation](https://arxiv.org/abs/1804.10427), Kuniaki Saito, Shohei Yamamoto, Yoshitaka Ushiku, Tatsuya Harada. (**ECCV 2018**).

##### 2019
- [Separate to Adapt: Open Set Domain Adaptation via Progressive Separation](http://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Separate_to_Adapt_Open_Set_Domain_Adaptation_via_Progressive_Separation_CVPR_2019_paper.pdf). Hong Liu, Zhangjie Cao, Mingsheng Long, Jianmin Wang, Qiang Yang. (**CVPR 2019**).
- [Unsupervised Open Domain Recognition by Semantic Discrepancy Minimization](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhuo_Unsupervised_Open_Domain_Recognition_by_Semantic_Discrepancy_Minimization_CVPR_2019_paper.pdf). Junbao Zhuo, Shuhui Wang, Shuhao Cui, Qingming Huang. (**CVPR 2019**).
- [Weakly Supervised Open-Set Domain Adaptation by Dual-Domain Collaboration](http://openaccess.thecvf.com/content_CVPR_2019/html/Tan_Weakly_Supervised_Open-Set_Domain_Adaptation_by_Dual-Domain_Collaboration_CVPR_2019_paper.html). Shuhan Tan, Jiening Jiao, Wei-Shi Zheng. (**CVPR 2019**). [**[code]**](https://github.com/junbaoZHUO/UODTN)
- [Learning Factorized Representations for Open-set Domain Adaptation](https://openreview.net/forum?id=SJe3HiC5KX), Mahsa Baktashmotlagh, Masoud Faraki, Tom Drummond, Mathieu Salzmann. (**ICLR 2019**).
- [Known-class Aware Self-ensemble for Open Set Domain Adaptation](https://arxiv.org/abs/1905.01068v1), Qing Lian, Wen Li, Lin Chen, Lixin Duan. (**arXiv 2019**).
- [Open Set Domain Adaptation: Theoretical Bound and Algorithm](https://arxiv.org/abs/1907.08375v1), Zhen Fang, Jie Lu, Feng Liu, Junyu Xuan, Guangquan Zhang. (**arXiv 2019**).
- [Open Set Domain Adaptation for Image and Action Recognition](https://arxiv.org/abs/1907.12865v1), Pau Panareda Busto, Ahsan Iqbal, Juergen Gall. (**arXiv 2019**).
- [Attract or Distract: Exploit the Margin of Open Set](https://arxiv.org/abs/1908.01925v1), Qianyu Feng, Guoliang Kang, Hehe Fan, Yi Yang. (**arXiv 2019**).


<a name="papers-owr"></a>
## Open World Recognition

##### 2015
- [Towards Open World Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Bendale_Towards_Open_World_2015_CVPR_paper.html), Bendale A, Boult T. (**CVPR, 2015**).

##### 2016
- [*Online open world recognition](https://arxiv.org/abs/1604.02275), De Rosa R, Mensink T, Caputo B. (**arXiv, 2016**).

##### 2017
- [*Open-World Visual Recognition Using Knowledge Graphs](https://arxiv.org/abs/1708.08310), Lonij V, Rawat A, Nicolae M I. (**arXiv, 2017**).

##### 2018
- [*Unseen Class Discovery in Open-world Classification](https://arxiv.org/abs/1801.05609), Shu L, Xu H, Liu B. (**arXiv, 2018**).
- [The extreme value machine](https://ieeexplore.ieee.org/abstract/document/7932895/), Rudd E M, Jain L P, Scheirer W J, et al. (**PAMI, 2018**).
- [*Learning to Accept New Classes without Training](https://arxiv.org/abs/1801.05609), Xu H, Liu B, Shu L, et al. (**arXiv, 2018**).
- [ODN: Opening the Deep Network for Open-Set Action Recognition](https://ieeexplore.ieee.org/abstract/document/8486601/), Shi Y, Wang Y, Zou Y, et al. (**ICME, 2018**).

##### 2019
- [P-ODN: Prototype based Open Deep Network for Open Set Recognition](https://arxiv.org/abs/1905.01851v1), Yu Shu, Yemin Shi, Yaowei Wang, Tiejun Huang, Yonghong Tian. (**arXiv 2019**).
- [Learning and the Unknown: Surveying Steps Toward Open World Recognition](https://www.vast.uccs.edu/~tboult/PAPERS/Learning_and_the_Unknown_Surveying_Steps_Toward_Open_World_Recognition_AAAI19.pdf), Terrance Boult, Steve Cruz, Akshay Dhamija, Manuel Günther, James Henrydoss,
Walter J. Scheirer. (**AAAI, 2019**).
- [Unified Probabilistic Deep Continual Learning through Generative Replay and Open Set Recognition](https://arxiv.org/abs/1905.12019). Martin Mundt, Sagnik Majumder, Iuliia Pliushch, Visvanathan Ramesh. (**arXiv 2019**).

# License

License [![CCBY-SA](https://i.creativecommons.org/l/by-sa/4.0/80x15.png)]()

To the extent possible under law, [Guangyao Chen](https://github.com/iCGY96/) has waived all https://arxiv.org/abs/1904.05160v1 and related or neighboring rights to this work.

# Contributing
Please see [CONTRIBUTING](https://github.com/iCGY96/awesome_OpenSet_list/blob/master/contributing.md) for details.
