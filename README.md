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
  	- [Out-of-Distribution](#papers-odd)
  	- [Anomaly Detection](#papers-ad)
  	- [Open Set Domain Adaptation](#papers-osda)
	- [Open World Recognition](#papers-owr)

- [License](#license)

- [Contributing](#contributing)

<a name="tutorials"></a>
# Tutorials

## Tutorial & survey

- [Toward Open Set Recognition](https://ieeexplore.ieee.org/abstract/document/6365193), Scheirer W J, de Rezende Rocha A, Sapkota A, et al. (**PAMI, 2013**).

- [Towards Open World Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Bendale_Towards_Open_World_2015_CVPR_paper.html), Bendale A, Boult T. (**CVPR, 2015**).

- [Lifelong Machine Learning](https://www.cs.uic.edu/~liub/lifelong-machine-learning.html), Zhiyuan Chen and Bing Liu. (**2018**).

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

##### 2019
- [Specialized Support Vector Machines for Open-set Recognition](https://arxiv.org/abs/1606.03802v8), Pedro Ribeiro Mendes Júnior, Terrance E. Boult, Jacques Wainer, Anderson Rocha (**arXiv, 2019**).

<a name="papers-osr-dnn"></a>
### Deep Neural Network-based

##### 2015
- [A bounded neural network for open set recognition](https://ieeexplore.ieee.org/abstract/document/7280680/), Cardoso D O, França F, Gama J. (**IJCNN, 2015**).

##### 2016
- [Towards open set deep networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html), Bendale A, Boult T E. (**CVPR, 2016**). [**[code]**](https://github.com/abhijitbendale/OSDN).

##### 2017
- [Weightless neural networks for open set recognition](https://link.springer.com/article/10.1007/s10994-017-5646-4), Cardoso D O, Gama J, França F M G. (**Machine Learning, 2017**).
- [Adversarial Robustness: Softmax versus Openmax](https://arxiv.org/abs/1708.01697), Rozsa A, Günther M, Boult T E. (**arXiv, 2017**).
- [DOC: Deep open classification of text documents](https://arxiv.org/abs/1709.08716), Shu L, Xu H, Liu B. Doc. (**arXiv, 2017**). [**[code]**](https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras).
- [Generative openmax for multi-class open set classification](https://arxiv.org/abs/1707.07418), Ge Z Y, Demyanov S, Chen Z, et al. (**arXiv, 2017**).
- [Open-category classification by adversarial sample generation](https://arxiv.org/abs/1705.08722), Yu Y, Qu W Y, Li N, et al. (**IJCAI, 2017**). [**[code]**](https://github.com/eyounx/ASG)


##### 2018
- [Open category detection with PAC guarantees](http://proceedings.mlr.press/v80/liu18e/liu18e.pdf), Si Liu, Risheek Garrepalli, Thomas G. Dietterich, Alan Fern, Dan Hendrycks. (**ICML, 2018**). [**[code]**](https://github.com/liusi2019/ocd).
- [Open Set Text Classification using Convolutional Neural Networks](http://www.cs.uccs.edu/~jkalita/papers/2017/SridhamaPrakhyaICON2017.pdf), Prakhya S, Venkataram V, Kalita J. (**NLPIR, 2018**).
- [Learning a Neural-network-based Representation for Open Set Recognition](https://arxiv.org/abs/1802.04365), Hassen M, Chan P K. (**arXiv, 2018**).
- [Unseen Class Discovery in Open-world Classification](https://arxiv.org/abs/1802.04365), Shu L, Xu H, Liu B. (**arXiv, 2018**).
- [Reducing Network Agnostophobia](https://papers.nips.cc/paper/8129-reducing-network-agnostophobia), Akshay Raj Dhamija, Manuel Günther, Terrance E. Boult. (**NeurIPS 2018**). [**[code]**](https://github.com/Vastlab/Reducing-Network-Agnostophobia).
- [Open Set Adversarial Examples](https://arxiv.org/abs/1809.02681), Zhedong Z, Liang Z, Zhilan H, et al. (**arXiv, 2018**).
- [Open Set Learning with Counterfactual Images](https://link.springer.com/chapter/10.1007/978-3-030-01231-1_38), Neal L, Olson M, Fern X, et al. (**ECCV, 2018**). [**[code]**](https://github.com/lwneal/counterfactual-open-set)
- [The extreme value machine](https://ieeexplore.ieee.org/abstract/document/7932895/), Rudd E M, Jain L P, Scheirer W J, et al. (**PAMI, 2018**). [**[code]**](https://github.com/EMRResearch/ExtremeValueMachine)
- [Extreme Value Theory for Open Set Classification-GPD and GEV Classifiers](https://arxiv.org/abs/1808.09902), Vignotto E, Engelke S. (**arXiv, 2018**).

##### 2019
- [The Importance of Metric Learning for Robotic Vision: Open Set Recognition and Active Learning](https://arxiv.org/abs/1902.10363), Benjamin J. Meyer, Tom Drummond. (**ICRA, 2019**).
- [Deep CNN-based Multi-task Learning for Open-Set Recognition](https://arxiv.org/abs/1903.03161), Poojan Oza, Vishal M. Patel. (**arXiv, 2019, Under Review**).
- [Classification-Reconstruction Learning for Open-Set Recognition](https://arxiv.org/abs/1812.04246), Ryota Yoshihashi, Wen Shao, Rei Kawakami, Shaodi You, Makoto Iida, Takeshi Naemura. (**CVPR, 2019**).
- [Alignment Based Matching Networks for One-Shot Classification and Open-Set Recognition](https://arxiv.org/abs/1903.06538), Paresh Malalur, Tommi Jaakkola. (**arXiv, 2019**).
- [Open-Set Recognition Using Intra-Class Splitting](https://arxiv.org/abs/1903.04774), Patrick Schlachter, Yiwen Liao, Bin Yang. (**EUSIPCO, 2019**).
- [Experiments on Open-Set Speaker Identification with Discriminatively Trained Neural Networks](https://arxiv.org/abs/1904.01269v1), Stefano Imoscopi, Volodya Grancharov, Sigurdur Sverrisson, Erlendur Karlsson, Harald Pobloth. (**arXiv, 2019**).
- [Large-Scale Long-Tailed Recognition in an Open World](https://arxiv.org/abs/1904.05160v1), ZiweiLiu, ZhongqiMiao, XiaohangZhan, et al. (**CVPR, Oral, 2019**).[[code]](https://github.com/zhmiao/OpenLongTailRecognition-OLTR)
- [Open Set Recognition Through Deep Neural Network Uncertainty: Does Out-of-Distribution Detection Require Generative Classifiers?](https://arxiv.org/abs/1908.09625v1), Martin Mundt, Iuliia Pliushch, Sagnik Majumder, Visvanathan Ramesh. (**ICCVW, 2019**). [**[code]**](https://github.com/MrtnMndt/Deep_Openset_Recognition_through_Uncertainty)
- [Deep Transfer Learning for Multiple Class Novelty Detection](https://arxiv.org/abs/1903.02196), Pramuditha Perera, Vishal M. Patel. (**CVPR, 2019**). [[code]](https://github.com/PramuPerera/TransferLearningNovelty)
- [From Open Set to Closed Set: Counting Objects by Spatial Divide-and-Conquer](https://arxiv.org/abs/1908.06473), Haipeng Xiong, Hao Lu, Chengxin Liu, Liang Liu, Zhiguo Cao, Chunhua Shen. (**ICCV, 2019**). [[code]](https://github.com/xhp-hust-2018-2011/S-DCNet)
- [Open-set human activity recognition based on micro-Doppler signatures](https://www.sciencedirect.com/science/article/pii/S003132031830270X), Yang Y, Hou C, Lang Y, et al. (**Pattern Recognition, 2019**).
- [C2AE: Class Conditioned Auto-Encoder for Open-set Recognition](https://arxiv.org/abs/1904.01198v1), Poojan Oza, Vishal M Patel. (**CVPR, 2019, oral**).
  
##### 2020
- [Conditional Gaussian Distribution Learning for Open Set Recognition](https://arxiv.org/abs/2003.08823v2), Xin Sun, Zhenning Yang, Chi Zhang, Guohao Peng, Keck-Voon Ling. (**CVPR 2020**). [[code]](https://github.com/BraveGump/CGDL-for-Open-Set-Recognition).
- [Generative-discriminative Feature Representations for Open-set Recognition](https://openaccess.thecvf.com/content_CVPR_2020/papers/Perera_Generative-Discriminative_Feature_Representations_for_Open-Set_Recognition_CVPR_2020_paper.pdf), Pramuditha Perera, Vlad I. Morariu, Rajiv Jain, Varun Manjunatha, Curtis Wigington, Vicente Ordonez, Vishal M. Patel. (**CVPR 2020**).
- [Few-Shot Open-Set Recognition Using Meta-Learning](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Few-Shot_Open-Set_Recognition_Using_Meta-Learning_CVPR_2020_paper.html), Bo Liu, Hao Kang, Haoxiang Li, Gang Hua, Nuno Vasconcelos. (**CVPR 2020**)
- [OpenGAN: Open Set Generative Adversarial Networks](https://arxiv.org/abs/2003.08074v1), Luke Ditria, Benjamin J. Meyer, Tom Drummond. (**arXiv 2020**)
- [Collective decision for open set recognition](https://arxiv.org/abs/1806.11258v4), Chuanxing Geng, Songcan Chen. (**IEEE TKDE, 2020**).
- [One-vs-Rest Network-based Deep Probability Model for Open Set Recognition](https://arxiv.org/abs/2004.08067), Jaeyeon Jang, Chang Ouk Kim. (**arXiv 2020**)
- [Hybrid Models for Open Set Recognition](https://arxiv.org/abs/2003.12506v1), Hongjie Zhang, Ang Li, Jie Guo, Yanwen Guo. (**ECCV 2020**).
- [Learning Open Set Network with Discriminative Reciprocal Points](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480511.pdf), Guangyao Chen, Limeng Qiao, Yemin Shi, Peixi Peng, Jia Li, Tiejun Huang, Shiliang Pu, Yonghong Tian. (**ECCV 2020**)
- [Open-set Adversarial Defense](https://arxiv.org/abs/2009.00814v1), Rui Shao, Pramuditha Perera, Pong C. Yuen, Vishal M. Patel. (**ECCV 2020**)
- [Multi-Task Curriculum Framework for Open-Set Semi-Supervised Learning](https://arxiv.org/abs/2007.11330v1), Qing Yu, Daiki Ikami, Go Irie, Kiyoharu Aizawa. (**ECCV 2020**)
- [Class Anchor Clustering: a Distance-based Loss for Training Open Set Classifiers](https://arxiv.org/abs/2004.02434v2), Dimity Miller, Niko Sünderhauf, Michael Milford, Feras Dayoub. (**ArXiv 2020**)
- [MMF: A loss extension for feature learning in open set recognition](https://arxiv.org/abs/2006.15117v1), Jingyun Jia, Philip K. Chan. (**ArXiv 2020**)
- [Fully Convolutional Open Set Segmentation](https://arxiv.org/abs/2006.14673v1), Hugo Oliveira, Caio Silva, Gabriel L. S. Machado, Keiller Nogueira, Jefersson A. dos Santos. (**ArXiv 2020**)
- [S2OSC: A Holistic Semi-Supervised Approach for Open Set Classification](https://arxiv.org/abs/2008.04662v1). Yang Yang, Zhen-Qiang Sun, Hui Xiong, Jian Yang. (**ArXiv 2020**)
- [Open Set Recognition with Conditional Probabilistic Generative Models](https://arxiv.org/abs/2008.05129v1). Xin Sun, Chi Zhang, Guosheng Lin, Keck-Voon Ling. (**ArXiv 2020**)
- [Convolutional Prototype Network for Open Set Recognition](https://ieeexplore.ieee.org/document/9296325). Hong-Ming Yang, Xu-Yao Zhang, Fei Yin, Qing Yang, Cheng-Lin Liu. (**PAMI 2020**)

<a name="papers-odd"></a>
## Out-of-Distribution

##### 2017
- [A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks](https://arxiv.org/abs/1610.02136). Dan Hendrycks and Kevin Gimpel. (**ICLR, 2017**). [**[code]**](https://github.com/hendrycks/error-detection).

##### 2018
- [Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks](https://arxiv.org/abs/1706.02690). Shiyu Liang, Yixuan Li, R. Srikant. (**ICLR, 2018**). [**[code]**](https://github.com/facebookresearch/odin).
- [Training Confidence-calibrated Classifiers for Detecting Out-of-Distribution Samples](https://arxiv.org/abs/1711.09325). Kimin Lee, Honglak Lee, Kibok Lee, Jinwoo Shin. (**ICLR, 2018**). [**[code]**](https://github.com/alinlab/Confident_classifier)
- [WAIC, but Why? Generative Ensembles for Robust Anomaly Detection](https://arxiv.org/abs/1810.01392). Hyunsun Choi, Eric Jang, Alexander A. Alemi. (**arXiv, 2018**).
- [A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks](https://arxiv.org/abs/1807.03888). Kimin Lee, Kibok Lee, Honglak Lee, Jinwoo Shin. (**NeurIPS, 2018**). [**[code]**](https://github.com/pokaxpoka/deep_Mahalanobis_detector)
- [Predictive uncertainty estimation via prior networks](https://papers.nips.cc/paper/2018/hash/3ea2db50e62ceefceaf70a9d9a56a6f4-Abstract.html). Andrey Malinin, Mark Gales. (**NeurIPS, 2018**). 

##### 2019
- [Deep Anomaly Detection with Outlier Exposure](https://arxiv.org/abs/1812.04606), Dan Hendrycks, Mantas Mazeika, Thomas Dietterich. (**ICLR, 2019**). [**[code]**](https://github.com/hendrycks/outlier-exposure)
- [Do Deep Generative Models Know What They Don't Know?](https://arxiv.org/abs/1810.09136). Eric Nalisnick, Akihiro Matsukawa, Yee Whye Teh, Dilan Gorur, Balaji Lakshminarayanan. (**ICLR, 2019**).
- [Likelihood Ratios for Out-of-Distribution Detection](https://arxiv.org/abs/1906.02845). Jie Ren, Peter J. Liu, Emily Fertig, Jasper Snoek, Ryan Poplin, Mark A. DePristo, Joshua V. Dillon, Balaji Lakshminarayanan. (**NeurIPS, 2019**). [**[code]**](https://github.com/google-research/google-research/tree/master/genomics_ood)
- [Unsupervised Out-of-Distribution Detection by Maximum Classifier Discrepancy](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Unsupervised_Out-of-Distribution_Detection_by_Maximum_Classifier_Discrepancy_ICCV_2019_paper.pdf). Qing Yu, Kiyoharu Aizawa. (**ICCV, 2019**)
- [Why ReLU networks yield high-confidence predictions far away from the training data and how to mitigate the problem](https://arxiv.org/abs/1812.05720). Matthias Hein, Maksym Andriushchenko, Julian Bitterwolf. (**CVPR 2019**). [**[code]**](https://github.com/max-andr/relu_networks_overconfident)
- [Outlier Exposure with Confidence Control for Out-of-Distribution Detection](https://arxiv.org/abs/1906.03509). Aristotelis-Angelos Papadopoulos, Mohammad Reza Rajati, Nazim Shaikh, Jiamian Wang. (**arXiv, 2019**). [**[code]**](https://github.com/nazim1021/OOD-detection-using-OECC)
- [Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty](https://arxiv.org/abs/1906.12340). Dan Hendrycks, Mantas Mazeika, Saurav Kadavath, Dawn Song. (**NeurIPS 2019**). [**[code]**](https://github.com/hendrycks/ss-ood)
- [Evidential Deep Learning to Quantify Classification Uncertainty](https://papers.nips.cc/paper/2018/hash/a981f2b708044d6fb4a71a1463242520-Abstract.html). Murat Sensoy, Lance Kaplan, Melih Kandemir. (**NeurIPS 2019**). [**[code]**](https://github.com/dougbrion/pytorch-classification-uncertainty)

##### 2020
- [Input Complexity and Out-of-distribution Detection with Likelihood-based Generative Models](https://openreview.net/forum?id=SyxIWpVYvr). Joan Serrà, David Álvarez, Vicenç Gómez, Olga Slizovskaia, José F. Núñez, Jordi Luque. (**ICLR, 2020**)
- [Generalized ODIN: Detecting Out-of-distribution Image without Learning from Out-of-distribution Data](https://arxiv.org/abs/2002.11297). Yen-Chang Hsu, Yilin Shen, Hongxia Jin, Zsolt Kira. (**CVPR 2020**)
- [Soft Labeling Affects Out-of-Distribution Detection of Deep Neural Networks](https://arxiv.org/abs/2007.03212v1). Doyup Lee, Yeongjae Cheon. (**ICML 2020 Workshop on Uncertainty and Robustness in Deep Learning**)
- [The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization](https://arxiv.org/abs/2006.16241v1). Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, Dawn Song, Jacob Steinhardt, Justin Gilmer. (**ArXiv 2020**). [**[code]**](https://github.com/hendrycks/imagenet-r)
- [The Effect of Optimization Methods on the Robustness of Out-of-Distribution Detection Approaches](https://arxiv.org/abs/2006.14584v1). Vahdat Abdelzad, Krzysztof Czarnecki, Rick Salay. (**ArXiv 2020**)
- [Density of States Estimation for Out-of-Distribution Detection](https://arxiv.org/abs/2006.09273v2). Warren R. Morningstar, Cusuh Ham, Andrew G. Gallagher, Balaji Lakshminarayanan, Alexander A. Alemi, Joshua V. Dillon. (**ArXiv 2020**)
- [Revisiting One-vs-All Classifiers for Predictive Uncertainty and Out-of-Distribution Detection in Neural Networks](https://arxiv.org/abs/2007.05134v1). Shreyas Padhy, Zachary Nado, Jie Ren, Jeremiah Liu, Jasper Snoek, Balaji Lakshminarayanan. (**ArXiv 2020**)
- [BaCOUn: Bayesian Classifers with Out-of-Distribution Uncertainty](https://arxiv.org/abs/2007.06096v1). Théo Guénais, Dimitris Vamvourellis, Yaniv Yacoby, Finale Doshi-Velez, Weiwei Pan. (**ICML 2020 Workshop on Uncertainty and Robustness in Deep Learning**)
- [Contrastive Training for Improved Out-of-Distribution Detection](https://arxiv.org/abs/2007.05566v1). Jim Winkens, Rudy Bunel, Abhijit Guha Roy, Robert Stanforth, Vivek Natarajan, Joseph R. Ledsam, Patricia MacWilliams, Pushmeet Kohli, Alan Karthikesalingam, Simon Kohl, Taylan Cemgil, S. M. Ali Eslami, Olaf Ronneberger. (**ArXiv 2020**)
- [OOD-MAML: Meta-Learning for Few-Shot Out-of-Distribution Detection and Classification](https://proceedings.neurips.cc/paper/2020/file/28e209b61a52482a0ae1cb9f5959c792-Paper.pdf), Taewon Jeong, Heeyoung Kim. (**NeurIPS 2020**).
- [CSI: Novelty Detection via Contrastive Learning on Distributionally Shifted Instances](https://arxiv.org/abs/2007.08176), Jihoon Tack, Sangwoo Mo, Jongheon Jeong, Jinwoo Shin. (**NeurIPS 2020**). [**[code]**](https://github.com/alinlab/CSI).
- [Iterative VAE as a predictive brain model for out-of-distribution generalization](https://openreview.net/forum?id=jE6SlVTOFPV), Victor Boutin, Aimen Zerroug, Minju Jung, Thomas Serre. (**NeurIPS 2020 Workshop SVRHM**).
- [Certifiably Adversarially Robust Detection of Out-of-Distribution Data](https://arxiv.org/abs/2007.08473v2), Julian Bitterwolf, Alexander Meinke, Matthias Hein. (**NeurIPS 2020**). [**[code]**](https://gitlab.com/Bitterwolf/GOOD).
- [Deep Evidential Regression](https://arxiv.org/abs/1910.02600), Alexander Amini, Wilko Schwarting, Ava Soleimany, Daniela Rus. (**NeurIPS 2020**). [**[code]**](https://github.com/aamini/evidential-deep-learning)

<a name="papers-ad"></a>
## Anomaly Detection
[need to survey more..](https://github.com/hoya012/awesome-anomaly-detection)

##### 2018
- [Deep Anomaly Detection Using Geometric Transformations](https://arxiv.org/abs/1805.10917). Izhak Golan, Ran El-Yaniv. (**NeurIPS, 2018**). [**[code]**](https://github.com/izikgo/AnomalyDetectionTransformations).

##### 2019
- [A Benchmark for Anomaly Segmentation](https://arxiv.org/abs/1911.11132). Dan Hendrycks, Steven Basart, Mantas Mazeika, Mohammadreza Mostajabi, Jacob Steinhardt, Dawn Song. (**arXiv 2019**). [**[code]**](https://github.com/hendrycks/anomaly-seg).

##### 2020
- [Classification-Based Anomaly Detection for General Data](https://openreview.net/forum?id=H1lK_lBtvS). Liron Bergman, Yedid Hoshen. (**ICLR, 2020**).

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
- [Attract or Distract: Exploit the Margin of Open Set](https://arxiv.org/abs/1908.01925v2), Qianyu Feng, Guoliang Kang, Hehe Fan, Yi Yang. (**ICCV 2019**).

##### 2020
- [Collaborative Training of Balanced Random Forests for Open Set Domain Adaptation](https://arxiv.org/abs/2002.03642v1), Jongbin Ryu, Jiun Bae, Jongwoo Lim. (**arXiv 2020**).
- [Mind the Gap: Enlarging the Domain Gap in Open Set Domain Adaptation](https://arxiv.org/abs/2003.03787v2), Dongliang Chang, Aneeshan Sain, Zhanyu Ma, Yi-Zhe Song, Jun Guo. (**arXiv 2020**). [**[code]**](https://github.com/dongliangchang/Mutual-to-Separate/)
- [Towards Inheritable Models for Open-Set Domain Adaptation](https://openaccess.thecvf.com/content_CVPR_2020/html/Kundu_Towards_Inheritable_Models_for_Open-Set_Domain_Adaptation_CVPR_2020_paper.html), Jogendra Nath Kundu, Naveen Venkat, Ambareesh Revanur, Rahul M V, R. Venkatesh Babu. (**CVPR 2020**)
- [Exploring Category-Agnostic Clusters for Open-Set Domain Adaptation](https://openaccess.thecvf.com/content_CVPR_2020/html/Pan_Exploring_Category-Agnostic_Clusters_for_Open-Set_Domain_Adaptation_CVPR_2020_paper.html), Yingwei Pan, Ting Yao, Yehao Li, Chong-Wah Ngo, Tao Mei. (**CVPR 2020**)
- [On the Effectiveness of Image Rotation for Open Set Domain Adaptation](https://arxiv.org/abs/2007.12360v1), Silvia Bucci, Mohammad Reza Loghmani, Tatiana Tommasi. (**ECCV 2020**). [**[code]**](https://github.com/silvia1993/ROS)
- [Open Set Domain Adaptation with Multi-Classifier Adversarial Network](https://arxiv.org/abs/2007.00384v2), Tasfia Shermin, Guojun Lu, Shyh Wei Teng, Manzur Murshed, Ferdous Sohel. (**arXiv 2020**).
- [Bridging the Theoretical Bound and Deep Algorithms for Open Set Domain Adaptation](https://arxiv.org/abs/2006.13022v1), Li Zhong, Zhen Fang, Feng Liu, Bo Yuan, Guangquan Zhang, Jie Lu. (**arXiv 2020**).
- [Progressive Graph Learning for Open-Set Domain Adaptation](https://arxiv.org/abs/2006.12087), Yadan Luo, Zijian Wang, Zi Huang, Mahsa Baktashmotlagh. (**ICML 2020**).
- [Adversarial Network with Multiple Classifiers for Open Set Domain Adaptation](https://arxiv.org/abs/2007.00384), Tasfia Shermin, Guojun Lu, Shyh Wei Teng, Manzur Murshed, Ferdous Sohel. (**IEEE TMM 2020**). [**[code]**](https://github.com/tasfia/DAMC)

<a name="papers-owr"></a>
## Open World Recognition

##### 2015
- [Towards Open World Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Bendale_Towards_Open_World_2015_CVPR_paper.html), Bendale A, Boult T. (**CVPR, 2015**).

##### 2016
- [Learning Cumulatively to Become More Knowledgeable](https://dl.acm.org/citation.cfm?id=2939835), Geli Fei, Shuai Wang, Bing Liu. (**KDD, 2016**).
- [Online open world recognition](https://arxiv.org/abs/1604.02275), De Rosa R, Mensink T, Caputo B. (**arXiv, 2016**).

##### 2017
- [Open-World Visual Recognition Using Knowledge Graphs](https://arxiv.org/abs/1708.08310), Lonij V, Rawat A, Nicolae M I. (**arXiv, 2017**).

##### 2018
- [Unseen Class Discovery in Open-world Classification](https://arxiv.org/abs/1801.05609), Shu L, Xu H, Liu B. (**arXiv, 2018**).
- [The extreme value machine](https://ieeexplore.ieee.org/abstract/document/7932895/), Rudd E M, Jain L P, Scheirer W J, et al. (**PAMI, 2018**).
- [Learning to Accept New Classes without Training](https://arxiv.org/abs/1801.05609), Xu H, Liu B, Shu L, et al. (**arXiv, 2018**).
- [ODN: Opening the Deep Network for Open-Set Action Recognition](https://ieeexplore.ieee.org/abstract/document/8486601/), Shi Y, Wang Y, Zou Y, et al. (**ICME, 2018**).

##### 2019
- [P-ODN: Prototype based Open Deep Network for Open Set Recognition](https://arxiv.org/abs/1905.01851v1), Yu Shu, Yemin Shi, Yaowei Wang, Tiejun Huang, Yonghong Tian. (**arXiv 2019**).
- [Learning and the Unknown: Surveying Steps Toward Open World Recognition](https://www.vast.uccs.edu/~tboult/PAPERS/Learning_and_the_Unknown_Surveying_Steps_Toward_Open_World_Recognition_AAAI19.pdf), Terrance Boult, Steve Cruz, Akshay Dhamija, Manuel Günther, James Henrydoss,
Walter J. Scheirer. (**AAAI, 2019**).
- [Unified Probabilistic Deep Continual Learning through Generative Replay and Open Set Recognition](https://arxiv.org/abs/1905.12019v2). Martin Mundt, Sagnik Majumder, Iuliia Pliushch, Visvanathan Ramesh. (**arXiv 2019**).
- [Open-world Learning and Application to Product Classification](https://arxiv.org/abs/1809.06004). Hu Xu, Bing Liu, Lei Shu, P. Yu. (**WWW 2019**). [**[code]**](https://github.com/howardhsu/Meta-Open-World-Learning)

# License

License [![CCBY-SA](https://i.creativecommons.org/l/by-sa/4.0/80x15.png)]()

To the extent possible under law, [Guangyao Chen](https://github.com/iCGY96/) has waived all and related or neighboring rights to this work.

# Contributing
Please see [CONTRIBUTING](https://github.com/iCGY96/awesome_OpenSet_list/blob/master/contributing.md) for details.

# Contributions
Contributions are most welcome, if you have any suggestions and improvements, please create an issue or raise a pull request.
