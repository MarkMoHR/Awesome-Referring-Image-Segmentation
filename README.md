# Awesome-Referring-Image-Segmentation

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)


A collection of referring image segmentation papers and datasets.

> Feel free to create a PR or an issue.

![examples](teaser.png)


**Outline**

- [Awesome-Referring-Image-Segmentation](#awesome-referring-image-segmentation)
  - [1. Datasets](#1-datasets)
  - [2. Traditional Referring Image Segmentation](#2-traditional-referring-image-segmentation)
  - [3. Interactive Referring Image Segmentation](#3-interactive-referring-image-segmentation)
  - [4. Referring Video Segmentation](#4-referring-video-segmentation)
  - [5. Referring 3D Instance Segmentation](#5-referring-3d-instance-segmentation)


## 1. Datasets

| Short name | Paper | Source | Code/Project Link  |
| --- | --- | --- | --- |
| ReferIt | [Referit game: Referring to objects in photographs of natural scenes](https://www.aclweb.org/anthology/D14-1086) | EMNLP 2014 | [[project]](http://tamaraberg.com/referitgame/) |
| Google-Ref | [Generation and comprehension of unambiguous object descriptions](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Mao_Generation_and_Comprehension_CVPR_2016_paper.pdf) | CVPR 2016 | [[dataset]](https://github.com/mjhucla/Google_Refexp_toolbox) |
| UNC | [Modeling context in referring expressions](https://arxiv.org/pdf/1608.00272) | ECCV 2016 | [[dataset]](https://github.com/lichengunc/refer) |
| UNC+ | [Modeling context in referring expressions](https://arxiv.org/pdf/1608.00272) | ECCV 2016 | [[dataset]](https://github.com/lichengunc/refer) |
| CLEVR-Ref+ | [CLEVR-Ref+: Diagnosing Visual Reasoning with Referring Expressions](https://arxiv.org/pdf/1901.00850.pdf) | CVPR 2019 | [[project]](https://cs.jhu.edu/~cxliu/2019/clevr-ref+) |
| VGPhraseCut | [PhraseCut: Language-based Image Segmentation in the Wild](https://people.cs.umass.edu/~smaji/papers/phrasecut+supp-cvpr20.pdf) | CVPR 2020 | [[project]](https://people.cs.umass.edu/~chenyun/phrasecut/) |
| ScanRefer | [ScanRefer: 3D Object Localization in RGB-D Scans using Natural Language](https://arxiv.org/pdf/1912.08830) | ECCV 2020 | [[project]](https://daveredrum.github.io/ScanRefer/) |
| ClevrTex | [ClevrTex: A Texture-Rich Benchmark for Unsupervised Multi-Object Segmentation](https://arxiv.org/abs/2111.10265) | NeurIPS 2021 | [[project]](https://www.robots.ox.ac.uk/~vgg/data/clevrtex/) |
| gRefCOCO | [GRES: Generalized Referring Expression Segmentation](https://arxiv.org/abs/2306.00968) | CVPR 2023 | [[dataset]](https://github.com/henghuiding/gRefCOCO) [[project]](https://henghuiding.github.io/GRES/) |

---

## 2. Traditional Referring Image Segmentation

| Short name | Paper | Source | Code/Project Link  |
| --- | --- | --- | --- |
| ETRIS | [Bridging Vision and Language Encoders: Parameter-Efficient Tuning for Referring Image Segmentation](https://arxiv.org/abs/2307.11545) | ICCV 2023 | [[code]](https://github.com/kkakkkka/ETRIS) |
| SEEM | [Segment Everything Everywhere All at Once](https://arxiv.org/abs/2304.06718) | arxiv 23.04 |  |
| WiCo | [WiCo: Win-win Cooperation of Bottom-up and Top-down Referring Image Segmentation](https://arxiv.org/abs/2306.10750) | IJCAI 2023 |  |
| M3Att | [Multi-Modal Mutual Attention and Iterative Interaction for Referring Image Segmentation](https://ieeexplore.ieee.org/abstract/document/10132374) | TIP 2023 |  |
| SADLR | [Semantics-Aware Dynamic Localization and Refinement for Referring Image Segmentation](https://arxiv.org/abs/2303.06345) | AAAI 2023 |  |
| Partial-RES | [Learning to Segment Every Referring Object Point by Point](https://openaccess.thecvf.com/content/CVPR2023/papers/Qu_Learning_To_Segment_Every_Referring_Object_Point_by_Point_CVPR_2023_paper.pdf) | CVPR 2023 | [[code]](https://github.com/qumengxue/Partial-RES) |
| MCRES | [Meta Compositional Referring Expression Segmentation](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Meta_Compositional_Referring_Expression_Segmentation_CVPR_2023_paper.pdf) | CVPR 2023 |  |
| Global-Local CLIP | [Zero-shot Referring Image Segmentation with Global-Local Context Features](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Zero-Shot_Referring_Image_Segmentation_With_Global-Local_Context_Features_CVPR_2023_paper.pdf) | CVPR 2023 | [[code]](https://github.com/Seonghoon-Yu/Zero-shot-RIS) |
| PolyFormer | [PolyFormer: Referring Image Segmentation as Sequential Polygon Generation](https://arxiv.org/abs/2302.07387) | CVPR 2023 | [[code]](https://github.com/amazon-science/polygon-transformer) [[project]](https://polyformer.github.io/) |
| GRES | [GRES: Generalized Referring Expression Segmentation](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_GRES_Generalized_Referring_Expression_Segmentation_CVPR_2023_paper.pdf) | CVPR 2023 | [[code]](https://github.com/henghuiding/ReLA) [[dataset]](https://github.com/henghuiding/gRefCOCO) [[project]](https://henghuiding.github.io/GRES/) |
| CGFormer | [Contrastive Grouping with Transformer for Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2023/papers/Tang_Contrastive_Grouping_With_Transformer_for_Referring_Image_Segmentation_CVPR_2023_paper.pdf) | CVPR 2023 | [[code]](https://github.com/Toneyaya/CGFormer) |
|  | [Learning From Box Annotations for Referring Image Segmentation](https://ieeexplore.ieee.org/abstract/document/9875225) | TNNLS 2022 | [[code]](https://github.com/fengguang94/Weakly-Supervised-RIS) |
|  | [Instance-Specific Feature Propagation for Referring Segmentation](https://ieeexplore.ieee.org/abstract/document/9745353) | TMM 2022 |  |
| LAVT | [LAVT: Language-Aware Vision Transformer for Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_LAVT_Language-Aware_Vision_Transformer_for_Referring_Image_Segmentation_CVPR_2022_paper.pdf) | CVPR 2022 | [[code]](https://github.com/yz93/LAVT-RIS) |
| CRIS | [CRIS: CLIP-Driven Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_CRIS_CLIP-Driven_Referring_Image_Segmentation_CVPR_2022_paper.pdf) | CVPR 2022 | [[code]](https://github.com/DerrickWang005/CRIS.pytorch) |
| ReSTR | [ReSTR: Convolution-free Referring Image Segmentation Using Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_ReSTR_Convolution-Free_Referring_Image_Segmentation_Using_Transformers_CVPR_2022_paper.pdf) | CVPR 2022 | [[project]](http://cvlab.postech.ac.kr/research/restr/) |
| VLT | [Vision-Language Transformer and Query Generation for Referring Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Ding_Vision-Language_Transformer_and_Query_Generation_for_Referring_Segmentation_ICCV_2021_paper.pdf) | ICCV 2021 | [[code]](https://github.com/henghuiding/Vision-Language-Transformer) |
| MDETR | [MDETR - Modulated Detection for End-to-End Multi-Modal Understanding](https://openaccess.thecvf.com/content/ICCV2021/papers/Kamath_MDETR_-_Modulated_Detection_for_End-to-End_Multi-Modal_Understanding_ICCV_2021_paper.pdf) | ICCV 2021 | [[code]](https://github.com/ashkamath/mdetr) [[project]](https://ashkamath.github.io/mdetr_page/) |
| CEFNet | [Encoder Fusion Network with Co-Attention Embedding for Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Feng_Encoder_Fusion_Network_With_Co-Attention_Embedding_for_Referring_Image_Segmentation_CVPR_2021_paper.pdf) | CVPR 2021 | [[code]](https://github.com/fengguang94/CEFNet) |
| BUSNet | [Bottom-Up Shift and Reasoning for Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Bottom-Up_Shift_and_Reasoning_for_Referring_Image_Segmentation_CVPR_2021_paper.pdf) | CVPR 2021 | [[code]](https://github.com/incredibleXM/BUSNet) |
| LTS | [Locate then Segment: A Strong Pipeline for Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Jing_Locate_Then_Segment_A_Strong_Pipeline_for_Referring_Image_Segmentation_CVPR_2021_paper.pdf) | CVPR 2021 |  |
| TV-Net | [Two-stage Visual Cues Enhancement Network for Referring Image Segmentation](https://arxiv.org/abs/2110.04435) | ACM MM 2021 | [[code]](https://github.com/sxjyjay/tv-net) |
| CGAN | [Cascade Grouped Attention Network for Referring Expression Segmentation](https://dl.acm.org/doi/abs/10.1145/3394171.3414006) | ACM MM 2020 |  |
| LSCM | [Linguistic Structure Guided Context Modeling for Referring Image Segmentation](http://colalab.org/media/paper/Linguistic_Structure_Guided_Context_Modeling_for_Referring_Image_Segmentation.pdf) | ECCV 2020 |  |
| CMPC-Refseg | [Referring Image Segmentation via Cross-Modal Progressive Comprehension](http://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_Referring_Image_Segmentation_via_Cross-Modal_Progressive_Comprehension_CVPR_2020_paper.pdf) | CVPR 2020 | [[code]](https://github.com/spyflying/CMPC-Refseg) |
| BRINet | [Bi-directional Relationship Inferring Network for Referring Image Segmentation](http://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_Bi-Directional_Relationship_Inferring_Network_for_Referring_Image_Segmentation_CVPR_2020_paper.pdf) | CVPR 2020 | [[code]](https://github.com/fengguang94/CVPR2020-BRINet) |
| PhraseCut | [PhraseCut: Language-based Image Segmentation in the Wild](https://people.cs.umass.edu/~smaji/papers/phrasecut+supp-cvpr20.pdf) | CVPR 2020 | [[code]](https://github.com/ChenyunWu/PhraseCutDataset) [[project]](https://people.cs.umass.edu/~chenyun/phrasecut/) |
| MCN | [Multi-task Collaborative Network for Joint Referring Expression Comprehension and Segmentation](https://arxiv.org/abs/2003.08813) | CVPR 2020 | [[code]](https://github.com/luogen1996/MCN) |
| - | [Dual Convolutional LSTM Network for Referring Image Segmentation](https://arxiv.org/abs/2001.11561) | TMM 2020 |  |
| STEP | [See-Through-Text Grouping for Referring Image Segmentation](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_See-Through-Text_Grouping_for_Referring_Image_Segmentation_ICCV_2019_paper.pdf) | ICCV 2019 |  |
| lang2seg | [Referring Expression Object Segmentation with Caption-Aware Consistency](https://arxiv.org/pdf/1910.04748.pdf) | BMVC 2019 | [[code]](https://github.com/wenz116/lang2seg) |
| CMSA | [Cross-Modal Self-Attention Network for Referring Image Segmentation](https://arxiv.org/pdf/1904.04745.pdf) | CVPR 2019 | [[code]](https://github.com/lwye/CMSA-Net) |
| KWA | [Key-Word-Aware Network for Referring Expression Image Segmentation](http://openaccess.thecvf.com/content_ECCV_2018/papers/Hengcan_Shi_Key-Word-Aware_Network_for_ECCV_2018_paper.pdf) | ECCV 2018 | [[code]](https://github.com/shihengcan/key-word-aware-network-pycaffe) |
| DMN | [Dynamic Multimodal Instance Segmentation Guided by Natural Language Queries](http://openaccess.thecvf.com/content_ECCV_2018/papers/Edgar_Margffoy-Tuay_Dynamic_Multimodal_Instance_ECCV_2018_paper.pdf) | ECCV 2018 | [[code]](https://github.com/BCV-Uniandes/DMS) |
| RRN | [Referring Image Segmentation via Recurrent Refinement Networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Referring_Image_Segmentation_CVPR_2018_paper.pdf) | CVPR 2018 | [[code]](https://github.com/liruiyu/referseg_rrn) |
| MAttNet | [MAttNet: Modular Attention Network for Referring Expression Comprehension](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_MAttNet_Modular_Attention_CVPR_2018_paper.pdf) | CVPR 2018 | [[code]](https://github.com/lichengunc/MAttNet) [[Demo]](http://vision2.cs.unc.edu/refer/comprehension) |
| RMI | [Recurrent Multimodal Interaction for Referring Image Segmentation](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Recurrent_Multimodal_Interaction_ICCV_2017_paper.pdf) | ICCV 2017 | [[code]](https://github.com/chenxi116/TF-phrasecut-public) |
| LSTM-CNN | [Segmentation from natural language expressions](https://arxiv.org/pdf/1603.06180.pdf) | ECCV 2016 | [[code]](https://github.com/ronghanghu/text_objseg) [[project]](http://ronghanghu.com/text_objseg/) |


## 3. Interactive Referring Image Segmentation

| Short name | Paper | Source | Code/Project Link  |
| --- | --- | --- | --- |
| PhraseClick | [PhraseClick: Toward Achieving Flexible Interactive Segmentation by Phrase and Click](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480426.pdf) | ECCV 2020 |  |


## 4. Referring Video Segmentation

| Short name | Paper | Source | Code/Project Link  |
| --- | --- | --- | --- |
|  | [Video Object Segmentation with Language Referring Expressions](https://link.springer.com/chapter/10.1007/978-3-030-20870-7_8) | ACCV 2018 |  |
| RefVOS | [RefVOS: A Closer Look at Referring Expressions for Video Object Segmentation](https://arxiv.org/abs/2010.00263) | arxiv 20.10 |  |
| URVOS | [URVOS: Unified Referring Video Object Segmentation Network with a Large-Scale Benchmark](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600205.pdf) | ECCV 2020 | [[code]](https://github.com/skynbe/Refer-Youtube-VOS) |
| YOFO | [You Only Infer Once: Cross-Modal Meta-Transfer for Referring Video Object Segmentation](https://www.aaai.org/AAAI22Papers/AAAI-1100.LiD.pdf) | AAAI 2022 | |
| MTTR | [End-to-End Referring Video Object Segmentation with Multimodal Transformers](https://arxiv.org/abs/2111.14821) | CVPR 2022 | [[code]](https://github.com/mttr2021/MTTR) |
| ReferFormer | [Language as Queries for Referring Video Object Segmentation](https://arxiv.org/abs/2201.00487) | CVPR 2022 | [[code]](https://github.com/wjn922/ReferFormer) |
| LBDT | [Language-Bridged Spatial-Temporal Interaction for Referring Video Object Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Ding_Language-Bridged_Spatial-Temporal_Interaction_for_Referring_Video_Object_Segmentation_CVPR_2022_paper.pdf) | CVPR 2022 | [[code]](https://github.com/dzh19990407/LBDT) |
|  | [Multi-Level Representation Learning with Semantic Alignment for Referring Video Object Segmentation](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_Multi-Level_Representation_Learning_With_Semantic_Alignment_for_Referring_Video_Object_CVPR_2022_paper.pdf) | CVPR 2022 |  |
| OnlineRefer | [OnlineRefer: A Simple Online Baseline for Referring Video Object Segmentation](https://arxiv.org/abs/2307.09356) | ICCV 2023 | [[code]](https://github.com/wudongming97/OnlineRefer) |


## 5. Referring 3D Instance Segmentation

| Short name | Paper | Source | Code/Project Link  |
| --- | --- | --- | --- |
| TGNN | [Text-Guided Graph Neural Networks for Referring 3D Instance Segmentation](https://www.aaai.org/AAAI21Papers/AAAI-4433.HuangP.pdf) | AAAI 2021 |  |
| InstanceRefer | [InstanceRefer: Cooperative Holistic Understanding for Visual Grounding on Point Clouds through Instance Multi-level Contextual Referring](https://arxiv.org/pdf/2103.01128.pdf) | ICCV 2021 | [[code]](https://github.com/CurryYuan/InstanceRefer) |

