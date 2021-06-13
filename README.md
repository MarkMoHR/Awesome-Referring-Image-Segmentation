# Awesome-Referring-Image-Segmentation

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)


A collection of referring image segmentation papers and datasets.

> Feel free to create a PR or an issue.

![examples](https://github.com/MarkMoHR/Awesome-Referring-Image-Segmentation/blob/master/teaser.png)


**Outline**

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


---

## 2. Traditional Referring Image Segmentation

| Short name | Paper | Source | Code/Project Link  |
| --- | --- | --- | --- |
| EFN | [Encoder Fusion Network with Co-Attention Embedding for Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Feng_Encoder_Fusion_Network_With_Co-Attention_Embedding_for_Referring_Image_Segmentation_CVPR_2021_paper.pdf) | CVPR 2021 |  |
| BUSNet | [Bottom-Up Shift and Reasoning for Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Bottom-Up_Shift_and_Reasoning_for_Referring_Image_Segmentation_CVPR_2021_paper.pdf) | CVPR 2021 | [[code]](https://github.com/incredibleXM/BUSNet) |
| LTS | [Locate then Segment: A Strong Pipeline for Referring Image Segmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Jing_Locate_Then_Segment_A_Strong_Pipeline_for_Referring_Image_Segmentation_CVPR_2021_paper.pdf) | CVPR 2021 |  |
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
| RefVOS | [RefVOS: A Closer Look at Referring Expressions for Video Object Segmentation](https://arxiv.org/abs/2010.00263) | arxiv |  |
| URVOS | [URVOS: Unified Referring Video Object Segmentation Network with a Large-Scale Benchmark](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600205.pdf) | ECCV 2020 | [[code]](https://github.com/skynbe/Refer-Youtube-VOS) |
|  | [Video Object Segmentation with Language Referring Expressions](https://link.springer.com/chapter/10.1007/978-3-030-20870-7_8) | ACCV 2018 |  |


## 5. Referring 3D Instance Segmentation

| Short name | Paper | Source | Code/Project Link  |
| --- | --- | --- | --- |
| TGNN | [Text-Guided Graph Neural Networks for Referring 3D Instance Segmentation](https://www.aaai.org/AAAI21Papers/AAAI-4433.HuangP.pdf) | AAAI 2021 |  |

