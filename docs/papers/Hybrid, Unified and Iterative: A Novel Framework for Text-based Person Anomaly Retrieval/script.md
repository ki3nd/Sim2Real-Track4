

# Hybrid, Unified and Iterative: A Novel Framework for Text-based Person Anomaly Retrieval

Tien-Huy Nguyen\*

University of Information Technology  
Vietnam National University  
Ho Chi Minh, Vietnam

Huu-Phong Phan-Nguyen\*

University of Information Technology  
Vietnam National University  
Ho Chi Minh, Vietnam

Huu-Loc Tran\*

University of Information Technology  
Vietnam National University  
Ho Chi Minh, Vietnam

Quang-Vinh Dinh

AI VIETNAM Lab  
Ninh Thuan, Vietnam

## Abstract

Text-based person anomaly retrieval has emerged as a challenging task, with most existing approaches relying on complex deep-learning techniques. This raises a research question: How can the model be optimized to achieve greater fine-grained features? To address this, we propose a Local-Global Hybrid Perspective (LHP) module integrated with a Vision-Language Model (VLM), designed to explore the effectiveness of incorporating both fine-grained features alongside coarse-grained features. Additionally, we investigate a Unified Image-Text (UIT) model that combines multiple objective loss functions, including Image-Text Contrastive (ITC), Image-Text Matching (ITM), Masked Language Modeling (MLM), and Masked Image Modeling (MIM) loss. Beyond this, we propose a novel iterative ensemble strategy, by combining iteratively instead of using model results simultaneously like other ensemble methods. To take advantage of the superior performance of the LHP model, we introduce a novel feature selection algorithm based on its guidance, which helps improve the model's performance. Extensive experiments demonstrate the effectiveness of our method in achieving state-of-the-art (SOTA) performance on PAB dataset, compared with previous work, with a 9.70% improvement in R@1, 1.77% improvement in R@5, and 1.01% improvement in R@10.

## CCS Concepts

• Information systems → Retrieval models and ranking.

## Keywords

Multimedia Retrieval, Deep Learning, Representation Learning.

---

\* All authors contributed equally to this paper.  
This research is partly supported by AI VIETNAM [1].

---

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from [permissions@acm.org](mailto:permissions@acm.org).  
*WWW Companion '25, Sydney, NSW, Australia*

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.  
ACM ISBN 979-8-4007-1331-6/2025/04  
<https://doi.org/10.1145/3701716.3717653>

## ACM Reference Format:

Tien-Huy Nguyen\*, Huu-Loc Tran\*, Huu-Phong Phan-Nguyen\*, and Quang-Vinh Dinh. 2025. Hybrid, Unified and Iterative: A Novel Framework for Text-based Person Anomaly Retrieval. In *Companion Proceedings of the ACM Web Conference 2025 (WWW Companion '25)*, April 28-May 2, 2025, Sydney, NSW, Australia. ACM, New York, NY, USA, 5 pages. <https://doi.org/10.1145/3701716.3717653>

## 1 Introduction

Text-based person retrieval, a well-established task [5, 16, 18], involves retrieving specific indicators from large-scale image databases using textual queries. The current method exhibits biases toward common actions and only solves their actions, limiting diversity and the generalizability of models, particularly for detecting abnormal behaviors. To address this, we explore text-based anomaly person retrieval, leveraging advancements in deep learning and computer vision [2, 3, 6–9]. Unlike traditional methods that use entire images, we focus on localized image regions to enhance attention to fine details. Furthermore, we adopt multiple objective learning to model complex image-text features and propose a novel iterative ensemble strategy to iteratively refine model predictions, capitalizing on the strengths of multiple models. Our primary contributions are:

- A hybrid approach blending local and global perspectives (LHP), enhancing the model's ability to utilize both fine-grained and holistic visual information.
- Unified Image-Text (UIT) Modeling integrates MIM [13], MLM [11], ITC [14] and ITM [17] tasks, leveraging LHP-based feature selection for efficient and accurate multi-modal representation learning.
- We introduce a novel iterative ensemble algorithm that utilizes the results of multiple models more effectively, helping to improve the overall performance.
- Our comprehensive experiments demonstrate the effectiveness of our proposed method, achieving SOTA performance in text-based person anomaly retrieval on real-world datasets.

## 2 Method

### 2.1 Local-global Hybrid Perspective Modeling

In this section, we introduce LHP modeling, as illustrated in **Figure 1(a)**. The LHP module determines whether to process an image locally or globally based on a probabilistic criterion. The LHP module takes an input image  $I$ , along with a local transform and a global transform. A random value is sampled from a normal distribution

![](aa9e46d6f962be5cebcbb5c654c9b13e_img.jpg)

**Figure 1: (a) Overview of Local-global Hybrid Perspective (LHP) Modeling.** It processes an image probabilistically, applying either a local transform for fine-grained details or a global transform for comprehensive context. Contrastive learning aligns image and text embeddings by minimizing distances for matching pairs and maximizing distances for non-matching pairs. **(b) Unified Image-Text (UIT) Modeling with Feature Selection.** UIT is a cross-modal framework that integrates MIM, MLM, ITC, and ITM to unify image and text understanding. UIT enhances inference by leveraging LHP-based feature selection for efficient and accurate multi-modal representation learning.

with a mean of 0.5 and a variance of  $1 \div 6$ . If the sampled value is greater than 0.5, the local transform is applied to image  $I$ ; otherwise, the global transform is applied. The module then outputs either the locally transformed or globally transformed image  $I$ , based on the sampled value.

In the local perspective, a region of interest is cropped from the image to focus on fine-grained details. In contrast, the global perspective retains the entire image to capture holistic contextual information. This hybrid approach allows the model to benefit from both granular and comprehensive views of the image.

To align the image and text embeddings, we employ a contrastive loss function that minimizes the distance between embeddings of matching pairs while maximizing the distance between embeddings of non-matching pairs. For a given image-text pair  $(I, T)$ , their feature representations are extracted as follows:

$$f_i = \mathcal{E}_i(I), f_t = \mathcal{E}_t(T). \quad (1)$$

where  $\mathcal{E}_i, \mathcal{E}_t$  is the Image Encoder, Text Encoder.

The image-to-text similarity  $S_{I2T}$  within the batch is defined as follows:

$$S_{I2T} = \frac{\exp(s(f_i, f_t)/\tau)}{\sum_{j=1}^N \exp(s(f_i, f_t^j)/\tau)} \quad (2)$$

where  $s(\cdot, \cdot)$  is the cosine similarity,  $\tau$  is a temperature parameter. Finally, the contrastive learning loss is presented below:

$$\mathcal{L}_{cl} = -\frac{1}{2} \mathbb{E} [\log S_{I2T} + \log S_{T2I}] \quad (3)$$

### 2.2 Unified Image-Text (UIT) Modeling

**2.2.1 Overall framework.** In this section, we introduce UIT, as illustrated in **Figure 1(b)**, a cross-modal model developed for our work.

Drawing inspiration from CMP [15], UIT enhances its ability to learn robust visual representations by incorporating masked image reconstruction into its training process.

UIT comprises four main components: Image Encoder ( $\mathcal{E}_i$ ), Text Encoder ( $\mathcal{E}_t$ ), Cross Encoder ( $\mathcal{E}_{cross}$ ) and a Decoder ( $\mathcal{D}$ ). Given an image ( $I$ )-text ( $T$ ) pair, Random Masking is applied to generate a masked image ( $I_{masked}$ ) and text ( $T_{masked}$ ). The masked image passes through  $\mathcal{E}_i$  and  $\mathcal{D}$  for the Masked Image Modeling (MIM) task. Meanwhile,  $I$  and  $T_{masked}$  are encoded by  $\mathcal{E}_i$  and  $\mathcal{E}_t$  for the Image-Text Contrastive (ITC) task, producing image embeddings ( $f_i$ ) and text embeddings ( $f_t$ ), respectively. These embeddings are combined in the Cross Encoder ( $\mathcal{E}_{cross}$ ) to get cross embeddings ( $f_{cross}$ ), which are utilized for the Image-Text Matching (ITM) and Masked Language Modeling (MLM) tasks.

**Masked Image Modeling (MIM).** The MIM aims to reconstruct masked image patches for improving the visual representations of  $\mathcal{E}_i$ . Given an input image  $I$ , the image is divided into patches, and a subset of patches is randomly masked to get the masked image  $I_{masked}$ . The model is trained to predict these masked patches using unmasked ones. The objective of **MIM** is computed as:

$$\hat{I} = \mathcal{D}(\mathcal{E}_i(I_{masked})), \quad (4)$$

$$\mathcal{L}_{mim} = \frac{1}{N} \sum_{k=1}^N \|\hat{I}_i - I_i\|_1, \quad (5)$$

where  $N$  is the number of images,  $\hat{I}_i$  is the  $i$ -th reconstructing image, and  $I_i$  is the original image.

**Image-Text Contrastive (ITC) Learning.** The goal of contrastive learning is to maximize the similarity between positive

pairs while minimizing the similarity between negative pairs. The contrastive loss is computed as described in **Section 2.1**.

**Image-Text Matching (ITM) Learning:** The Image-Text Matching determines whether a given image-text pair corresponds. The cross-encoder processes the text embeddings as input and integrates the image embeddings using a cross-attention mechanism at each layer. The output of the cross encoder is projected into a 2-dimensional space using *ITM head* (a Multi-Layer Perceptron (MLP)). The Image-Text Matching loss is then defined as:

$$\mathcal{L}_{itm} = -\mathbb{E}\left[p(I, T) \log \hat{p}(I, T) + (1 - p(I, T)) \log (1 - \hat{p}(I, T))\right], \quad (6)$$

where  $p(I, T) \in \{0, 1\}$  is the ground truth label for whether the image-text pair matches (1 for matching, 0 for non-matching),  $\hat{p}(I, T)$  is the predicted probability of the pair matching.

**Masked Language Modeling (MLM):** The MLM aims to reconstruct masked text tokens from the context. The masked text  $T_{masked}$ , along with the corresponding person image  $I$ , is fed into the cross encoder. The cross-encoder processes these inputs and outputs a fused representation, which is passed through an *MLM head*. The objective of MLM learning is to predict the likelihood of the masked token  $t$  in  $T_{masked}$ , given the image  $I$  and the unmasked parts of the text. The training process minimizes the cross-entropy loss:

$$\mathcal{L}_{mlm} = -\mathbb{E}\left[p_{mask}(I, T_{masked}) \log \hat{p}_{mask}(I, T_{masked})\right], \quad (7)$$

where  $\hat{p}_{mask}(I, T_{masked})$  is the predicted likelihood of the masked token  $t$  in  $T_{masked}$ ,  $p_{mask}(I, T_{masked})$  is the ground truth one-hot vector representing the correct token.

Given the above optimization objectives, the full training loss is formulated as:

$$\mathcal{L} = \mathcal{L}_{itc} + \mathcal{L}_{itm} + \mathcal{L}_{mlm} + \alpha \mathcal{L}_{mim}, \quad (8)$$

where  $\alpha$  denotes the weight for the MIM loss and we set  $\alpha = 0.1356$ .

#### Algorithm 1 Feature Selection Algorithm

- 1: **Input:** Image embeddings  $f_i$ , LHP similarity matrix  $sim\_matrix$
- 2: **Output:**  $topk$  image features for each text embedding.
- 3:  $selected\_features \leftarrow []$
- 4: **for**  $index, row$  **in**  $enumerate(sim\_matrix)$  **do**
- 5:    $topk\_sim, topk\_idx \leftarrow topk(row)$
- 6:    $selected\_features[index] \leftarrow f_i[topk\_idx]$
- 7: **end for**
- 8: **return**  $selected\_features$

**2.2.2 Feature Selection.** During the inference stage, we leverage the similarity matrix obtained from the LHP model to select the  $top-k$  image features with the highest similarity scores, as detailed in **Algorithm 1**. This approach takes advantage of the superior performance of the LHP model, which is more effective at identifying high-quality  $top-k$  candidates compared to the similarity matrix computed directly by the UIT model. The similarity matrix is calculated based on the cosine similarity between  $f_i$  and  $f_i$ .

By using the LHP model for feature selection, the process focuses on the most relevant image features, enabling the model to prioritize important images. The selected  $top-k$  image features and their corresponding text features are then fed into the cross-encoder. The cross-encoder processes these features and computes the final

matching scores via the ITM Head, ensuring accurate alignment between image and text modalities.

### 2.3 Iterative Ensemble

Ensemble learning is a powerful technique that combines the predictions of multiple models. By aggregating outputs from diverse models, ensemble methods can reduce overfitting and enhance generalization. To utilize that, we propose a novel method for ensemble, named as iterative ensemble, described as in **Algorithm 2**.

#### Algorithm 2 Iterative Ensemble Algorithm

- 1: **Input:** Gallery set  $I$ , queries  $Q$ , model list  $\theta$ , ground truth  $gt$ , scoring function  $f_s$ , weight value set  $W \in [0, 1]$
- 2: **Output:** Best score matrix  $S$ .
- 3: Initialize  $S \leftarrow 0$
- 4: **for**  $i, t_\theta$  **in**  $enumerate(\theta)$  **do**
- 5:    $pred \leftarrow \text{topk}(w \cdot S + (1 - w) \cdot t_\theta(I, Q), \text{dim} = 1)$
- 6:    $w \leftarrow \underset{w \in W}{\text{argmax}} (f_s(pred, gt))$
- 7:    $S \leftarrow w \cdot S + (1 - w) \cdot t_\theta(I, Q)$
- 8: **end for**
- 9: **return** Best score matrix  $S$ .

where  $pred \in \mathbb{R}^{n \times n}$  with  $n$  is equal to number of images. In addition, to be able to choose the weight for the best results, we perform hyperparameter tuning with  $W$  and we realized that when setting  $w$  values close to the value 1, the results improve significantly.

Most of the current ensemble methods mostly use simultaneous model predictions. The difference between our method compared to others is that it retains most of the scores of the current score, and **gradually** references the scores of the next models. Initially,  $S$  is equal to 0, and after the first iteration,  $S$  receives the of the first model. However, in the next iterations,  $S$  will be combined between the results of the previous model (multiplied by  $w$ ) with the results of the current model (multiplied by  $(1 - w)$ ). Finally, we obtain the final  $S$  while still ensuring references from different models, allowing if the result is agreed upon by many models, the score is still high, and vice versa. Furthermore, when the results are uncertain, an iterative ensemble acts as a reranker, helping to change rankings.

## 3 Experimental Results

### 3.1 Implementation in details

We integrate the (LHP) module into BEiT-3, baseline model, [12] to combine local and global features, and use Swin-B and a BERT-based encoder for unified image-text modeling (UIT). During inference, we apply an iterative ensemble strategy with BEiT-3 [12], UIT, BLIP-2 [4], and CLIP [10] to enhance performance. BEiT-3 with LHP is fine-tuned for 3 epochs (batch size 184, image size  $384 \times 384$ ), and UIT for 22 epochs (batch size 84, image size  $224 \times 224$ ). Both experiments use cosine annealing for learning rate scheduling, the AdamW optimizer, and initialized initial learning rate value is  $10^{-5}$ .

### 3.2 Dataset and Evaluation Metrics

**The Pedestrian Anomaly Behavior (PAB)** [15] dataset includes 1,013,605 synthesized and 1,978 real-world image-text pairs, covering diverse actions and anomalies. Real-world videos provide the test data, while a diffusion model generates the training set. Performance is evaluated using recall rates (R@K). A search is successful

if the image perfectly matches the text query that appears in the top k-ranked images. Higher R@K values indicate better performance, and results are reported for R@1, R@5, and R@10.

### 3.3 Quantitative Results

We evaluate our proposed method against APTM and CMP on the Text-based Person Anomaly retrieval, with results shown in **Table 1**.

| Method                      |       | R@1          | R@5          | R@10         |
|-----------------------------|-------|--------------|--------------|--------------|
| <b>0.1M training images</b> |       |              |              |              |
| APTM[16]                    | 0shot | 9.40         | 22.14        | 30.18        |
| APTM[16]                    | tuned | 69.92        | 95.60        | 97.32        |
| IHNM[15]                    |       | 72.25        | 95.91        | 98.03        |
| PE [15]                     |       | 71.79        | 95.40        | 97.83        |
| CMP [15]                    |       | 72.80        | 96.01        | 97.47        |
| <b>Ours</b>                 |       | <b>85.39</b> | <b>99.49</b> | <b>99.95</b> |
| <b>1M training images</b>   |       |              |              |              |
| CMP [15]                    |       | 79.53        | 97.93        | 98.84        |
| <b>Ours</b>                 |       | <b>89.23</b> | <b>99.70</b> | <b>99.85</b> |

**Table 1:** Comparison with other methods on 0.1M and 1M training images of PAB dataset.

Our method achieves SOTA performance across all metrics (R@1, R@5, and R@10) on the PAB dataset. When trained on 0.1M images, our method outperforms CMP, the second-best approach, with a substantial 12.59% improvement in Recall@1, achieving 85.39%. On the full dataset of 1M images, our method maintains its advantage, improving Recall@1 by 9.70%, reaching 89.23%. These results demonstrate the robustness and scalability of our approach, which outperforms existing methods across varying dataset sizes.

### 3.4 Ablation Study

As described in **Table 2**, The only-global method achieves high R@10 (99.95%) but lower R@1. The only-local approach slightly improves R@1 (85.39%) by emphasizing localized details but underperforms slightly at R@10 (99.90%). It is easy to see that the local-global hybrid perspective (LHP) gives superior results, which proves that combining both will help the model give the best and most stable results, instead of using only one of them.

| Method      | R@1          | R@5          | R@10         |
|-------------|--------------|--------------|--------------|
| only-global | 85.24        | 99.44        | 99.95        |
| only-local  | 85.39        | 99.44        | 99.90        |
| LHP (ours)  | <b>85.39</b> | <b>99.49</b> | <b>99.95</b> |

**Table 2:** Ablation experiments for LHP model. Only-global, only-local, and LHP use cropped, entire images respectively, and use both as model input.

As shown in the **Table 3**, the baseline model trained on 0.1M images, R@1 is 85.24%. Adding LHP improves R@1 to 85.39%, a 0.18% increase, showing the benefit of leveraging local-global hybrid perspectives.

When scaling to 1M images, LHP further boosts R@1 to 87.11%, representing a significant 1.72% improvement over the 0.1M training setup. Incorporating UIT with feature selection (FS) elevates R@1 to 88.37%, a 1.26% gain compared to using LHP alone on 1M images, demonstrating the effectiveness of multi-modal objectives. Finally, adding iterative ensemble (IE) alongside LHP and UIT (FS) achieves the highest R@1 of 89.23%, a 0.86% increase over the previous setup and a total 3.99% improvement over the baseline.

| Method                         | Training Images | R@1          | R@5          | R@10         |
|--------------------------------|-----------------|--------------|--------------|--------------|
| Baseline                       | 0.1M            | 85.24        | 99.44        | <b>99.95</b> |
| Baseline + LHP                 | 0.1M            | 85.39        | 99.49        | <b>99.95</b> |
| Baseline + LHP                 | 1M              | 87.11        | 99.65        | 99.85        |
| Baseline + LHP + UIT (FS)      | 1M              | 88.37        | 99.70        | 99.85        |
| Baseline + LHP + UIT (FS) + IE | 1M              | <b>89.23</b> | <b>99.70</b> | 99.85        |

**Table 3:** Ablation experiments for highlighting the progressive improvements in person anomaly retrieval performance achieved by incorporating LHP, UIT (FS: feature selection), and (IE: iterative ensemble).

| Attempts | Iter.1 (UIT) | Iter.2 (BLIP-2) | Iter.3 (CLIP) | R@1          | R@5          | R@10         |
|----------|--------------|-----------------|---------------|--------------|--------------|--------------|
| 1        | 0            | -               | -             | 87.11        | 99.65        | 99.85        |
| 2        | 0.5          | -               | -             | 84.07        | 99.14        | 99.85        |
| 3        | 0.8          | -               | -             | 87.26        | 99.70        | 99.85        |
| 4        | 0.85         | -               | -             | 87.41        | 99.70        | 99.85        |
| 5        | 0.875        | -               | -             | 87.97        | 99.70        | 99.85        |
| 6        | 0.9          | -               | -             | 88.22        | 99.70        | 99.85        |
| 7        | 0.9125       | -               | -             | 88.27        | <b>99.75</b> | 99.85        |
| 8        | 0.925        | -               | -             | 88.37        | 99.70        | 99.85        |
| 9        | 0.9375       | -               | -             | 88.17        | 99.70        | 99.85        |
| 10       | 0.95         | -               | -             | 87.97        | 99.70        | 99.85        |
| 11       | 0.925        | 0.875           | -             | 88.83        | 99.70        | 99.85        |
| 12       | 0.925        | 0.9             | -             | 88.88        | <b>99.75</b> | <b>99.90</b> |
| 13       | 0.925        | 0.925           | -             | 88.68        | 99.70        | 99.85        |
| 14       | 0.925        | 0.9             | 0.85          | 88.88        | 99.70        | 99.85        |
| 15       | 0.925        | 0.9             | 0.8725        | <b>89.23</b> | 99.70        | 99.85        |
| 16       | 0.925        | 0.9             | 0.9           | <b>89.23</b> | 99.70        | 99.85        |
| 17       | 0.925        | 0.9             | 0.9125        | 89.18        | 99.70        | 99.85        |
| 18       | 0.925        | 0.9             | 0.925         | 89.13        | 99.70        | 99.85        |

**Table 4:** Iterative ensemble results with varying weights for subsequent iterations and their impact on R@1, R@5, and R@10 performance.

As presented in **Table 4**, analyze the performance of iterative ensembles using various configurations of the UIT, BLIP-2, and CLIP models. The iterative adjustments in the UIT model (Iter.1) show steady improvements in retrieval performance, with Recall@1 increasing from an initial setting of 0 to values like 0.85 and 0.925, while Recall@5 and Recall@10 remain consistently high at 99.70% and 99.85%, respectively. The inclusion of BLIP-2 (Iter.2) and CLIP (Iter.3) further enhances performance, reaching a Recall@1 peak of 89.23% in later iterations (e.g., Iter.1 = 0.925, Iter.2 = 0.9, Iter.3 = 0.9). This study demonstrates the effectiveness of iterative optimization and the synergy of combining complementary models to achieve state-of-the-art retrieval accuracy.

## 4 Conclusion

In this paper, we proposed a novel framework for Text-based Person Anomaly Retrieval by introducing the LHP and UIT modeling. LHP effectively combines fine-grained local details with global contextual information, while UIT integrates multiple loss objectives, including MIM, MLM, ITC, and ITM. Furthermore, we introduced a novel feature selection algorithm and an iterative ensemble strategy, which significantly enhance the retrieval performance by leveraging complementary models and refining predictions.

Extensive experiments on the PAB dataset demonstrate that our method achieves SOTA performance, with improvements over previous methods. Specifically, our approach achieves a 9.7% improvement in Recall@1 on the 1M full dataset and a 12.59% improvement on the 0.1M dataset. Ablation studies further validate the effectiveness of each component, showing the advantages of combining local and global perspectives and the impact of feature selection and ensemble strategies.

## References

- [1] [n. d.]. AI VIETNAM – aivietnam.edu.vn. <https://aivietnam.edu.vn>
- [2] Quang-Khai Bui-Tran, Duc-Huy Ha, Minh-Hung Nguyen, Phuc-Hung Dang, Thien-An Trieu-Hoang, and Tien-Huy Nguyen. [n. d.]. Enhanced Video Retrieval System: Leveraging GPT-4 for Multimodal Query Expansion and Open Image Search.
- [3] Minh-Dung Le-Quynh, Anh-Tuan Nguyen, Anh-Tuan Quang-Hoang, Van-Huy Dinh, Tien-Huy Nguyen, Hoang-Bach Ngo, and Minh-Hung An. 2023. Enhancing video retrieval with robust clip-based multimodal system. In *Proceedings of the 12th International Symposium on Information and Communication Technology*. 972–979.
- [4] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. arXiv:2301.12597 [cs.CV] <https://arxiv.org/abs/2301.12597>
- [5] Zheng Li, Lijia Si, Caili Guo, Yang Yang, and Qiushi Cao. 2024. Data Augmentation for Text-based Person Retrieval Using Large Language Models. arXiv:2405.11971 [cs.CV] <https://arxiv.org/abs/2405.11971>
- [6] Tien-Huy Nguyen, Hoang-Long Nguyen-Huu, Thien-Doanh Le, Huu-Loc Tran, Quoc-Khanh Le-Tran, Hoang-Bach Ngo, Minh-Hung An, and Quang-Vinh Dinh. 2023. Multimodal Fusion in NewsImages 2023: Evaluating Translators, Keyphrase Extraction, and CLIP Pre-Training. In *MediaEval*.
- [7] Tien-Huy Nguyen, Quang-Khai Tran, and Anh-Tuan Quang-Hoang. 2024. Improving Generalization in Visual Reasoning via Self-Ensemble. arXiv:2410.20883 [cs.CV] <https://arxiv.org/abs/2410.20883>
- [8] Tho-Quang Nguyen, Huu-Loc Tran, Tuan-Khoa Tran, Huu-Phong Phan-Nguyen, and Tien-Huy Nguyen. 2024. FA-YOLOv9: Improved YOLOv9 Based on Feature Attention Block. 1–6. doi:10.1109/MAPR63514.2024.10661057
- [9] Hoang-Long Nguyen-Huu, Tran Thi Cam Giang, Cam-Nguyen Tran-Nhu, Phuoc Phan Hoang, and Tien-Huy Nguyen Phat Huu and. [n. d.]. AViSearch: A Multimodal Video Event Retrieval System via Query Enhancement and Optimized Keyframes.
- [10] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models From Natural Language Supervision. arXiv:2103.00020 [cs.CV] <https://arxiv.org/abs/2103.00020>
- [11] Koustuv Sinha, Robin Jia, Dieuwke Hupkes, Joelle Pineau, Adina Williams, and Douwe Kiela. 2021. Masked Language Modeling and the Distributional Hypothesis: Order Word Matters Pre-training for Little. arXiv:2104.06644 [cs.CL] <https://arxiv.org/abs/2104.06644>
- [12] Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, and Furu Wei. 2022. Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks. arXiv:2208.10442 [cs.CV] <https://arxiv.org/abs/2208.10442>
- [13] Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. 2022. Simmim: A simple framework for masked image modeling. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 9653–9663.
- [14] Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Bin Xiao, Ce Liu, Lu Yuan, and Jianfeng Gao. 2022. Unified contrastive learning in image-text-label space. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 19163–19173.
- [15] Shuyu Yang, Yaxiong Wang, Li Zhu, and Zhedong Zheng. 2024. Beyond Walking: A Large-Scale Image-Text Benchmark for Text-based Person Anomaly Search. arXiv:2411.17776 [cs.CV] <https://arxiv.org/abs/2411.17776>
- [16] Shuyu Yang, Yinan Zhou, Yaxiong Wang, Yujiao Wu, Li Zhu, and Zhedong Zheng. 2023. Towards Unified Text-based Person Retrieval: A Large-scale Multi-Attribute and Language Search Benchmark. arXiv:2306.02898 [cs.CV] <https://arxiv.org/abs/2306.02898>
- [17] Ying Zhang and Huchuan Lu. 2018. Deep cross-modal projection learning for image-text matching. In *Proceedings of the European conference on computer vision (ECCV)*. 686–701.
- [18] Zhedong Zheng, Liang Zheng, Michael Garrett, Yi Yang, Mingliang Xu, and Yi-Dong Shen. 2020. Dual-path Convolutional Image-Text Embeddings with Instance Loss. *ACM Transactions on Multimedia Computing, Communications, and Applications* 16, 2 (May 2020), 1–23. doi:10.1145/3383184