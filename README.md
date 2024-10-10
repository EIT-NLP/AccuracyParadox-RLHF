<div align="center">

# The Accuracy Paradox in RLHF: When Better Reward Models Don't Yield Better Language Models

![QA-FEEDBACK](https://img.shields.io/badge/Dataset-QA--FEEDBACK-blue)
![ASQA](https://img.shields.io/badge/Dataset-ASQA-blue)

![Longformer](https://img.shields.io/badge/Model-Longformer-21C2A4)
![T5-small](https://img.shields.io/badge/Model-T5--small-21C2A4)
![T5-base](https://img.shields.io/badge/Model-T5--base-21C2A4)
![T5-large](https://img.shields.io/badge/Model-T5--large-21C2A4)

📰 [Paper](https://arxiv.org/abs/2410.06554)

</div>

## 1. Introduction
Reinforcement Learning from Human Feedback (RLHF) significantly improves language models by aligning their outputs with human preferences. Traditionally, stronger reward models—those with higher accuracy—are expected to enhance language model performance. However, our research presents a counterintuitive finding: **language models guided by moderately accurate reward models often outperform those trained with highly accurate ones**. 

This study focuses on relevance, factuality, and completeness tasks using the **QA-FEEDBACK** dataset and reward models based on **Longformer**. Through extensive experimentation, we show that overly accurate reward models can lead to **overfitting** or poor generalization, while moderate accuracy yields better performance. This raises critical questions about how to balance reward model accuracy to optimize language model outputs in RLHF.

## 2. The Role of Reward Model Accuracy
### Accuracy in RLHF
In RLHF, reward models evaluate the outputs of language models based on specific criteria such as relevance or factuality. A common assumption is that higher reward model accuracy should always lead to better LM performance, as more accurate models provide better feedback. However, our findings indicate that **moderate accuracy** is more effective in striking a balance between guiding model training and preventing overfitting.

### Key Insights
We introduce a framework that explores the relationship between reward model accuracy and language model performance. The key factors include:
- **Task Alignment**: Moderately accurate reward models tend to offer feedback that is more aligned with the overall task, preventing LMs from overfitting to overly specific or narrow criteria.
- **Training Stability**: Reward models of moderate accuracy foster a more stable and generalizable training process, particularly in tasks requiring complex reasoning, such as QA and long-form answer generation.

## 3. Experimental Setup
### Models
We conducted experiments using models from the **T5** family, including **T5-small**, **T5-base**, and **T5-large**, trained with **Longformer-based** reward models for tasks focusing on **factuality**, **relevance**, and **completeness**.

### Datasets
The **QA-FEEDBACK** dataset, derived from the **ASQA** dataset, focuses on generating long-form answers to ambiguous, open-domain factual questions. The dataset is divided into training, validation, and testing sets, requiring models to generate detailed responses from multiple knowledge sources.

## 4. Results Across Tasks
Our experiments reveal a **consistent trend**: models trained with moderately accurate reward models tend to outperform those trained with highly accurate ones across a broad range of tasks, including individual cases.
<div align="center">
  <img src="Pictures/NSB_image.png" width="60%" />
  <p>RM accuracy vs. LM performance.</p>
</div>

## 5. Conclusion
This study challenges the prevailing assumption that higher reward model accuracy always leads to better language model performance in RLHF. Our findings show that **moderate accuracy in reward models** can improve task alignment and training stability, leading to better outcomes across relevance, factuality, and completeness tasks. Future research should explore how to fine-tune reward models to achieve the optimal balance between accuracy and generalization, particularly in complex NLP tasks.

## 6. Citation
```bibtex
@inproceedings{chen2024accuracyparadoxrlhfbetter,
  title={The Accuracy Paradox in RLHF: When Better Reward Models Don't Yield Better Language Models},
  author={Yanjun Chen, Dawei Zhu, Yirong Sun, Xinghao Chen, Wei Zhang, Xiaoyu Shen},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2024},
}
```

## 7. Contact
For questions or collaborations, please contact us at <yan-jun.chen@connect.polyu.hk>.
