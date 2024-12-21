# Bridging Accuracy and Interpretability: A Rescaled Cluster-then-Predict Approach for Enhanced Credit Scoring

By Huei-Wen Teng, Ming-Hsuan Kang, I-Han Lee, Le-Chi Bai 

Credit scoring is pivotal in the financial industry for assessing individuals’ creditworthiness and optimizing financial institutions' risk-adjusted returns. While the XGBoost algorithm stands as the state-of-the-art classifier for credit scoring, its intricate nature impedes easy interpretation, a critical aspect for stakeholders' decision-making. This paper introduces a novel approach termed the “Rescaled Cluster-then-Predict Method,” aimed at enhancing both the interpretability and predictive performance of credit scoring models. Our method employs a two-step process, initially rescaling the features and subsequently clustering the data into subgroups. Consequently, we employ Logistic Regression within each subgroup to generate predictions. The paper's primary contributions are twofold. Firstly, empirical evaluations on two distinct datasets demonstrate that our proposed method attains a competitive performance compared to XGBoost while substantially improving interpretability. Notably, in some instances, the Logistic Regression outperforms XGBoost. Secondly, we reveal that clustering solely the positive cases, as opposed to the entire dataset, yields comparable results while markedly reducing computational requirements. These insights hold significant practical implications for the financial industry, which consistently seeks credit scoring models that are not only accurate but also interpretable and computationally efficient.

Keywords: credit scoring, cluster-then-predict, rescaling, XGBoost, Logistic Regresssion

Data of source
1. GMC dataset: https://www.kaggle.com/competitions/GiveMeSomeCredit/data
2. PAK dataset: https://pakdd.org/archive/pakdd2010/PAKDDCompetition.html

Link to overleaf: https://www.overleaf.com/read/nfqdcvdrrkbt#85ba4e

