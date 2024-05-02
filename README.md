# FeatureQualityMetrics

The code in this repository provides an efficient metric to evaluate and compare possible feature extractors for a downstream classification task.  These metrics are typically called "transferability metrics" - for a good survey of the space, see Ding, Y., Jiang, B., Yu, A., Zheng, A., & Liang, J. (2024). Which Model to Transfer? A Survey on Transferability Estimation. ArXiv, abs/2402.15231.  

Our method forward passes a selected number of training examples (for the target task) through the feature extractor, collects the corresponding extracted feature vectors at the penultimate feature layer, and then measures the ratio of intra-class compactness to inter-class separation (measured by cosine similarity) for each class individually.  The final metric averages over all classes.

The metric correlates well to the eventual test performance of a finetuned version of the feature extractor (with a new classification head), thus precluding the need to perform a full finetuning on every possible feature extractor and reducing the computation time on the end-user.