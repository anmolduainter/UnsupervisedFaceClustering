import numpy as np
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)

    tprs = np.zeros(nrof_thresholds)
    fprs = np.zeros(nrof_thresholds)
    accuracy = np.zeros(nrof_thresholds)
    indices = np.arange(nrof_pairs)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)

    # Find the best threshold for the fold
    for threshold_idx, threshold in enumerate(thresholds):
        tprs[threshold_idx], fprs[threshold_idx], accuracy[threshold_idx] = calculate_accuracy(threshold, dist,actual_issame)

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def similarity_function(embeddings1, embeddings2):
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    similarity_scores = 1 / (1 + dist)
    return similarity_scores


def test_model_pred_quality(embeddings, labels):
    print (labels)
    unique_label = np.unique(labels)
    rocl = []
    for l in unique_label:
        pos_indexes = np.where(labels == l)[0]
        neg_indexes = np.where(labels != l)[0]
        
        embeddings1 = []
        embeddings2 = []
        issame = []
        # Positive cases
        for idx in pos_indexes:
            for idx1 in pos_indexes:
                if (idx != idx1 and idx1 > idx):
                    embeddings1.append(embeddings[idx])
                    embeddings2.append(embeddings[idx1])
                    issame.append(1)

        # Neg cases
        for idx in pos_indexes:
            for idx1 in neg_indexes:
                if (idx != idx1):
                    embeddings1.append(embeddings[idx])
                    embeddings2.append(embeddings[idx1])
                    issame.append(0)

        embeddings1 = np.array(embeddings1)
        embeddings2 = np.array(embeddings2)
        similarity_scores = similarity_function(embeddings1, embeddings2)

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(issame, similarity_scores)
        roc_auc = auc(fpr, tpr)
        rocl.append(roc_auc)

        # # Plot ROC curve
        # plt.figure(figsize=(8, 6))
        # plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        # plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver Operating Characteristic (ROC) Curve')
        # plt.legend(loc='lower right')
        # plt.grid()
        # plt.show()

    # exit()

    roc_percentile = np.mean(rocl)
    return roc_percentile