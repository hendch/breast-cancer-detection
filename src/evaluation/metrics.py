from sklearn.metrics import confusion_matrix, accuracy_score


def compute_paper_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "accuracy": float(acc),
        "tpr": float(tpr),
        "tnr": float(tnr),
        "fpr": float(fpr),
        "fnr": float(fnr),
    }
