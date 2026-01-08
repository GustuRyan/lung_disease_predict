import os
import cv2
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

def load_gray_image(path, size=128):
    img = Image.open(path).convert("L")
    img = img.resize((size, size))
    return np.array(img, dtype=np.float32) / 255.0

def extract_hog_opencv(img):
    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    img_uint8 = np.array(img.resize((128, 128)), dtype=np.uint8)
    return hog.compute(img_uint8).flatten()

def extract_hog_features(img):
    img_pil = Image.fromarray((img * 255).astype(np.uint8))
    return extract_hog_opencv(img_pil)

def train_svm_dynamic_classes(
    selected_classes,
    dataset_root,
    img_size=128,
    C=1.0,
    gamma="scale",
    progress_callback=None,
    status_callback=None
):
    if len(selected_classes) < 2:
        raise ValueError("Minimal pilih 2 class")

    X, y = [], []
    class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}

    # ==============================
    # LOAD DATA
    # ==============================
    for cls in selected_classes:
        cls_path = os.path.join(dataset_root, cls)

        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(cls_path, fname)

                img = load_gray_image(img_path, img_size)
                feat = extract_hog_features(img)

                X.append(feat)
                y.append(class_to_idx[cls])

    X = np.array(X)
    y = np.array(y)

    if status_callback:
        status_callback(f"Dataset loaded: {len(X)} images")

    # ==============================
    # SPLIT DATA (STRATIFIED)
    # ==============================
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.3,
        stratify=y,
        random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42
    )

    # ==============================
    # TRAIN SVM
    # ==============================
    if status_callback:
        status_callback("Training SVM model...")

    svm_model = SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        probability=True,
        decision_function_shape="ovr"  # multiclass
    )

    svm_model.fit(X_train, y_train)

    # ==============================
    # EVALUATION
    # ==============================
    y_train_pred = svm_model.predict(X_train)
    y_val_pred   = svm_model.predict(X_val)
    y_test_pred  = svm_model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc   = accuracy_score(y_val, y_val_pred)
    test_acc  = accuracy_score(y_test, y_test_pred)

    # ===== Detailed Metrics (TEST SET) =====
    report = classification_report(
        y_test,
        y_test_pred,
        target_names=selected_classes,
        output_dict=True
    )

    cm = confusion_matrix(y_test, y_test_pred)

    if progress_callback:
        progress_callback(1.0)

    history = {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc
    }

    return history, svm_model, selected_classes, report, cm

