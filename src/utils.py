import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


def evaluate_model(model, Xtrain, ytrain, Xtest, ytest, num_augmentations, get_aug_function):
    """
    Оценивает модель с аугментацией данных.
    
    Args:
        model: Модель для обучения
        Xtrain: Обучающие данные
        ytrain: Метки обучающих данных
        Xtest: Тестовые данные
        ytest: Метки тестовых данных
        num_augmentations: Количество аугментаций на одно изображение
        get_aug_function: Функция для аугментации
        
    Returns:
        tuple: (accuracy, predictions)
    """
    XtrainAug = []
    ytrainAug = []

    for (img, label) in zip(Xtrain, ytrain):
        # Всегда добавляем оригинальное изображение
        XtrainAug.append(img)
        ytrainAug.append(label)

        # Добавляем аугментированные версии
        for i in range(num_augmentations):
            aug_img = get_aug_function(img)
            XtrainAug.append(aug_img)
            ytrainAug.append(label)

    # Преобразуем и нормализуем
    XtrainAug_flat = np.asarray([el.ravel() for el in XtrainAug], dtype=np.float32) / 255.0

    model.fit(XtrainAug_flat, ytrainAug)
    pred = model.predict(Xtest)

    acc = accuracy_score(ytest, pred)
    return acc, pred


def plot_predictions_with_labels(Xtest_list, ytest_list, pred_list, n_images=10):
    """
    Отображает изображения с истинными и предсказанными метками.
    
    Args:
        Xtest_list: Список тестовых изображений
        ytest_list: Список истинных меток
        pred_list: Список предсказанных меток
        n_images: Количество изображений для отображения
    """
    n_rows = 2
    n_cols = n_images // n_rows
    fig, axx = plt.subplots(n_rows, n_cols, figsize=(15, 6))
    axx = axx.flatten()

    for i in range(min(n_images, len(Xtest_list))):
        axx[i].imshow(Xtest_list[i].reshape(64, 64), cmap='gray')
        true_label = ytest_list[i]
        pred_label = pred_list[i]
        title = f"True: {true_label}\nPred: {pred_label}"
        if true_label != pred_label:
            axx[i].set_title(title, color='red', fontsize=10)
        else:
            axx[i].set_title(title, color='green', fontsize=10)
        axx[i].axis('off')

    plt.suptitle('Predicted Names; Incorrect Labels in Red', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_accuracy_results(results):
    """
    Визуализирует результаты точности модели.
    
    Args:
        results (dict): Словарь с количеством аугментаций и соответствующей точностью
    """
    plt.figure(figsize=(8, 5))
    plt.plot(results.keys(), results.values(), marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Augmentations per Image')
    plt.ylabel('Accuracy')
    plt.title('Effect of Data Augmentation on Model Accuracy')
    plt.grid(True, alpha=0.3)
    plt.xticks(list(results.keys()))
    plt.ylim(0.3, 1.0)
    plt.show()


def print_classification_report(ytest, predictions, labels=None):
    """
    Выводит расширенный отчет о классификации.
    
    Args:
        ytest: Истинные метки
        predictions: Предсказанные метки
        labels: Список меток для отображения
    """
    print("=== Classification Report ===")
    print(classification_report(ytest, predictions, labels=labels))
    
    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(ytest, predictions, labels=labels)
    print(cm)
    
    # Визуализация матрицы ошибок
    if labels:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()


def plot_sample_images(X, y, n_images=5):
    """
    Отображает примеры обработанных изображений.
    
    Args:
        X: Список изображений
        y: Список меток
        n_images: Количество изображений для отображения
    """
    fig, axx = plt.subplots(1, n_images, figsize=(12, 3))
    for i in range(n_images):
        img = X[i]
        axx[i].imshow(img, cmap='gray')
        axx[i].set_title(y[i])
        axx[i].axis('off')
    plt.tight_layout()
    plt.show()
