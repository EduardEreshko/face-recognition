from sklearn.svm import SVC
from sklearn.decomposition import PCA, KernelPCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def create_model(n_components=200, kernel='poly', random_state=42):
    """
    Создает модельный пайплайн для классификации лиц.
    
    Args:
        n_components (int): Количество компонент для KernelPCA
        kernel (str): Ядро для KernelPCA
        random_state (int): Seed для воспроизводимости
        
    Returns:
        sklearn.pipeline.Pipeline: Готовый модельный пайплайн
    """
    pca = KernelPCA(n_components=n_components, kernel=kernel, random_state=random_state)
    scaler = StandardScaler()
    model_lr = LogisticRegression(random_state=random_state, max_iter=1000)
    
    model = make_pipeline(scaler, pca, model_lr)
    return model


def create_svm_model(n_components=200, kernel='poly', random_state=42):
    """
    Создает модельный пайплайн с SVM для классификации лиц.
    
    Args:
        n_components (int): Количество компонент для KernelPCA
        kernel (str): Ядро для KernelPCA
        random_state (int): Seed для воспроизводимости
        
    Returns:
        sklearn.pipeline.Pipeline: Готовый модельный пайплайн с SVM
    """
    pca = KernelPCA(n_components=n_components, kernel=kernel, random_state=random_state)
    scaler = StandardScaler()
    model_svm = SVC(random_state=random_state, kernel='rbf')
    
    model = make_pipeline(scaler, pca, model_svm)
    return model
