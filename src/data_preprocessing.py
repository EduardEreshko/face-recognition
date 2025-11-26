"""# Функции"""

def get_aug(image):
    angle = np.arange(-10, 11, 1)
    angle0 = random.choice(angle)
    shift = 0.01 * np.arange(-10, 11, 1)
    shift0 = random.choice(shift)
    transform = A.ShiftScaleRotate(
        shift_limit=shift0,
        rotate_limit=angle0,
        scale_limit=0,
        p=0.5
    )
    augmented_image = transform(image=image)['image']
    transform = A.HorizontalFlip(p=0.5)
    return transform(image=augmented_image)['image']

def filter_directories(dirname):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    valid_directories = []

    for item in os.listdir(dirname):
        full_path = os.path.join(dirname, item)
        if not os.path.isdir(full_path):
            continue

        try:
            # Быстрая проверка первых 5 файлов
            files = os.listdir(full_path)[:5]
            if not files:  # пустая директория
                continue

            if all(os.path.splitext(f)[1].lower() in image_extensions
                   for f in files if os.path.isfile(os.path.join(full_path, f))):
                valid_directories.append(item)
        except (PermissionError, OSError) as e:
            print(f"Error accessing {full_path}: {e}")
            continue

    return valid_directories

"""# Обработка изображений"""

dirname = "/content/drive/MyDrive/Stepik/Старт в ML: создайте свою систему распознавания лиц/Start_ML"
labels = filter_directories(dirname)[:126:25]  # чтобы долго не ждать обработки проходимся лишь по некоторым папкам с изображениями

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
X, y = [], []

for label in labels:
    subdir = os.path.join(dirname, label)
    filelist = os.listdir(subdir)
    for fname in filelist:
        img_path = os.path.join(subdir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f'Failed to load image: {img_path}')
            continue

        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

        faces = face_cascade.detectMultiScale(img)
        if len(faces) == 0:
            print(f'no face found for {fname}')
            continue

        # Берем первое найденное лицо
        (a, b, w, h) = faces[0]
        face_img = img[b:b + h, a:a + w]

        try:
            h_face, w_face = face_img.shape
            size = min(h_face, w_face)
            h0 = int((h_face - size) / 2)
            w0 = int((w_face - size) / 2)
            cropped_face = face_img[h0: h0 + size, w0: w0 + size]
            cropped_face = cv2.resize(cropped_face, (64, 64), interpolation=cv2.INTER_AREA)
            X.append(cropped_face)
            y.append(label)
        except Exception as e:
            print(f'Error processing {fname}: {e}')

len(X), len(y)
