import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.layers import Dense, Dropout, Reshape, MultiHeadAttention, LayerNormalization, Add, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
# dataset
base_dir = 'C:/Users/Donete/Documents/prog'  
# Создание генераторов данных с дополнением для поезда
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
# Загрузка изображений
train_generator = train_datagen.flow_from_directory(
    directory=base_dir + '/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    directory=base_dir + '/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    directory=base_dir + '/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
# Создание модели CNN
cnn_model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(256, activation='relu')
])
cnn_model.summary()

inputs = layers.Input(shape=(224, 224, 3))
x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Dropout(0.3)(x)  
# Transformer
x = layers.Reshape((26*26, 128))(x)
attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
x = layers.Add()([x, attention_output])
x = layers.LayerNormalization()(x)
ffn = layers.Dense(256, activation='relu')(x)
ffn = layers.Dense(128)(ffn)
x = layers.Add()([x, ffn])
x = layers.LayerNormalization()(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)  #  Dropout final
outputs = layers.Dense(5, activation='softmax')(x)
model = Model(inputs, outputs)
model.summary()
# Компиляция модели
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
#  EarlyStopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))
# Entraînement avec EarlyStopping
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    callbacks=[early_stop],
    class_weight=class_weights
)

# Сохранение обученной модели
model.save("модель_CNN_Трансформер.keras")
# Оценка модели
score = model.evaluate(test_generator)
print("Précision finale sur le test set :", round(score[1]*100, 2), "%")
# Список классов (в том же порядке, что и папки в наборе данных)
class_labels = ['Выбоины на проезжей части', 'Дорога без видимых повреждений', 'Нормальные дорожные знаки', 'Повреждённые дорожные знаки', 'Продольные трещины']
# Предсказания
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes
# Пользовательские цвета для каждого класса
couleurs = ['skyblue', 'orange', 'lightgreen', 'salmon', 'violet']
# Подсчитать предсказанные классы
from collections import Counter
counts = Counter(y_pred_classes)
# Показать точность на валидации
print("Итоговая точность на валидационной выборке :", round(history.history['val_accuracy'][-1]*100, 2), "%")

# Отчёт о классификации
from sklearn.metrics import classification_report, confusion_matrix
print("Отчёт о классификации :")
print(classification_report(y_true, y_pred_classes, target_names=class_labels, zero_division=0))

# Матрица путаницы
cm = confusion_matrix(y_true, y_pred_classes)

# Graphique : classes identifiées
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
bars = plt.bar(
    [class_labels[i] for i in counts.keys()],
    counts.values(),
    color=couleurs
)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 2, str(height), ha='center', fontsize=10)

plt.title("Распределение классов определенных моделью")
plt.xlabel("Предсказанный класс")
plt.ylabel("Количество изображений")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("graphique_classes11.png", dpi=300)


# Matrice de confusion
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis',
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Прогнозируемый', fontsize=12)
plt.ylabel('Реальный', fontsize=12)
plt.title('Матрица путаницы', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("fina222.png")

# Распознанные изображения
import random
indices = list(range(len(y_pred_classes)))
random.shuffle(indices)

plt.figure(figsize=(15, 10))
for i in range(20):
    idx = indices[i]
    img_batch, _ = test_generator[idx // test_generator.batch_size]
    img = img_batch[idx % test_generator.batch_size]
    plt.subplot(4, 5, i + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Прогноз : {class_labels[y_pred_classes[idx]]}", fontsize=9)

plt.suptitle("Пример 20 изображений, для которых модель правильно определила класс", fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig("20_images_identifiees222.png", dpi=300)

# Courbe de perte
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], 'o-', label='Тренировка (потеря)')
plt.plot(history.history['val_loss'], '-', label='Проверка (потеря)')
plt.title("Потери на тренировочном и проверочном наборе")
plt.xlabel("Эпоха")
plt.ylabel("Потери (loss)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Pertefinal222.png")

#  Courbe de précision
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], 'o-', label='Тренировка (точность)')
plt.plot(history.history['val_accuracy'], '-', label='Проверка (точность)')
plt.title("Точность на тренировочном и проверочном наборе")
plt.xlabel("Эпоха")
plt.ylabel("Точность (accuracy)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("final222.png")

#  Résumé console
print("Clés disponibles dans l'historique :", history.history.keys())
total = len(y_true)
nb_corrects = np.sum(y_true == y_pred_classes)
accuracy_test = round((nb_corrects / total) * 100, 2)

print(f"Nombre d’images dans le test : {total}")
print(f"Nombre d’images correctement détectées : {nb_corrects}")
print(f"Précision calculée sur le test : {accuracy_test} %")

# === ПОСЛЕДОВАТЕЛЬНЫЕ ДАННЫЕ ДЛЯ ПРОГНОЗИРОВАНИЯ ЭВОЛЮЦИИ ===
base_path = "C:/Users/Donete/Documents/4 semestre/BKR/sequences"
class_labels = ['Выбоины (высокая степень)', 'Выбоины (средняя степень)', 'Трещины продольные (средняя ширина)', 'Трещины продольные (широкие)']
img_size = (224, 224)
sequence_length = 3

X = []
y = []

valid_ext = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

def imread_unicode(path):
    stream = open(path, "rb")
    bytes_array = bytearray(stream.read())
    numpy_array = np.asarray(bytes_array, dtype=np.uint8)
    return cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)

class_labels = [
    'Трещины продольные (широкие)',
    'Трещины продольные (средняя ширина)',
    'Выбоины (высокая степень)',
    'Выбоины (средняя степень)',
]

for idx, class_name in enumerate(class_labels):

    class_folder = os.path.join(base_path, class_name)
    for seq_name in os.listdir(class_folder):
        seq_path = os.path.join(class_folder, seq_name)
        images = []

        for img_file in sorted(os.listdir(seq_path)):
            if not img_file.lower().endswith(valid_ext):
                continue

            img_path = os.path.join(seq_path, img_file)
            img = imread_unicode(img_path)
            if img is None:
                print(f" Image introuvable ou illisible : {img_path}")
                continue

            img = cv2.resize(img, img_size)
            img = img.astype("float32") / 255.0
            images.append(img)

        if len(images) == sequence_length:
            X.append(images)
            y.append(idx)
        else:
            print(f"Игнорируемая последовательность (неполные или нечитаемые изображения) : {seq_path}")

if len(X) == 0:
    raise RuntimeError(" Не найдено ни одной допустимой последовательности.")

X = np.array(X)
y = to_categorical(np.array(y), num_classes=len(class_labels))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === МОДЕЛЬ CNN + LSTM ДЛЯ ПРОГНОЗИРОВАНИЯ ЭВОЛЮЦИИ ===
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import TimeDistributed, GlobalAveragePooling2D, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import TimeDistributed, GlobalAveragePooling2D, LSTM, Dense, Input
from tensorflow.keras.models import Model

sequence_length = 3  # ou ce que tu utilises déjà
input_seq = Input(shape=(sequence_length, 224, 224, 3))

# Загрузить MobileNetV2 без скачивания
base_cnn = MobileNetV2(
    include_top=False,
    input_shape=(224, 224, 3),
    weights=None
)

# Charger les poids manuellement
weights_path = "C:/Users/Donete/Documents/4 semestre/BKR/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
base_cnn.load_weights(weights_path)

# Geler les couches CNN
base_cnn.trainable = False

# Architecture CNN + LSTM
x = TimeDistributed(base_cnn)(input_seq)
x = TimeDistributed(GlobalAveragePooling2D())(x)
x = LSTM(128)(x)
x = Dense(64, activation='relu')(x)
output = Dense(len(class_labels), activation='softmax')(x)

model_seq = Model(inputs=input_seq, outputs=output)
model_seq.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_seq.summary()

# === ОБУЧЕНИЕ ВРЕМЕННОЙ МОДЕЛИ ===
history_seq = model_seq.fit(
    X_train, y_train,
    epochs=50,
    batch_size=4,
    validation_data=(X_test, y_test),
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
)

# Sauvegarde
model_seq.save("модель_прогноза_эволюции.keras")

# === ВИЗУАЛИЗАЦИЯ ПРОГНОЗА ЭВОЛЮЦИИ ===
import matplotlib.pyplot as plt
import random

idx = random.randint(0, len(X_test)-1)
sequence = X_test[idx]
true_label = np.argmax(y_test[idx])
prediction = model_seq.predict(sequence[np.newaxis, ...])
pred_label = np.argmax(prediction)

jours = ['1 jour', '10 jours', '20 jours']
plt.figure(figsize=(12, 4))
for i, img in enumerate(sequence):
    plt.subplot(1, 3, i + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{jours[i]}", fontsize=12)

plt.suptitle(f"Classe réelle : {class_labels[true_label]} | Prédiction : {class_labels[pred_label]}", fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.savefig("prediction_evolution_sequence.png", dpi=300)

# Graphique horizontal des probabilités 
plt.figure(figsize=(6, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
bars = plt.barh(class_labels, prediction[0], color=colors)

plt.xlim(0, 1.0)
plt.xlabel("Предсказанная вероятность", fontsize=12)
plt.title("Прогнозирование развития дефектов", fontsize=14)

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{width:.2f}", va='center', fontsize=10)

plt.tight_layout()
plt.savefig("probabilite_evolution1609.png", dpi=300)

# Предупреждающее сообщение на основе официальных названий
classe_predite = class_labels[pred_label]

if 'высокая степень' in classe_predite or 'широкие' in classe_predite:
    print(" ТРЕВОГА: Рекомендуется срочное вмешательство (прогнозируется высокая степень тяжести).")
elif 'средняя степень' in classe_predite or 'средняя ширина' in classe_predite:
    print(" Внимание: Рекомендуется усиленное наблюдение")
else:
    print(" Ситуация стабильна, немедленное вмешательство не требуется")
plt.show()


