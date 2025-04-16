# Classification_of_images_of_minerals_and_rock_cuts
Данный код представляет собой реализацию процесса обучения нейронной сети для задачи классификации изображений минералов и шлифов пород с использованием архитектуры EfficientNet-B3. Ниже приведено описание ключевых компонентов и этапов работы:
## Проверка доступности GPU
Код проверяет, доступен ли GPU (CUDA), и выбирает устройство для вычислений (GPU или CPU).
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
## Параметры обучения
Задаются параметры, такие как размер изображений (`img_size`), размер батча (`batch_size`), количество эпох (`num_epochs`), путь к данным (`data_dir`) и доля данных для обучения (`train_ratio`).

## Преобразования данных
Используются аугментации для увеличения разнообразия данных:<p>
* `RandomResizedCrop`: случайное изменение размера и обрезка изображений.<p>
* `RandomHorizontalFlip`: случайное отражение по горизонтали.<p>
* `RandomRotation`: случайный поворот изображений.<p>
* `ColorJitter`: случайное изменение яркости и контраста.<p>
* `RandomAffine`: случайное смещение изображений.<p>
* `ToTensor`: преобразование изображений в тензоры.<p>
* `Normalize`: нормализация данных.

## Загрузка данных
Данные загружаются с использованием `ImageFolder`, который автоматически разбивает данные на классы на основе структуры папок:<p> 
```
Data # Папка с классами
├── class_1
│   ├── image_1_1.jpg
│   ├── image_1_2.jpg
│   └── ...
├── class_2
│   ├── image_2_1.jpg
│   ├── image_2_2.jpg
│   └── ...
└── ...
```
Данные разделяются на обучающую и валидационную выборки с помощью `random_split`.

## Создание DataLoader
Для обучающей и валидационной выборок создаются `DataLoader`, которые обеспечивают пакетную загрузку данных.

## Модель EfficientNet-B3
Загружается предобученная модель `EfficientNet-B3` с измененным последним слоем для соответствия количеству классов в данных. Модель перемещается на выбранное устройство (GPU или CPU).

## Функция потерь и оптимизатор
* Используется `CrossEntropyLoss` с весами классов для учета дисбаланса данных.
* Оптимизатор — `AdamW` с learning rate scheduler (`CosineAnnealingWarmRestarts`).

## Ранняя остановка
Реализован механизм ранней остановки, если точность на валидационной выборке не улучшается в течение заданного числа эпох (`early_stop_patience`).

## Обучение модели
* В каждой эпохе модель обучается на обучающей выборке и валидируется на валидационной.
* Сохраняются метрики: `loss`, `accuracy`, `precision`, `recall` и `F1-score`.
* Лучшая модель сохраняется на основе точности на валидационной выборке.

## Визуализация данных
Реализована функция `show_images_with_titles` для отображения изображений с заголовками, соответствующими их классам.

## Оценка модели
После обучения строится `confusion matrix` и визуализируются метрики (`precision`, `recall`, `F1-score`) для каждого класса.

## Сохранение модели
Сохраняются веса лучшей модели (`best_model.pth`), последней модели (`last_model.pth`).

## Используемые библиотеки
* `torch`: PyTorch для работы с нейронными сетями.
* `torchvision`: для загрузки данных и аугментаций.
* `sklearn.metrics`: для расчета метрик (`precision`, `recall`, `F1-score`).
* `matplotlib` и `seaborn`: для визуализации данных и метрик.
* `efficientnet_pytorch` для импорта `EfficientNet`.

## Графики метрик и пример использования
### Классификация изображений минералов
![image](https://github.com/user-attachments/assets/ce3545ed-1041-4c51-a76b-eab18e0f784c)
![image](https://github.com/user-attachments/assets/8e36eccd-a449-402e-90cf-14c5a5b165be)
![image](https://github.com/user-attachments/assets/ca56b40b-fae1-4ef8-bc3a-6cce45c99c08)
![image](https://github.com/user-attachments/assets/6272ec5b-9218-4559-bf44-c654a0aefa62)
![image](https://github.com/user-attachments/assets/021443ea-c5fa-4c46-8add-55d28c619a3b)
![image](https://github.com/user-attachments/assets/b835e89b-a622-42f6-aea5-5bd609d76266)
![image](https://github.com/user-attachments/assets/bb912bb4-54ca-46c0-bd97-ea1173f31537)
![image](https://github.com/user-attachments/assets/177d3079-2570-4889-b368-56b13e029111)
![image](https://github.com/user-attachments/assets/d0b70a1b-eaf7-4a26-b0f6-dad7284f0f89)
![image](https://github.com/user-attachments/assets/fcc6d9e5-5456-4c01-8cb5-3610590b678d)
![image](https://github.com/user-attachments/assets/f6ae563b-ad53-4212-b726-0df8e35df497)
![image](https://github.com/user-attachments/assets/1b613b25-0cfe-49eb-903e-91e25de0ec64)


### Классификация изображений шлифов пород
![image](https://github.com/user-attachments/assets/84846144-a5c6-4df6-9e84-02857b299b5b)
![image](https://github.com/user-attachments/assets/442b68e2-a726-4b3b-9137-133514ff6462)
![image](https://github.com/user-attachments/assets/b666766d-fcef-4319-b57d-b56e78c68862)
![image](https://github.com/user-attachments/assets/a6be71fe-e746-4528-8c99-1030386bd4a0)
![image](https://github.com/user-attachments/assets/1f7fc746-6dcb-4c2a-906d-3854326482b4)
![image](https://github.com/user-attachments/assets/0ad0bc64-53f1-40d6-a5af-3468596cfbdb)
![image](https://github.com/user-attachments/assets/5adf13a8-ac15-4ccc-af35-25de52ce2e8b)
![image](https://github.com/user-attachments/assets/9ddc1016-e41a-4deb-98fe-1dfc4c6189b9)
![image](https://github.com/user-attachments/assets/f702893f-0afe-4ee0-9b15-df0e9c3b291f)
![image](https://github.com/user-attachments/assets/1dadf142-e9c9-4301-9d7d-e266bdb8cb23)
![image](https://github.com/user-attachments/assets/93aaf36d-21cc-42cf-9749-cd5724620c75)

## Лицензия
Этот проект распространяется под лицензией MIT. Подробности см. в файле `LICENSE`.

## Ссылки
Ссылки на данные: https://www.kaggle.com/datasets/asiedubrempong/minerals-identification-dataset<p>
Ссылка на конвертацию PyTorch модели в ONNX формат: https://github.com/DaniilKostashchuk/Converting_the_Pwtorch_model_to_ONNX_format<p>
Сылка на TG-бота: https://github.com/DaniilKostashchuk/Telegram_Bot_for_Mineral_Classification
