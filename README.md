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


### Классификация изображений шлифов пород
![image](https://github.com/user-attachments/assets/460b2eed-7229-4cf0-8e88-556883588bcf)
![image](https://github.com/user-attachments/assets/a7803afc-3180-4a16-8e67-12e9830bac87)
![image](https://github.com/user-attachments/assets/881ff2bb-7376-4add-bc81-60881611ab03)
![image](https://github.com/user-attachments/assets/1e12f21c-1704-4cfa-807a-690b0e734172)
![image](https://github.com/user-attachments/assets/bb593699-161d-4f7a-a4b0-a2ca288e0eab)
![image](https://github.com/user-attachments/assets/cce9de8d-c67b-43fb-af1f-a9a1a580b613)
![image](https://github.com/user-attachments/assets/8ba28f11-5db6-4ebe-905a-7e3abdce6ed6)
![image](https://github.com/user-attachments/assets/2df1b42e-803b-4947-9615-09c386dc6423)
![image](https://github.com/user-attachments/assets/7f4a5582-24cd-4c63-a683-de24cd4e2fb4)
![image](https://github.com/user-attachments/assets/3b7705a8-0e1f-4b13-b940-ee48e4d788c5)
![image](https://github.com/user-attachments/assets/a5399dcf-655b-4427-a2bc-4f60468e3f8b)





## Лицензия
Этот проект распространяется под лицензией MIT. Подробности см. в файле `LICENSE`.

## Ссылки на данные
https://www.kaggle.com/datasets/asiedubrempong/minerals-identification-dataset
