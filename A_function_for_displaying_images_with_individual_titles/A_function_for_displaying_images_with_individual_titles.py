def show_images_with_titles(images, titles, nrows=4, ncols=8):
  
    # Преобразуем тензор в numpy и меняем порядок осей (C, H, W) -> (H, W, C)
    images = images.cpu().numpy().transpose((0, 2, 3, 1))
    
    # Обратная нормализация
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = std * images + mean
    images = np.clip(images, 0, 1)  # Ограничиваем значения в диапазоне [0, 1]

    # Создаем фигуру
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10), dpi=200)  # Размер фигуры
    fig.subplots_adjust(hspace=0.5)  # Отступ между изображениями

    # Отображаем каждое изображение с заголовком
    for i, (image, title) in enumerate(zip(images, titles)):
        ax = axes[i // ncols, i % ncols]  # Определяем позицию изображения в сетке
        ax.imshow(image)
        ax.set_title(title, fontsize=10)  # Заголовок для каждого изображения
        ax.axis('off')  # Отключаем оси

    # Скрываем пустые subplots, если изображений меньше, чем nrows * ncols
    for i in range(len(images), nrows * ncols):
        axes[i // ncols, i % ncols].axis('off')

    plt.show()

# Получение одного батча данных
inputs, classes = next(iter(dataloaders['train']))

# Подготовка заголовков
titles = [class_names[x] for x in classes]

# Визуализация
show_images_with_titles(inputs, titles, nrows=4, ncols=8)  # 4 строки, 8 столбцов
