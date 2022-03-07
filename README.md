## Задача по распознаванию рукописного текста в школьных тетрадях
#### Соревнование проводилось в рамках олимпиады НТО, при поддержке Сбера на платформе [ODS](https://ods.ai/competitions/nto_final_21-22).

### [Результаты Public](https://ods.ai/competitions/nto_final_21-22/leaderboard)
![leaderbord](https://cdn.discordapp.com/attachments/660500818180440084/950114565029523506/unknown.png)

### Задача
> Вам нужно разработать алгоритм, который способен распознать рукописный текст в школьных тетрадях. В качестве входных данных вам будут предоставлены фотографии целых листов. Предсказание модели — список распознанных строк с координатами полигонов и получившимся текстом.
---

### Как должно работать решение?
> Последовательность двух моделей: сегментации и распознавания. Сначала сегментационная модель предсказывает полигоны маски каждого слова на фото. Затем эти слова вырезаются из изображения по контуру маски (получаются кропы на каждое слово) и подаются в модель распознавания. В итоге получается список распознанных слов с их координатами.
---

### Модели

**Instance Segmentation**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Anonumous796/2nd-place-solution-NTO-AI-2022/blob/main/train/model_for_detection.ipynb)

- Мы использовали модель [R101-FPN](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#coco-instance-segmentation-baselines-with-mask-r-cnn) из зоопарка моделей detectron2 в совокупности с аугментациями и высоким разрешением

**Optical Character Recognition (OCR)**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Anonumous796/2nd-place-solution-NTO-AI-2022/blob/main/train/model_for_ocr.ipynb)

- архитектура CRNN с бекбоном Resnet-34, предобученным на синтезе от [StackMix](https://github.com/sberbank-ai/StackMix-OCR)

### Вычислительные мощности & Submit`
---
**Christofari** с **NVIDIA Tesla V100** и образом **jupyter-cuda11.0-tf2.4.0-gpu:0.0.76**

Submit можно найти по ссылкам:
[One Drive](https://1drv.ms/u/s!AkPiJU-1XuQSgYkM5DJkxnywM8MfqQ?e=JCEel1) и 
[Yandex](https://storage.yandexcloud.net/datasouls-ods/submissions/e7c3d807-0f20-4003-9935-977432b4d615/2d91525d/ocr_submit%20%2810%29.zip)

### Цитирование
```BibTeX
@misc{nto-ai-text-recognition,
  author =       {Ilia Kuleshov and Danil Astafurov},
  title =        {notebook-recognition},
  howpublished = {\url{https://github.com/Anonumous796/2nd-place-solution-NTO-AI-2022}},
  year =         {2022}
}
```