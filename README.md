Aygaz Yapay Zekaya Giriş Bootcamp Projesi

Bu projede Kaggle üzerinden elde ettiğim Brain Tumor MRI Dataset üzerinde bulunan dört farklı sınıf etiketinde sınıflandırma uygulaması yapıyorum.

Veri Setinin Linki: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

4 adet sınıf etiketimiz bulunmaktadır bunlar:
  - glioma
  - meningioma
  - notumor
  - pituitary

Bu sınıflardan 3 tanesi tümör içeren beyin mri görüntülerini temsil etmektedir (glioma, meningioma, pituitary), geriye kalan 1 sınıf ise sağlıklı bir beyin mri görüntüsünü temsil etmektedir (notumor).

Bu çalışma da Tensorflow ve Keras kütüphaneleri kullanılarak yapıldı.

# Kullanılan veri setinin rasgele seçilen örnek görselleri
![image](https://github.com/berkandymn/ImageClasssification/assets/86782845/b53e1cc8-b40d-4f88-b59f-8e194c6eb997)

Çalışmada iki farklı model kullanılmıştır. 
  - İlk olarak kendi oluşturduğumuz bir model
  - ikinci olarak MobileNetV3 Large modeli kullanılmıştır

Kendi oluşturduğumuz model daha za katman sayısına ve karmaşıklığa sahip, MobileNetV3 ise daha karmaşık ve yeterli veri ile yüksek doğrulukta sonuçlar verebilmektedir.

Modellerin eğitiminden ve sonuç doğurlamasından sonra model eğitim sürecinde doğruluk değerlerinin validation şlemi ile nasıl değişiklik gösterdiği bir grafik ile görselleştirilmiştir.

![image](https://github.com/berkandymn/ImageClasssification/assets/86782845/89fc6be7-ffdd-47ee-9c9f-bb8ff75e9390)

Daha sonra modelin doğruluk değerleri ve tahmin sürecinin sonuçlarını içeren Confusion Matrix görselleştirlmiştir.

![image](https://github.com/berkandymn/ImageClasssification/assets/86782845/3b01e7f6-4bcb-4dc2-aef6-1f9cb999b78b)

# Performans metriklerinin değerleri

Eğitim sonunda kendi oluşturduğumuz modelde performans değerleri şu şekildedir:
  - Loss: 0.0481 
  - Accuracy: 0.9870
  - Validation Loss: 1.2302
  - Validation Accuracy: 0.6944
  - F1 Score: 0.257522499234347
  - Recall: 0.27635327635327633
