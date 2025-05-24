# 📝 Türkçe Ürün Yorumları Duygu Analizi

Bu proje, Türkçe ürün yorumlarının duygu analizini (Pozitif mi? Nötr mü? yoksa Negatif mi?) yapan ve arka plan'da birçok derin öğrenme
algoritması barındıran bir projedir. LSTM, Bidirectional LSTM, GRU algoritmalarıyla eğitilen birden fazla model kullanılabilir. UI Kıs
mında ise Gradio kullanılmıştır. Kullanıcı istediği model'i seçebilir ve ardından yorum girip yorumun analizini yapabilir.

## 🚀 Arayüz

![Kullanıcı Arayüzü](/images/ui.png)

## 🔥 Kullanılan Algoritmalar

1. **LSTM**

2. **Bidirectional LSTM**

3. **GRU**

## 📈 Kullanılan Veri Seti

Modelin eğitimi sırasında yaklaşık olarak 450.000 veri, testinde ise 50.000 veri kullanılmıştır.

![Dataset](/images/dataset-example.png)

Bu görsel veri setinin ilk 5 satırını gösteriyor. Veri seti 3 adet sütundan oluşmaktadır. Bu sütunlar text, label, dataset olarak nitelendirilmiştir. Burada bizi ilgilendiren kısım text ve label'dır.

[Bu link ile veri setine erişebilirsiniz!](https://www.kaggle.com/datasets/winvoker/turkishsentimentanalysisdataset)

## ✨ Eğitim Ortamı

**Donanım Bilgisi**

Model eğitimi için Google Colab kullanılmıştır. Burada yer alan;

1. A100 40 GB VRAM GPU Kullanılmıştır. 
2. L4 GPU kullanılmıştır.
3. T4 GPU Kullanılmıştır.

## 🚀 Eğitim Ayarları

3 Adet farklı algoritma kullanılmıştır. Öncelikle LSTM Algoritmasına değinebiliriz.

**1. LSTM Algoritması Ayarları**

```python
class LSTMClassifier(nn.Module):  # LSTM tabanlı sınıflandırıcı sınıfı
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx, n_layers=1, dropout=0.3):  # Sınıf kurucu metodu
        super().__init__()  # Üst sınıfın kurucu metodunu çağır
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)  # Sözcükleri sayısal vektörlere dönüştür
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            dropout=dropout, batch_first=True)  # LSTM katmanını tanımla
        self.fc = nn.Linear(hidden_dim, output_dim)  # Son katmanda sınıflandırma yap
        self.dropout = nn.Dropout(dropout)  # Aşırı öğrenmeyi engellemek için dropout uygula

    def forward(self, text, lengths):  # İleri yayılım fonksiyonu
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)  # Değişken uzunluktaki dizileri paketle
        packed_output, (hidden, _) = self.lstm(packed_embedded)  # LSTM katmanından geç
        output = self.fc(self.dropout(hidden[-1]))  # Dropout sonrası tam bağlantılı katman
        return output  # Son tahmin değerini döndür
```

**2. Bidirectional LSTM Algoritması Ayarları**

```python
class BiLSTM(nn.Module):  # BiLSTM sınıfı, PyTorch'un nn.Module sınıfından türetilir
    def __init__(self, vocab_size, emb_dim, hid_dim, out_dim, pad_idx, n_layers=1, dropout=0.5):  # Yapıcı metot, modelin genel ayarlarını yapar
        super().__init__()  # Üst sınıfın yapıcısını çağırır
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)  # Giriş metinlerini sayısal vektörlere dönüştürür
        self.lstm = nn.LSTM(
            emb_dim,        # Giriş katman boyutu
            hid_dim,        # Gizli katman boyutu
            num_layers=n_layers,    # LSTM katman sayısı
            bidirectional=True,     # Çift yönlü LSTM
            batch_first=True,       # Veri sırası: (batch, seq, feature)
            dropout=dropout if n_layers>1 else 0.0  # Katman sayısına göre dropout oranı
        )
        self.fc = nn.Linear(hid_dim * 2, out_dim)  # Çift yönlü LSTM çıktısını sınıflandırma katmanına gönderir
        self.drop = nn.Dropout(dropout)  # Aşırı öğrenmeyi önlemek için dropout uygular

    def forward(self, texts, lengths):  # İleri yayılım fonksiyonu
        emb = self.drop(self.embedding(texts))  # Metin gömmeleri oluşturur ve dropout uygular
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)  # Değişken uzunluktaki dizileri paketler
        _, (hidden, _) = self.lstm(packed)  # LSTM katmanını çalıştırarak gizli durum bilgisine ulaşır
        h_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)  # İleri ve geri yönlü çıktıları birleştirir
        return self.fc(self.drop(h_cat))  # Çıktıyı tam bağlı katmandan geçirir ve döndürür
```

**3. GRU Algoritması Ayarları**

```python
class GRUClassifier(nn.Module):  # GRU tabanlı sınıflandırıcı sınıfı
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, pad_idx):  # Sınıfı başlatmak için gerekli parametreleri tanımla
        super().__init__()  # Üst sınıfın kurucu metodunu çağır
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)  # Metinleri sayısal vektörlere dönüştür
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)  # İki yönlü (bidirectional) GRU katmanı tanımla
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Çift yönlü GRU çıktısını sınıflandırma katmanına aktar
        self.dropout = nn.Dropout(0.3)  # Overfitting'i önlemek için dropout uygula

    def forward(self, input_ids):  # Modelin ileri yayılım fonksiyonu
        x = self.embedding(input_ids)  # Girişleri embedding katmanından geçir
        _, h = self.gru(x)  # GRU katmanından çıktı al, yalnızca gizli durum kullan
        h = torch.cat((h[-2], h[-1]), dim=1)  # İleri ve geri yönlü son katmanları birleştir
        h = self.dropout(h)  # Dropout uygula
        return self.fc(h)  # Çıktıyı sınıflandırma katmanından geçir ve döndür
```

## 📈 Eğitim Raporları

**1. Bidirectional LSTM Raporları**

![Epoch 1](/images/bidirectional-lstm-epoch-1.png)

![Epoch 2](/images/bidirectional-lstm-epoch-2.png)

![Epoch 3](/images/bidirectional-lstm-epoch-3.png)

![Epoch 4](/images/bidirectional-lstm-epoch-4.png)

![Epoch 5](/images/bidirectional-lstm-epoch-5.png)

**2. LSTM Raporları**

![Epoch 1](/images/lstm-epoch-1.png)

![Epoch 2](/images/lstm-epoch-2.png)

![Epoch 3](/images/lstm-epoch-3.png)

![Epoch 4](/images/lstm-epoch-4.png)

![Epoch 5](/images/lstm-epoch-5.png)

![Epoch 6](/images/lstm-epoch-6.png)

![Epoch 7](/images/lstm-epoch-7.png)

![Epoch 8](/images/lstm-epoch-8.png)

![Epoch 9](/images/lstm-epoch-9.png)

![Epoch 10](/images/lstm-epoch-10.png)

**3. GRU Raporları**

![GRU Reports](/images/gru-reports.png)

![GRU Classification Report](/images/gru-classification-report.png)

![GRU Confusion Matrix](/images/gru-confusion-matrix.png)

## 🌐 Dosya Yapısı

```bash
.
├── app.py                                                  # Gradio tabanlı duygu analizi uygulama arayüzü
├── final_nlp_deep_learning_turkish_sentiment_analysis      # Bidirectional LSTM ile eğitim          
├── models/                                                 # Modellerin barındırıldığı klasör
│──── bidirectional_lstm/
│        ├── algorithm.py                                   # RNN sınıfını içeren dosya
│        ├── itos.pkl                                       # index-to-string sözlüğü
│        ├── model.pth                                      # Eğitilmiş model dosyası
│        ├── stoi.pkl                                       # string-to-index sözlüğü
│──── lstm/                                                
│        ├── algorithm.py
│        ├── itos.pkl
│        ├── model.pth
│        ├── stoi.pkl
├── utils/                                                  # Ortak kullanılan metotların barındırıldığı klasör
│   ├── string.py
│   └── ...
├── requirements.txt                                        # Gerekli Python kütüphaneleri
└── README.md                                               # Proje açıklamaları
```

## 📦 Gereksinimler

* Python >= 3.8
* Aşağıdaki kurulumları yapmanız gerekmektedir.
* **(Opsiyonel)** CUDA Destekli Cihaz veya Google Colab PRO Üyeliği

## 🛠️ Kurulum

**1. Gerekli Kütüphanelerin Kurulumu**

```bash
pip install -r requirements.txt
```

*MacOS için;*

```bash
pip3 install -r requirements.txt
```

**2. Gradio Uygulamasını Çalıştırın**

```bash
python3 app.py
```

## 📚 Kullanılan Teknolojiler

* [PyTorch](https://pytorch.org/)
* [Scikit-Learn](https://scikit-learn.org/)
* [Gradio](https://www.gradio.app/)
* [Kagglehub](https://www.kaggle.com/)

## 📜 Lisans

Bu proje `MIT Lisansı` ile lisanslanmıştır. Daha fazla bilgi için `LICENSE` dosyasına göz atabilirsiniz.