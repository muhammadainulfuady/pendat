# Penjelasan Sel 1: Import Library

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from IPython.display import display
```

### Penjelasan per baris:

| Library | Fungsi |
|---------|--------|
| `pandas` | Membaca & mengolah data dalam bentuk tabel (DataFrame) |
| `numpy` | Operasi matematika & array numerik |
| `matplotlib.pyplot` | Membuat grafik/visualisasi dasar (bar chart, histogram, dll) |
| `seaborn` | Visualisasi statistik yang lebih cantik (heatmap, boxplot) |
| `LabelEncoder` | Mengubah data kategorikal → angka (misal: "laki-laki" → 0) |
| `MinMaxScaler` | Normalisasi data ke rentang 0–1 |
| `train_test_split` | Membagi data jadi training & testing |
| **`AdaBoostClassifier`** | **Model utama** — algoritma ensemble boosting untuk klasifikasi |
| **`DecisionTreeClassifier`** | **Base estimator** untuk AdaBoost (weak learner / decision stump) |
| `accuracy_score` | Menghitung akurasi model |
| `classification_report` | Menampilkan precision, recall, F1-score per kelas |
| `confusion_matrix` | Membuat tabel prediksi benar vs salah |
| `display` | Menampilkan tabel DataFrame dengan format HTML yang rapi di Jupyter |

### 🔑 Peran AdaBoost di sini:
Di tahap import, kita memuat **2 komponen inti AdaBoost**:

1. **`AdaBoostClassifier`** — Ini adalah algoritma utamanya. AdaBoost bekerja dengan prinsip **boosting**, yaitu menggabungkan banyak model lemah (weak learner) menjadi 1 model kuat (strong learner).

2. **`DecisionTreeClassifier`** — Ini adalah **weak learner** yang dipakai AdaBoost. Secara default, AdaBoost menggunakan **Decision Stump** (pohon keputusan dengan kedalaman hanya 1 level). Satu stump saja akurasinya rendah, tapi kalau digabungkan 100 stump oleh AdaBoost → jadi model yang kuat.

### 📊 Output:
Sel ini hanya menghasilkan teks:
```
Semua library berhasil diimport!
```
Artinya semua library sudah terinstall dan siap digunakan.

---

Bilang kalau mau lanjut ke **sel 2 (Membaca Data)** ya! 👍


# Penjelasan Sel 3.1: Label Encoding

```python
# Buat salinan dataframe agar data asli tidak berubah
df_encoded = df.copy()

# Label Encoding untuk Jenis Kelamin
le_jk = LabelEncoder()
df_encoded['Jenis Kelamin'] = le_jk.fit_transform(df_encoded['Jenis Kelamin'])

# Label Encoding untuk Status Gizi (target)
le_sg = LabelEncoder()
df_encoded['Status Gizi'] = le_sg.fit_transform(df_encoded['Status Gizi'])
```

### Penjelasan per baris:

| Kode | Penjelasan |
|------|------------|
| `df.copy()` | Membuat salinan data agar `df` asli tidak berubah (tetap bisa dipakai nanti) |
| `le_jk = LabelEncoder()` | Membuat objek encoder khusus untuk kolom **Jenis Kelamin** |
| `fit_transform(...)` | `fit` = belajar mapping (laki-laki→?, perempuan→?), lalu `transform` = terapkan mapping ke data |
| `le_sg = LabelEncoder()` | Membuat objek encoder khusus untuk kolom **Status Gizi** |

### 📊 Output yang keluar:

**Mapping Jenis Kelamin:**
```
  laki-laki            -> 0
  perempuan            -> 1
```
> LabelEncoder mengurutkan **alfabetis**: L sebelum P, jadi laki-laki = 0

**Mapping Status Gizi:**
```
  normal               -> 0
  severely stunted     -> 1
  stunted              -> 2
  tinggi               -> 3
```
> Juga diurutkan alfabetis: N < S (severely) < S (stunted) < T

**Tabel data setelah encoding:**

| | Umur (bulan) | Jenis Kelamin | Tinggi Badan (cm) | Status Gizi |
|---|---|---|---|---|
| 0 | 0 | 0 | 44.59 | 2 |
| 1 | 0 | 0 | 56.71 | 3 |
| 2 | 0 | 0 | 46.86 | 0 |
| 3 | 0 | 0 | 47.51 | 0 |
| 4 | 0 | 0 | 42.74 | 1 |

> Perhatikan: kolom **Jenis Kelamin** dan **Status Gizi** yang tadinya teks sekarang jadi angka.

### 🔑 Kenapa perlu Label Encoding?

Algoritma machine learning (termasuk **AdaBoost**) **tidak bisa memproses teks** — hanya bisa angka. Jadi:
- `"laki-laki"` → `0`, `"perempuan"` → `1`
- `"normal"` → `0`, `"severely stunted"` → `1`, `"stunted"` → `2`, `"tinggi"` → `3`

### 🔑 Peran untuk AdaBoost:
AdaBoost menggunakan **Decision Tree** sebagai base learner. Decision Tree membelah data berdasarkan **nilai angka** (misal: `Jenis Kelamin <= 0.5`). Tanpa encoding, tree tidak bisa membelah data teks. Jadi **Label Encoding wajib dilakukan sebelum training AdaBoost**.

---

Mau lanjut ke **sel 3.2 (Normalisasi)** atau langsung loncat ke sel lain? 👍


# Penjelasan Sel 3.2: Normalisasi (MinMaxScaler)

```python
# Pisahkan fitur (X) dan target (y)
X = df_encoded.drop('Status Gizi', axis=1)
y = df_encoded['Status Gizi']

# Normalisasi fitur menggunakan MinMaxScaler
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
```

### Penjelasan per baris:

| Kode | Penjelasan |
|------|------------|
| `X = df_encoded.drop(...)` | Memisahkan variabel independen (fitur/input). Kolom `Status Gizi` dibuang dari `X`. Jadi `X` hanya berisi Umur, Jenis Kelamin, dan Tinggi Badan. |
| `y = df_encoded['Status Gizi']` | Menentukan variabel dependen (target/output). `y` hanya berisi kolom `Status Gizi`. |
| `scaler = MinMaxScaler()` | Membuat objek untuk normalisasi. `MinMaxScaler` mengubah rentang nilai menjadi 0 hingga 1. |
| `scaler.fit_transform(X)` | Mencari nilai min & max dari setiap kolom (`fit`), lalu mengubah nilainya (`transform`) ke rentang 0-1. |
| `pd.DataFrame(...)` | Mengubah kembali hasil normalisasi (yang berupa array numpy) menjadi DataFrame pandas agar ada nama kolomnya. |

### 📊 Output yang keluar:

**Data fitur setelah Normalisasi MinMaxScaler (rentang 0-1):**

| | Umur (bulan) | Jenis Kelamin | Tinggi Badan (cm) |
|---|---|---|---|
| 0 | 0.0 | 0.0 | 0.0463 |
| 1 | 0.0 | 0.0 | 0.2016 |
| 2 | 0.0 | 0.0 | 0.0754 |
| 3 | 0.0 | 0.0 | 0.0837 |
| 4 | 0.0 | 0.0 | 0.0226 |

> Perhatikan nilainya:
> - **Umur**: Tadinya 0-60 bulan, sekarang diubah jadi rentang 0-1 (Umur 0 bulan = 0.0).
> - **Tinggi Badan**: Tadinya misal 42-120 cm, sekarang diubah proporsional ke 0-1 (44.59 cm jadi 0.0463).
> - **Jenis Kelamin**: Memang sudah 0 dan 1, jadi tetap 0.0 dan 1.0.

Maksudnya adalah tentang **cara kerja rumus MinMaxScaler**. 

MinMaxScaler "memampatkan" atau "memperkecil" skala data asli agar semuanya muat di dalam rentang angka **0,0 sampai 1,0**.

Rumus matematika di baliknya sangat sederhana:
**Nilai Baru = (Nilai Asli - Nilai Minimum) / (Nilai Maksimum - Nilai Minimum)**

Mari kita bedah satu per satu dengan contoh dari data kamu:

### 1. Contoh pada Umur (bulan)
Misalkan di dataset kamu:
- Umur bayi termuda (Min) = 0 bulan
- Umur anak tertua (Max) = 60 bulan

Jika ada bayi yang berumur **0 bulan** (seperti di baris pertama datamu):
* Nilai Baru = (0 - 0) / (60 - 0)
* Nilai Baru = 0 / 60 = **0.0**

*Bagaimana jika umurnya 30 bulan?*
* Nilai Baru = (30 - 0) / (60 - 0)
* Nilai Baru = 30 / 60 = **0.5** (Pas di tengah-tengah rentang 0-1)

### 2. Contoh pada Tinggi Badan (cm)
Misalkan di dataset kamu (berdasarkan hasil di layar):
- Tinggi badan terpendek (Min) = 40.96 cm
- Tinggi badan tertinggi (Max) = 118.88 cm

Jika ada bayi dengan tinggi badan **44.59 cm** (seperti di baris pertama datamu):
* Nilai Baru = (44.59 - 40.96) / (118.88 - 40.96)
* Nilai Baru = 3.63 / 77.92
* Nilai Baru = **0.0463**

### Intinya:
- Nilai yang **paling kecil** (minimum) di data asli akan selalu menjadi **0.0**.
- Nilai yang **paling besar** (maksimum) di data asli akan selalu menjadi **1.0**.
- Nilai-nilai di antaranya akan menjadi angka desimal (seperti 0.0463 atau 0.5) secara proporsional sesuai dengan nilainya.

**Kenapa harus repot-repot diubah seperti ini?**
Bayangkan mesin melihat selisih umur 2 bulan (misal 10 ke 12 bulan) dan selisih tinggi badan 2 cm (misal 50 ke 52 cm). Di dunia nyata, maknanya beda. Dengan normalisasi ke rentang 0-1, mesin melihat pergerakan atau perubahan nilai umur dan tinggi badan dalam "bahasa" atau rasio skala yang sama, sehingga tidak ada fitur yang mendominasi hanya karena angkanya lebih besar.

Apakah penjelasan ini cukup mudah dipahami?

### 🔑 Kenapa perlu Normalisasi?

Dalam dataset kita, rentang nilai tiap fitur sangat berbeda:
- **Umur**: 0 sampai 60
- **Jenis Kelamin**: 0 sampai 1
- **Tinggi Badan**: 40 sampai 128 (rentangnya sangat besar dibanding fitur lain).

Tanpa normalisasi, perhitungan matematika di beberapa algoritma akan lebih didominasi oleh `Tinggi Badan` karena nilainya paling besar. Normalisasi memastikan **semua fitur diperlakukan sama adilnya** dalam rentang 0-1.

### 🔑 Peran untuk AdaBoost:
Sebenarnya, **Decision Tree** (base learner dari AdaBoost) **tidak terlalu sensitif terhadap skala data**, jadi normalisasi tidak mutlak wajib untuk AdaBoost. Namun:
1. Normalisasi adalah **best practice** di pipeline Machine Learning.
2. Membantu **konvergensi dan komputasi** agar lebih stabil.
3. Karena kita menghitung **Korelasi Pearson** (yang berbasis varians/kovarians), skala yang diseragamkan membantu interpretasi data menjadi lebih baik.

---

Mau lanjut ke **sel 4 (Analisis Korelasi)**?


# Penjelasan Sel 4: Analisis Korelasi

```python
# Gabungkan fitur yang sudah dinormalisasi dengan target
df_corr = X_scaled.copy()
df_corr['Status Gizi'] = y.values

# Hitung korelasi Pearson
corr_matrix = df_corr.corr(method='pearson')

# Visualisasi Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.3f',
            linewidths=0.5, center=0, square=True,
            cbar_kws={'label': 'Korelasi Pearson'})
plt.title('Heatmap Korelasi Pearson', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Penjelasan per baris:

| Kode | Penjelasan |
|------|------------|
| `df_corr = X_scaled.copy()` | Membuat tabel baru khusus untuk mencari korelasi, isinya data fitur (X) yang sudah dinormalisasi. |
| `df_corr['Status Gizi'] = y.values` | Menambahkan kolom target (`Status Gizi`) ke dalam tabel baru tersebut, agar kita bisa melihat hubungan antara fitur dengan target. |
| **`df_corr.corr(method='pearson')`** | **Inti dari kode ini**. Menghitung nilai Korelasi Pearson antara semua kolom yang ada. |
| `sns.heatmap(...)` | Membuat visualisasi Heatmap (grafik kotak-kotak warna-warni) dari matriks korelasi tadi. `annot=True` memunculkan angkanya, `cmap='coolwarm'` warnanya biru (negatif) ke merah (positif). |

### 📊 Membaca Outputnya:

**Korelasi setiap fitur terhadap Status Gizi:**
* `Umur (bulan)` : **-0.1548**
* `Jenis Kelamin` : **+0.0061**
* `Tinggi Badan (cm)` : **-0.0495**

**Cara membaca nilai Korelasi Pearson:**
Nilai korelasi selalu berada di antara **-1.0** sampai **+1.0**.
* **Mendekati +1.0**: Korelasi positif kuat (Kalau A naik, B ikut naik).
* **Mendekati -1.0**: Korelasi negatif kuat (Kalau A naik, B malah turun).
* **Mendekati 0.0**: Tidak ada hubungan sama sekali.

**Interpretasi dari Data Balita:**
1. **Umur (-0.1548)**: Ada korelasi **negatif lemah**. Artinya, semakin bertambah umur, ada sedikit kecenderungan status gizinya ke arah angka yang lebih kecil (misal dari "tinggi" ke "normal" atau "stunted"). Tapi karena angkanya kecil (-0.15), hubungannya tidak terlalu kuat.
2. **Tinggi Badan (-0.0495)**: Sangat mendekati 0. Hubungan linier langsung antara nilai mentah tinggi badan dengan label status gizi sangat lemah. (Ini masuk akal karena status gizi balita biasanya ditentukan oleh **rasio** antara Tinggi Badan dan Umur, bukan cuma dari tinggi badannya saja).
3. **Jenis Kelamin (+0.0061)**: Hampir persis 0. Artinya, status gizi laki-laki dan perempuan di data ini relatif seimbang, tidak ada satu gender yang lebih mendominasi gizi buruk/baik.

### 🔑 Peran untuk AdaBoost:
1. Analisis korelasi membantu kita (manusia) paham mana fitur yang paling "berguna" sebelum model dilatih.
2. Karena nilai korelasi masing-masing fitur ke target **sangat kecil** (mendekati 0), ini membuktikan bahwa masalah klasifikasi status gizi balita itu **non-linear dan kompleks**.
3. **Di sinilah letak kehebatan AdaBoost!** Kalau masalahnya sederhana (korelasi besar), pakai model linier biasa sudah cukup. Tapi karena korelasinya kecil dan tersembunyi, kita butuh **AdaBoost** yang bisa menemukan pola rumit dari gabungan Umur, Jenis Kelamin, dan Tinggi Badan sekaligus secara bersamaan.

---

Bisa dipahami sejauh ini? Bilang ya kalau mau lanjut ke **Sel 5 (Split Data)**!



# Penjelasan Sel 5: Split Data (Train/Test)

```python
# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
```

### Penjelasan per bagian:

| Kode | Penjelasan |
|------|------------|
| `X_train`, `y_train` | Data fitur dan target yang akan digunakan untuk **mengajari/melatih** model (Training). |
| `X_test`, `y_test` | Data fitur dan target yang akan digunakan untuk **menguji** kepintaran model (Testing/Evaluasi). |
| `train_test_split(...)` | Fungsi bawaan sklearn untuk memecah data secara acak. |
| `test_size=0.2` | Menentukan bahwa **20%** data akan digunakan untuk testing, dan sisanya (80%) untuk training. |
| `random_state=42` | Menjaga agar acakan datanya selalu **sama setiap kali di-run**. (Kenapa angka 42? Ini cuma tradisi *programmer*, kamu bisa pakai angka berapa saja asal konsisten). |
| **`stratify=y`** | **Sangat Penting!** Ini memastikan proporsi kelas (jumlah anak gizi baik, gizi buruk, dsb) di data training dan testing **tetap seimbang**. |

### 📊 Output yang Keluar:

Outputnya menunjukkan angka detail pembagian datanya.
Karena total datamu ada 120.999 baris, maka:
* **Data Training (80%)** = 96.799 baris
* **Data Testing (20%)** = 24.200 baris

Di output juga terlihat **Distribusi target**. Berkat `stratify=y`, persentase jumlah balita stunting di data training akan sama persis dengan persentase di data testing. Ini mencegah model "kaget" karena tiba-tiba di data testing banyak kasus yang jarang dia pelajari di data training.

### 🔑 Analogi Sederhana untuk Dosen:
"Proses Split Data ini ibarat **sekolah**. 
* **Data Training (80%)** adalah buku cetak dan soal latihan yang dikerjakan murid saat belajar di kelas (untuk melatih model AdaBoost).
* **Data Testing (20%)** adalah Soal Ujian Akhir yang belum pernah dilihat murid sebelumnya (untuk menguji akurasi model)."

---
Mau lanjut ke **Sel 6 (Training Model AdaBoost)** yang merupakan inti dari tugasmu?



# Penjelasan Sel 6: Training Model AdaBoost (Inti Program)

Ini adalah bagian paling utama dari tugasmu, tempat di mana "otak" mesin mulai belajar!

```python
# Definisikan base estimator (Decision Stump)
base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)

# Buat model AdaBoost
model = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)

# Training model
model.fit(X_train, y_train)
```

### Penjelasan per baris:

| Kode | Penjelasan |
|------|------------|
| `base_estimator = DecisionTreeClassifier(max_depth=1)` | **Weak Learner (Murid Lemah)**. Kita membuat *Decision Tree* (pohon keputusan) dengan kedalaman maksimal hanya 1 (sering disebut *Decision Stump*). Model ini sangat sederhana dan akurasinya pasti jelek kalau dibiarkan sendirian. |
| `model = AdaBoostClassifier(...)` | **Strong Learner (Murid Pintar/Guru)**. Ini adalah algoritma utamanya. |
| `estimator=base_estimator` | Kita memberitahu AdaBoost: *"Tolong pakai si Decision Stump tadi sebagai dasar belajarmu ya!"* |
| `n_estimators=100` | Jumlah *Decision Stump* yang akan dibuat. Artinya AdaBoost akan membuat **100 model lemah**, di mana model ke-2 akan memperbaiki kesalahan model ke-1, model ke-3 memperbaiki kesalahan model ke-2, begitu seterusnya sampai 100 kali. |
| `learning_rate=1.0` | Seberapa besar bobot perbaikan yang dilakukan setiap kali ganti model (1.0 itu standar/normal). |
| **`model.fit(X_train, y_train)`** | **Proses Belajar (Training)**. Di sinilah AdaBoost memproses 96.799 data trainingmu untuk mencari pola. |

### 📊 Apa yang Terjadi di Balik Layar (Penjelasan untuk Dosen):

"Pak/Bu, AdaBoost yang kami bangun menggunakan prinsip **Boosting**. Awalnya, kami membuat satu *Decision Stump* (pohon dangkal) untuk memprediksi status gizi balita. Pasti banyak tebakannya yang salah."

"Kemudian, AdaBoost akan mencari tahu balita mana saja yang **salah ditebak** oleh pohon pertama. Balita yang salah tebak ini akan **diberi bobot lebih berat (difokuskan)**. Lalu, pohon kedua dibuat khusus untuk mencoba menjawab kasus yang susah/salah tadi. Proses ini diulang terus menerus sampai **100 kali** (`n_estimators=100`)."

"Hasil akhirnya, keputusan dari 100 pohon lemah itu **digabungkan (di-voting)**. Pohon yang akurasinya bagus punya hak suara lebih besar. Gabungan dari 100 pohon ini menghasilkan **Satu Model yang Sangat Akurat**."

---
Ini konsep yang sangat disukai dosen kalau kamu bisa menjelaskannya dengan lancar saat presentasi! 

Mau lanjut ke **Sel 7 (Evaluasi Model / Melihat Nilai Akurasi)**?
