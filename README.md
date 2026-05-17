# Penjelasan File `index.ipynb`

Dokumen ini menjelaskan isi notebook `index.ipynb` satu per satu. Notebook tersebut berisi implementasi algoritma AdaBoost untuk klasifikasi `Status Gizi` balita berdasarkan tiga fitur utama:

- `Umur (bulan)`
- `Jenis Kelamin`
- `Tinggi Badan (cm)`

Target atau label yang diprediksi adalah `Status Gizi`, yaitu kategori seperti `normal`, `stunted`, `severely stunted`, dan `tinggi`.

## Ringkasan Alur Notebook

Secara garis besar, notebook melakukan langkah berikut:

1. Mengimpor library Python yang dibutuhkan.
2. Membaca dataset `data_balita.csv`.
3. Mengecek struktur data, statistik, missing value, dan distribusi kelas.
4. Mengubah data kategorikal menjadi angka dengan `LabelEncoder`.
5. Memisahkan fitur `X` dan target `y`.
6. Membagi data menjadi data training dan testing.
7. Membuat model AdaBoost dengan weak learner berupa decision tree kecil.
8. Melatih model, melakukan prediksi, dan mengevaluasi performanya.
9. Membuat beberapa visualisasi seperti confusion matrix, feature importance, error weak learner, dan bobot weak learner.
10. Mencoba prediksi pada data baru.

## Cell 1 - Identitas Kelompok

Cell pertama berisi markdown daftar anggota kelompok.

Bagian ini tidak memengaruhi proses program. Isinya hanya informasi identitas kelompok untuk laporan tugas.

## Cell 2 - Judul Notebook

Bagian ini menjelaskan bahwa notebook membahas implementasi AdaBoost untuk klasifikasi status gizi balita.

Di sini sudah disebutkan bahwa:

- Target prediksi adalah `Status Gizi`.
- Fitur yang digunakan adalah umur, jenis kelamin, dan tinggi badan.
- Model yang dipakai adalah AdaBoost.

## Cell 3 - Penjelasan Singkat AdaBoost

Cell ini menjelaskan konsep dasar AdaBoost.

AdaBoost adalah algoritma ensemble learning. Maksudnya, model tidak hanya memakai satu model tunggal, tetapi menggabungkan banyak model kecil yang disebut weak learner.

Weak learner biasanya model sederhana yang performanya tidak terlalu kuat jika berdiri sendiri. Dalam notebook ini, weak learner yang digunakan adalah `DecisionTreeClassifier(max_depth=1)`.

`max_depth=1` berarti decision tree hanya boleh membuat satu keputusan utama. Model seperti ini sering disebut decision stump.

Cara kerja AdaBoost secara sederhana:

1. Membuat model kecil pertama.
2. Melihat data mana yang salah diprediksi.
3. Memberi perhatian lebih besar pada data yang salah.
4. Membuat model kecil berikutnya yang mencoba memperbaiki kesalahan sebelumnya.
5. Menggabungkan seluruh weak learner menjadi satu model akhir.

## Cell 4 - Penjelasan Import Library

Cell markdown ini menjelaskan library yang akan dipakai.

Library yang digunakan:

- `pandas`: membaca dan mengolah data tabel.
- `numpy`: operasi numerik.
- `matplotlib.pyplot`: membuat grafik.
- `seaborn`: membuat visualisasi data yang lebih mudah dibaca.
- `train_test_split`: membagi dataset menjadi data training dan testing.
- `LabelEncoder`: mengubah teks kategori menjadi angka.
- `DecisionTreeClassifier`: model decision tree yang dipakai sebagai weak learner.
- `plot_tree`: menampilkan bentuk pohon keputusan.
- `AdaBoostClassifier`: model utama AdaBoost.
- `accuracy_score`, `classification_report`, `confusion_matrix`: evaluasi model.

## Cell 5 - Import Library

Kode:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

Penjelasan:

Kode ini memanggil semua library yang diperlukan. Alias `pd`, `np`, `plt`, dan `sns` dipakai agar penulisan kode menjadi lebih singkat.

Contoh:

- `pd.read_csv()` dipakai untuk membaca file CSV.
- `plt.figure()` dipakai untuk membuat area grafik.
- `sns.countplot()` dipakai untuk membuat grafik jumlah data per kategori.

## Cell 6 - Penjelasan Membaca Dataset

Cell markdown ini menjelaskan bahwa dataset dibaca dari file `data_balita.csv`, lalu lima baris pertama ditampilkan.

Tujuannya adalah melihat bentuk awal dataset sebelum dilakukan preprocessing.

## Cell 7 - Membaca Dataset

Kode:

```python
df = pd.read_csv('data_balita.csv')
df.head()
```

Penjelasan:

`pd.read_csv('data_balita.csv')` membaca file CSV dan menyimpannya ke variabel `df`.

`df.head()` menampilkan lima baris pertama dataset.

Output awal menunjukkan kolom:

- `Umur (bulan)`
- `Jenis Kelamin`
- `Tinggi Badan (cm)`
- `Status Gizi`

Contoh isi data:

| Umur | Jenis Kelamin | Tinggi Badan | Status Gizi |
|---:|---|---:|---|
| 0 | laki-laki | 44.591973 | stunted |
| 0 | laki-laki | 56.705203 | tinggi |
| 0 | laki-laki | 46.863358 | normal |

## Cell 8 - Penjelasan Informasi Dataset

Bagian ini menjelaskan pentingnya memeriksa dataset sebelum membuat model.

Pemeriksaan dataset dilakukan untuk mengetahui:

- Jumlah baris dan kolom.
- Tipe data tiap kolom.
- Statistik data numerik.
- Apakah ada missing value.
- Jumlah data pada setiap kelas `Status Gizi`.

## Cell 9 - Informasi Dataset

Kode:

```python
print('Jumlah baris dan kolom:', df.shape)

print('\n===== INFO DATASET =====')
df.info()

print('\n===== DESKRIPSI DATA NUMERIK =====')
display(df.describe())

print('\n===== MISSING VALUE =====')
display(df.isnull().sum())

print('\n===== JUMLAH STATUS GIZI =====')
display(df['Status Gizi'].value_counts())
```

Penjelasan per baris:

- `df.shape` menampilkan ukuran dataset dalam format `(baris, kolom)`.
- `df.info()` menampilkan nama kolom, jumlah data non-null, dan tipe data.
- `df.describe()` menampilkan statistik untuk data numerik seperti rata-rata, nilai minimum, maksimum, dan kuartil.
- `df.isnull().sum()` menghitung jumlah missing value di setiap kolom.
- `df['Status Gizi'].value_counts()` menghitung jumlah data pada setiap kategori status gizi.

Hasil penting:

- Jumlah data: 120.999 baris dan 4 kolom.
- Tidak ada missing value.
- Kolom numerik: `Umur (bulan)` dan `Tinggi Badan (cm)`.
- Kolom kategorikal: `Jenis Kelamin` dan `Status Gizi`.

Distribusi target:

| Status Gizi | Jumlah |
|---|---:|
| normal | 67.755 |
| severely stunted | 19.869 |
| tinggi | 19.560 |
| stunted | 13.815 |

Catatan penting: kelas `normal` jauh lebih banyak daripada kelas lain. Ini bisa membuat model cenderung lebih sering memprediksi `normal`.

## Cell 10 - Penjelasan Grafik Distribusi Status Gizi

Cell ini menjelaskan grafik jumlah data untuk setiap kelas `Status Gizi`.

Grafik ini berguna untuk melihat apakah data seimbang atau tidak.

Jika satu kelas terlalu dominan, model bisa bias ke kelas tersebut. Pada dataset ini, kelas `normal` adalah kelas terbesar.

## Cell 11 - Grafik Distribusi Status Gizi

Kode:

```python
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Status Gizi', order=df['Status Gizi'].value_counts().index)
plt.title('Distribusi Kelas Status Gizi')
plt.xlabel('Status Gizi')
plt.ylabel('Jumlah Data')
plt.xticks(rotation=20)
plt.show()
```

Penjelasan:

- `plt.figure(figsize=(8, 5))` mengatur ukuran grafik.
- `sns.countplot(...)` membuat grafik batang jumlah data per kelas.
- `order=df['Status Gizi'].value_counts().index` mengurutkan batang dari kelas terbanyak ke paling sedikit.
- `plt.xticks(rotation=20)` memiringkan tulisan label agar lebih mudah dibaca.
- `plt.show()` menampilkan grafik.

## Cell 12 - Penjelasan Preprocessing Data

Cell markdown ini menjelaskan bahwa data kategorikal perlu diubah menjadi angka.

Model machine learning seperti AdaBoost dari scikit-learn tidak bisa langsung memproses teks seperti `laki-laki`, `perempuan`, `normal`, atau `stunted`.

Karena itu digunakan `LabelEncoder`.

## Cell 13 - Encoding Data Kategorikal

Kode:

```python
data = df.copy()

le_gender = LabelEncoder()
le_target = LabelEncoder()

data['Jenis Kelamin'] = le_gender.fit_transform(data['Jenis Kelamin'])
data['Status Gizi'] = le_target.fit_transform(data['Status Gizi'])
```

Penjelasan:

- `data = df.copy()` membuat salinan dataset agar data asli `df` tetap aman.
- `le_gender` dipakai untuk mengubah `Jenis Kelamin`.
- `le_target` dipakai untuk mengubah `Status Gizi`.
- `fit_transform()` mempelajari kategori yang ada, lalu mengubahnya menjadi kode angka.

Mapping hasil encoding:

`Jenis Kelamin`:

| Label | Kode |
|---|---:|
| laki-laki | 0 |
| perempuan | 1 |

`Status Gizi`:

| Label | Kode |
|---|---:|
| normal | 0 |
| severely stunted | 1 |
| stunted | 2 |
| tinggi | 3 |

Catatan: angka hasil encoding bukan berarti urutan kualitas. Misalnya `tinggi -> 3` bukan berarti lebih baik daripada `normal -> 0`. Angka hanya kode agar model bisa memproses data.

## Cell 14 - Penjelasan Fitur dan Target

Cell ini menjelaskan pemisahan antara fitur dan target.

Fitur adalah input model, sedangkan target adalah output yang ingin diprediksi.

Dalam notebook ini:

- `X`: fitur/input, berisi umur, jenis kelamin, dan tinggi badan.
- `y`: target/output, berisi status gizi yang sudah diencoding.

## Cell 15 - Menentukan Fitur dan Target

Kode:

```python
X = data[['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)']]
y = data['Status Gizi']
```

Penjelasan:

`X` berisi tiga kolom yang dipakai model untuk belajar pola:

- `Umur (bulan)`
- `Jenis Kelamin`
- `Tinggi Badan (cm)`

`y` berisi label yang ingin ditebak model, yaitu `Status Gizi`.

Output `X.head()` menampilkan lima baris pertama fitur. Output `y.head()` menampilkan lima target pertama dalam bentuk angka hasil encoding.

## Cell 16 - Penjelasan Pembagian Data

Cell ini menjelaskan bahwa dataset dibagi menjadi data training dan data testing.

Data training dipakai untuk melatih model. Data testing dipakai untuk menguji apakah model bisa memprediksi data yang belum pernah dilihat sebelumnya.

## Cell 17 - Membagi Data Training dan Testing

Kode:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Penjelasan parameter:

- `X` adalah fitur.
- `y` adalah target.
- `test_size=0.2` berarti 20 persen data dipakai untuk testing, 80 persen untuk training.
- `random_state=42` membuat hasil pembagian data tetap sama setiap kali kode dijalankan.
- `stratify=y` menjaga proporsi kelas target tetap mirip antara training dan testing.

Hasil pembagian:

- Data training: 96.799 baris.
- Data testing: 24.200 baris.

## Cell 18 - Penjelasan Model AdaBoost

Cell markdown ini menjelaskan pembuatan model AdaBoost.

Weak learner yang digunakan:

```python
DecisionTreeClassifier(max_depth=1)
```

Artinya, setiap weak learner hanya berupa pohon keputusan sangat sederhana.

AdaBoost kemudian membangun 50 weak learner dan menggabungkan hasil prediksinya.

## Cell 19 - Membuat dan Melatih Model AdaBoost

Kode:

```python
base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)

ada_model = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

ada_model.fit(X_train, y_train)
```

Penjelasan:

- `base_estimator` adalah model kecil yang menjadi weak learner.
- `max_depth=1` membuat decision tree hanya punya kedalaman 1.
- `AdaBoostClassifier(...)` membuat model AdaBoost.
- `estimator=base_estimator` berarti AdaBoost memakai decision tree kecil sebagai model dasar.
- `n_estimators=50` berarti AdaBoost membuat maksimal 50 weak learner.
- `learning_rate=1.0` mengatur besar kontribusi tiap weak learner.
- `ada_model.fit(X_train, y_train)` melatih model menggunakan data training.

Catatan kompatibilitas: pada versi scikit-learn lama, parameter `estimator` kadang masih bernama `base_estimator`.

## Cell 20 - Penjelasan Prediksi Data Testing

Cell ini menjelaskan bahwa model yang sudah dilatih digunakan untuk memprediksi `X_test`.

Hasil prediksi kemudian dibandingkan dengan nilai asli `y_test`.

## Cell 21 - Prediksi Data Testing

Kode:

```python
y_pred = ada_model.predict(X_test)

hasil_prediksi = pd.DataFrame({
    'Aktual': le_target.inverse_transform(y_test),
    'Prediksi': le_target.inverse_transform(y_pred)
})

hasil_prediksi.head(10)
```

Penjelasan:

- `ada_model.predict(X_test)` menghasilkan prediksi untuk data testing.
- `y_pred` berisi hasil prediksi dalam bentuk angka.
- `inverse_transform()` mengubah kode angka kembali menjadi label teks, misalnya `0` menjadi `normal`.
- `pd.DataFrame(...)` membuat tabel perbandingan antara nilai aktual dan prediksi.
- `head(10)` menampilkan 10 hasil prediksi pertama.

Dari output terlihat ada prediksi yang benar dan ada yang salah. Misalnya data aktual `severely stunted` beberapa kali diprediksi sebagai `normal`.

## Cell 22 - Penjelasan Evaluasi Model

Cell ini menjelaskan metrik evaluasi:

- `accuracy`: persentase prediksi yang benar.
- `precision`: dari semua data yang diprediksi sebagai kelas tertentu, berapa yang benar.
- `recall`: dari semua data aktual pada kelas tertentu, berapa yang berhasil ditemukan model.
- `f1-score`: rata-rata harmonis antara precision dan recall.
- `support`: jumlah data asli pada kelas tersebut.

## Cell 23 - Evaluasi Model AdaBoost

Kode:

```python
akurasi = accuracy_score(y_test, y_pred)

print(f'Akurasi AdaBoost: {akurasi:.4f}')

print('\n===== CLASSIFICATION REPORT =====')
print(classification_report(y_test, y_pred, target_names=le_target.classes_))
```

Penjelasan:

- `accuracy_score(y_test, y_pred)` menghitung akurasi.
- `classification_report(...)` menampilkan precision, recall, f1-score, dan support untuk tiap kelas.

Hasil akurasi:

```text
Akurasi AdaBoost: 0.5606
```

Artinya model benar sekitar 56,06 persen pada data testing.

Ringkasan classification report:

| Kelas | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| normal | 0.59 | 0.92 | 0.72 | 13.551 |
| severely stunted | 0.31 | 0.20 | 0.24 | 3.974 |
| stunted | 0.00 | 0.00 | 0.00 | 2.763 |
| tinggi | 0.90 | 0.09 | 0.16 | 3.912 |

Interpretasi:

- Model sangat sering mengenali kelas `normal`, terbukti recall `normal` mencapai 0,92.
- Model gagal mengenali kelas `stunted`, karena precision, recall, dan f1-score semuanya 0.
- Kelas `tinggi` memiliki precision tinggi 0,90, tetapi recall hanya 0,09. Artinya saat model memprediksi `tinggi`, sering benar, tetapi model jarang sekali memilih kelas `tinggi`.
- Akurasi 56,06 persen belum terlalu baik, terutama karena performa antar kelas tidak seimbang.

## Cell 24 - Penjelasan Confusion Matrix

Cell ini menjelaskan confusion matrix.

Confusion matrix adalah tabel yang membandingkan label aktual dengan label prediksi.

Nilai diagonal menunjukkan prediksi benar. Nilai di luar diagonal menunjukkan prediksi salah.

## Cell 25 - Grafik Confusion Matrix

Kode:

```python
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=le_target.classes_,
    yticklabels=le_target.classes_
)
plt.title('Confusion Matrix AdaBoost')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.show()
```

Penjelasan:

- `confusion_matrix(y_test, y_pred)` membuat matriks perbandingan aktual vs prediksi.
- `sns.heatmap(...)` menampilkan matriks dalam bentuk peta warna.
- `annot=True` menampilkan angka di dalam kotak.
- `fmt='d'` memastikan angka ditampilkan sebagai bilangan bulat.
- `xticklabels` adalah label kelas pada sumbu prediksi.
- `yticklabels` adalah label kelas pada sumbu aktual.

Grafik ini membantu melihat kesalahan terbesar model. Dari classification report, kemungkinan besar banyak kelas selain `normal` yang masuk ke prediksi `normal`.

## Cell 26 - Penjelasan Visualisasi Weak Learner

Cell ini menjelaskan bahwa AdaBoost terdiri dari banyak weak learner.

Grafik yang ditampilkan hanya weak learner pertama:

```python
ada_model.estimators_[0]
```

Jadi grafik ini bukan keseluruhan model AdaBoost, melainkan satu decision tree kecil yang menjadi bagian dari AdaBoost.

## Cell 27 - Visualisasi Weak Learner Pertama

Kode:

```python
plt.figure(figsize=(10, 5))
plot_tree(
    ada_model.estimators_[0],
    feature_names=X.columns,
    class_names=le_target.classes_,
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title('Visualisasi Weak Learner Pertama pada AdaBoost')
plt.show()
```

Penjelasan:

- `ada_model.estimators_[0]` mengambil weak learner pertama.
- `plot_tree(...)` menggambar decision tree.
- `feature_names=X.columns` menampilkan nama fitur pada node pohon.
- `class_names=le_target.classes_` menampilkan nama kelas target.
- `filled=True` memberi warna pada node.
- `rounded=True` membuat bentuk node lebih rapi.

Karena `max_depth=1`, pohonnya hanya membuat satu percabangan keputusan.

## Cell 28 - Penjelasan Feature Importance

Cell ini menjelaskan bahwa feature importance menunjukkan seberapa besar kontribusi fitur terhadap model.

Nilai importance makin besar berarti fitur makin sering atau makin kuat dipakai dalam proses prediksi.

## Cell 29 - Menghitung Feature Importance

Kode:

```python
feature_importance = pd.DataFrame({
    'Fitur': X.columns,
    'Importance': ada_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

display(feature_importance)
```

Penjelasan:

- `ada_model.feature_importances_` mengambil nilai kepentingan tiap fitur.
- `pd.DataFrame(...)` membuat tabel importance.
- `sort_values(..., ascending=False)` mengurutkan dari importance terbesar ke terkecil.

Hasil:

| Fitur | Importance |
|---|---:|
| Tinggi Badan (cm) | 0.683879 |
| Umur (bulan) | 0.316121 |
| Jenis Kelamin | 0.000000 |

Interpretasi:

- `Tinggi Badan (cm)` adalah fitur paling berpengaruh.
- `Umur (bulan)` juga berpengaruh, tetapi lebih kecil.
- `Jenis Kelamin` tidak dipakai secara berarti oleh model pada hasil ini.

## Cell 30 - Penjelasan Grafik Feature Importance

Cell markdown ini menjelaskan bahwa grafik feature importance dibuat agar kontribusi fitur lebih mudah dilihat.

## Cell 31 - Grafik Feature Importance

Kode:

```python
plt.figure(figsize=(8, 5))
sns.barplot(data=feature_importance, x='Importance', y='Fitur')
plt.title('Feature Importance AdaBoost')
plt.xlabel('Nilai Importance')
plt.ylabel('Fitur')
plt.show()
```

Penjelasan:

- `sns.barplot(...)` membuat grafik batang horizontal.
- Sumbu `x` menunjukkan nilai importance.
- Sumbu `y` menunjukkan nama fitur.

Grafik ini memperjelas bahwa tinggi badan adalah fitur yang paling dominan.

## Cell 32 - Penjelasan Error Setiap Weak Learner

Cell ini menjelaskan grafik error tiap weak learner.

Dalam AdaBoost, setiap weak learner punya nilai error. Error ini dipakai untuk menentukan seberapa besar pengaruh weak learner tersebut dalam model akhir.

## Cell 33 - Grafik Error Setiap Weak Learner

Kode:

```python
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(ada_model.estimator_errors_) + 1), ada_model.estimator_errors_, marker='o')
plt.title('Error Setiap Weak Learner pada AdaBoost')
plt.xlabel('Weak Learner ke-')
plt.ylabel('Error')
plt.grid(True)
plt.show()
```

Penjelasan:

- `ada_model.estimator_errors_` berisi daftar error dari setiap weak learner.
- `range(1, len(...) + 1)` membuat nomor weak learner mulai dari 1.
- `plt.plot(...)` menampilkan error dalam bentuk garis.
- `marker='o'` memberi tanda titik pada setiap weak learner.
- `plt.grid(True)` menampilkan garis bantu.

Jika error weak learner kecil, weak learner tersebut lebih baik dalam memisahkan data.

## Cell 34 - Penjelasan Bobot Setiap Weak Learner

Cell ini menjelaskan bobot weak learner.

Dalam AdaBoost, weak learner dengan error lebih kecil biasanya diberi bobot lebih besar, sehingga suaranya lebih berpengaruh dalam prediksi akhir.

## Cell 35 - Grafik Bobot Setiap Weak Learner

Kode:

```python
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(ada_model.estimator_weights_) + 1), ada_model.estimator_weights_, marker='o')
plt.title('Bobot Setiap Weak Learner pada AdaBoost')
plt.xlabel('Weak Learner ke-')
plt.ylabel('Bobot')
plt.grid(True)
plt.show()
```

Penjelasan:

- `ada_model.estimator_weights_` berisi bobot setiap weak learner.
- Bobot menunjukkan seberapa besar pengaruh weak learner dalam voting akhir.
- Grafik ini bisa dibandingkan dengan grafik error. Umumnya, error lebih kecil menghasilkan bobot lebih besar.

## Cell 36 - Penjelasan Prediksi Data Baru

Cell ini menjelaskan cara menggunakan model untuk data baru.

Hal penting: data baru harus memiliki format yang sama dengan data training.

Karena `Jenis Kelamin` saat training sudah diencoding, data baru juga harus diencoding memakai encoder yang sama, yaitu `le_gender`.

## Cell 37 - Contoh Prediksi Data Baru

Kode:

```python
data_baru = pd.DataFrame({
    'Umur (bulan)': [1],
    'Jenis Kelamin': le_gender.transform(['perempuan']),
    'Tinggi Badan (cm)': [1.0]
})

prediksi_baru = ada_model.predict(data_baru)
hasil_prediksi_baru = le_target.inverse_transform(prediksi_baru)

print('Data baru:')
display(data_baru)

print('Prediksi Status Gizi:', hasil_prediksi_baru[0])
```

Penjelasan:

- `data_baru` membuat satu data balita baru.
- Umur balita adalah 1 bulan.
- Jenis kelamin adalah `perempuan`, lalu diubah menjadi kode angka memakai `le_gender.transform(...)`.
- Tinggi badan adalah 1.0 cm.
- `ada_model.predict(data_baru)` memprediksi status gizi.
- `le_target.inverse_transform(...)` mengubah hasil prediksi angka menjadi teks.

Output:

```text
Prediksi Status Gizi: severely stunted
```

Catatan penting: nilai `Tinggi Badan (cm) = 1.0` tidak realistis untuk balita. Karena sangat jauh dari rentang data asli, wajar jika model memprediksi `severely stunted`. Contoh yang lebih realistis sebaiknya memakai tinggi badan dalam rentang sekitar 40 sampai 128 cm, sesuai dataset.

## Cell 38 - Perbedaan AdaBoost dan XGBoost

Cell markdown ini menjelaskan perbedaan AdaBoost dan XGBoost.

Ringkasnya:

| Aspek | AdaBoost | XGBoost |
|---|---|---|
| Nama lengkap | Adaptive Boosting | Extreme Gradient Boosting |
| Cara memperbaiki model | Fokus pada data yang salah diprediksi | Mengoptimasi loss function dengan gradient |
| Kompleksitas | Lebih sederhana | Lebih kompleks |
| Kecepatan dan fitur | Standar | Lebih cepat dan banyak optimasi |
| Regularisasi | Terbatas | Ada regularisasi untuk mengurangi overfitting |

AdaBoost cocok untuk memahami konsep boosting dasar. XGBoost biasanya lebih kuat untuk kompetisi atau kasus industri, tetapi pengaturannya lebih kompleks.

## Cell 39 - Kesimpulan

Cell terakhir menyimpulkan bahwa AdaBoost dapat digunakan untuk klasifikasi status gizi balita berdasarkan umur, jenis kelamin, dan tinggi badan.

Kesimpulan teknis dari hasil notebook:

- Dataset berhasil dibaca dan tidak memiliki missing value.
- Data kategorikal berhasil diubah menjadi angka.
- Model AdaBoost berhasil dilatih dengan 50 weak learner.
- Akurasi model adalah sekitar 56,06 persen.
- Model paling banyak bergantung pada fitur `Tinggi Badan (cm)`.
- Performa model belum merata untuk semua kelas, terutama kelas `stunted`.

## Catatan Evaluasi untuk Pemahaman

Walaupun notebook sudah berjalan, ada beberapa hal yang perlu dipahami:

1. Akurasi 56,06 persen belum bisa dianggap sangat baik.
2. Dataset tidak seimbang karena kelas `normal` jauh lebih banyak.
3. Model terlihat bias ke kelas `normal`.
4. Kelas `stunted` tidak berhasil dikenali model pada hasil evaluasi.
5. Perlu mencoba perbaikan seperti balancing data, tuning parameter, atau model lain.

## Ide Perbaikan Model

Beberapa hal yang bisa dicoba untuk meningkatkan hasil:

- Menggunakan `class_weight` pada decision tree jika cocok dengan pendekatan model.
- Melakukan oversampling kelas minoritas atau undersampling kelas mayoritas.
- Mencoba parameter `n_estimators` lain, misalnya 100 atau 200.
- Mencoba `learning_rate` lain, misalnya 0.1, 0.5, atau 1.5.
- Membandingkan dengan model lain seperti Random Forest, Gradient Boosting, atau XGBoost.
- Mengecek apakah fitur lain dibutuhkan, misalnya berat badan, jika tersedia.

## Penjelasan Singkat Istilah Penting

`DataFrame`: struktur data tabel dari pandas.

`Fitur`: kolom input yang digunakan untuk prediksi.

`Target`: kolom output yang ingin diprediksi.

`Training data`: data untuk melatih model.

`Testing data`: data untuk menguji model.

`LabelEncoder`: alat untuk mengubah label teks menjadi angka.

`Weak learner`: model sederhana yang menjadi penyusun model ensemble.

`AdaBoost`: algoritma boosting yang menggabungkan banyak weak learner.

`Confusion matrix`: tabel perbandingan prediksi dan data asli.

`Feature importance`: nilai kontribusi fitur dalam model.

`Precision`: ukuran ketepatan prediksi pada suatu kelas.

`Recall`: ukuran kemampuan model menemukan seluruh data pada suatu kelas.

`F1-score`: gabungan precision dan recall.

