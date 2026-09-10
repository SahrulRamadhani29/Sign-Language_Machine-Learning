# LatihIsyarat: Konsep PBL dan Spesifikasi untuk AI Agent

Versi: 1.0 | Tanggal: 10 September 2026

Dokumen ini menjadi spesifikasi proyek sekaligus instruksi kerja untuk AI agent. Keputusan yang sudah disepakati harus dipertahankan. Nilai performa, ukuran model, ambang prediksi, dan pilihan paket yang belum diuji ditandai sebagai target atau konfigurasi awal, bukan hasil yang sudah tercapai.

## 1. Ringkasan proyek

Bangun **LatihIsyarat**, aplikasi Android berbasis Flutter untuk belajar dan berlatih 24 huruf statis alfabet American Sign Language (ASL). Pengguna membuka kamera, memosisikan satu tangan dalam kotak panduan, dan melihat prediksi huruf secara langsung. Pengguna tidak perlu memotret atau mengirim foto satu per satu secara manual.

Sistem mengambil beberapa frame kamera secara otomatis, memproses area tangan, lalu menjalankan CNN secara lokal di HP. Preview kamera tetap berjalan selama latihan. Hasil hanya ditampilkan sebagai huruf yang dikenali setelah cukup stabil.

CNN dirancang dan dilatih sendiri dari bobot acak menggunakan TensorFlow/Keras di Google Colab. Dataset utama adalah Sign Language MNIST. Model terbaik diekspor ke TensorFlow Lite/LiteRT (`.tflite`) dan dibundel dalam aplikasi.

Versi utama bekerja offline. Tidak memerlukan VPS, web hosting, REST API prediksi, akun pengguna, langganan cloud, atau Hugging Face untuk menjalankan prediksi. Hugging Face hanya menjadi pilihan di masa depan jika kelompok ingin membagikan model atau membuat demo terpisah.

## 2. Identitas dan tujuan akademik

**Nama aplikasi:** LatihIsyarat.

**Judul PBL:** Pengembangan Aplikasi Mobile Pembelajaran 24 Huruf Alfabet ASL dengan Pengenalan Kamera Langsung Menggunakan CNN yang Dilatih dari Nol.

**Sasaran pengguna:** pemula yang ingin mempelajari ejaan jari ASL, termasuk mahasiswa. Aplikasi bukan alat penerjemahan percakapan untuk komunitas Tuli Indonesia.

**Masalah yang diselesaikan:** pengguna dapat membaca contoh alfabet tangan, tetapi membutuhkan latihan interaktif untuk membandingkan bentuk tangan dengan contoh dan melihat perkiraan huruf yang dikenali sistem.

**Tujuan PBL:**

- Menyiapkan dan memahami dataset klasifikasi citra multikelas.
- Merancang CNN sederhana dan melatihnya tanpa bobot pretrained.
- Mengevaluasi model pada data terpisah dan pada kamera HP nyata.
- Mengintegrasikan model lokal ke Flutter.
- Membuat pengalaman belajar yang lengkap: materi, latihan kamera, kuis, dan progres lokal.
- Mengukur waktu prediksi, penggunaan sumber daya, serta batas kemampuan aplikasi.

## 3. Keputusan yang sudah ditetapkan

| Aspek | Keputusan |
|---|---|
| Platform awal | Android, aplikasi Flutter/Dart |
| Bentuk penggunaan | Kamera langsung, sampling frame otomatis |
| Bahasa isyarat | Alfabet ASL, bukan BISINDO atau SIBI |
| Kelas model | 24 huruf statis, tanpa J dan Z |
| Training | TensorFlow/Keras di Google Colab |
| Inisialisasi model | Bobot acak; tidak menggunakan classifier pretrained |
| Dataset utama | Sign Language MNIST dari datamunge di Kaggle |
| Lokasi inferensi | HP pengguna, secara offline |
| Format model mobile | `.tflite` |
| Penyimpanan progres | Lokal di HP |
| Backend/API | Tidak diperlukan untuk versi utama |
| Distribusi PBL | APK untuk pemasangan langsung pada perangkat uji |
| Biaya layanan | Dirancang tanpa layanan berbayar wajib; perangkat, internet, dan listrik tetap diperlukan |

Google Play tidak termasuk distribusi wajib. Publikasi melalui toko aplikasi dapat memiliki persyaratan dan biaya akun tersendiri. GPU Colab gratis juga tidak dijamin tersedia atau tanpa batas waktu.

## 4. Batasan bahasa dan kemampuan model

ASL dan BISINDO merupakan bahasa berbeda. Jangan mengganti nama materi ASL menjadi BISINDO hanya karena antarmuka menggunakan bahasa Indonesia.

Dataset mendukung:

`A B C D E F G H I K L M N O P Q R S T U V W X Y`

Huruf J dan Z dalam ejaan jari ASL melibatkan gerakan dan tidak ada di dataset. Aplikasi boleh menampilkan keduanya dalam daftar alfabet dengan penjelasan "Belum didukung: membutuhkan pengenalan gerakan", tetapi tidak boleh menilai keduanya dengan model 24 kelas.

Pengenalan dari video kamera pada proyek ini adalah **klasifikasi frame statis yang diulang**, bukan pemahaman urutan gerakan. CNN tidak menerjemahkan kalimat, ekspresi wajah, tata bahasa ASL, atau percakapan.

Gambar 28 x 28 dari dataset berbeda dari kondisi kamera nyata. Performa live harus diuji; akurasi tinggi pada dataset tidak cukup untuk mengklaim aplikasi andal di lapangan.

## 5. Fitur wajib versi pertama

### 5.1 Panduan awal

- Jelaskan bahwa aplikasi merupakan latihan alfabet ASL.
- Jelaskan dukungan 24 huruf dan batasan J/Z.
- Jelaskan prediksi dilakukan di HP dan gambar kamera tidak diunggah oleh aplikasi.
- Minta izin kamera ketika fitur kamera pertama kali digunakan, bukan langsung saat aplikasi dibuka.
- Berikan panduan pencahayaan, jarak, posisi tangan, dan latar sederhana.

### 5.2 Beranda

- Akses Belajar Alfabet, Latihan Kamera, Kuis, dan Progres.
- Ringkasan latihan terakhir dan jumlah sesi lokal.
- Tidak membutuhkan login.

### 5.3 Materi alfabet

- Grid 24 huruf yang didukung.
- Detail huruf berisi foto/ilustrasi referensi yang benar dan petunjuk singkat.
- Sumber visual dan izin penggunaannya dicatat.
- Validasi materi terhadap sumber ASL yang layak atau penutur/pengajar ASL yang bersedia membantu.
- Jangan membuat foto tangan secara generatif lalu menganggapnya benar tanpa validasi.
- Dataset 28 x 28 bukan satu-satunya referensi visual pengajaran; resolusinya terlalu kecil untuk menjelaskan posisi jari secara rinci.

### 5.4 Latihan kamera langsung

- Preview kamera dengan kotak panduan persegi untuk satu tangan.
- Tombol mulai/jeda, pemilihan kamera jika tersedia, dan akses panduan.
- Pemrosesan frame otomatis; tidak memerlukan tombol foto.
- Tampilkan status seperti "Posisikan tangan", "Sedang membaca", atau "Belum terbaca".
- Tampilkan huruf setelah prediksi melewati kebijakan kestabilan yang diuji.
- Jangan mengganti huruf di layar setiap frame jika hasil masih berfluktuasi.
- Tampilkan penjelasan singkat ketika fitur kamera gagal atau izin ditolak.
- Batasi satu proses inferensi aktif; frame baru boleh dilewati saat proses sebelumnya belum selesai.
- Hentikan kamera dan pemrosesan saat pengguna meninggalkan halaman atau aplikasi berada di latar belakang.

### 5.5 Kuis praktik

- Pilih target dari 24 huruf yang didukung.
- Tampilkan target huruf dan minta pengguna membentuk isyarat dalam kotak kamera.
- Nilai jawaban berdasarkan hasil yang stabil, bukan satu frame.
- Kegagalan model membaca tidak otomatis dihitung sebagai jawaban pengguna yang salah.
- Sediakan coba lagi dan lewati; tidak perlu batas waktu ketat pada versi awal.
- Simpan ringkasan sesi secara lokal: target, hasil, jumlah percobaan, dan waktu sesi.
- Jika deteksi live belum cukup andal, tampilkan kuis pengenalan contoh sebagai fitur dasar dan nyatakan status kuis kamera secara jujur.

### 5.6 Progres lokal

- Riwayat sesi, jumlah latihan, hasil kuis, dan huruf yang sering memerlukan percobaan ulang.
- Progres adalah catatan penggunaan aplikasi, bukan sertifikasi kemampuan ASL.
- Sediakan hapus progres dengan konfirmasi.
- Tidak perlu menyimpan frame kamera atau video untuk fitur ini.

### 5.7 Tentang aplikasi

- Jelaskan identitas PBL, versi aplikasi, versi model, dan referensi dataset.
- Jelaskan keterbatasan model dan bahwa sistem belum mendukung penerjemahan percakapan.
- Jangan menampilkan klaim akurasi sebelum evaluasi nyata tersedia.

## 6. Alur pengguna utama

1. Pengguna memasang APK dan membuka aplikasi.
2. Pengguna membaca panduan ringkas.
3. Pengguna memilih huruf dari materi atau membuka latihan bebas.
4. Pengguna memberikan izin kamera ketika diminta.
5. Preview muncul bersama kotak panduan.
6. Pengguna menempatkan satu tangan dalam kotak dan menahan bentuk huruf.
7. Aplikasi mengambil beberapa frame per detik secara otomatis.
8. Model lokal melakukan prediksi; sistem menggabungkan hasil beberapa frame.
9. Prediksi yang stabil muncul di layar atau sistem meminta pengguna memperbaiki posisi.
10. Pengguna melanjutkan kuis atau melihat progres.

## 7. Arsitektur sistem

### Saat pengembangan

`Sign Language MNIST -> notebook Colab -> CNN Keras -> evaluasi -> ekspor TFLite -> aplikasi Flutter`

### Saat digunakan

`Kamera HP -> frame terpilih -> koreksi orientasi -> crop ROI -> grayscale -> resize -> CNN TFLite -> stabilisasi -> UI`

### Tanggung jawab komponen

| Komponen | Tanggung jawab |
|---|---|
| Camera service | Mengelola izin, kamera, stream, orientasi, dan lifecycle |
| Preprocessing service | Konversi piksel kamera menjadi input yang identik dengan training |
| Classifier service | Memuat interpreter, tensor, label, dan menjalankan model |
| Stabilizer | Menentukan kapan prediksi cukup konsisten untuk ditampilkan |
| Learning/quiz controller | Mengelola sesi, target, percobaan, dan progres |
| Local repository | Menyimpan progres dan pengaturan |
| UI | Materi, preview, status, prediksi, kuis, dan progres |

Tidak ada transfer dataset ke HP. APK hanya membawa model yang sudah dilatih, metadata, materi, dan kode aplikasi. Model dimuat sekali untuk sesi, bukan dibuat ulang pada setiap frame.

## 8. Teknologi dan struktur proyek

Gunakan Flutter/Dart untuk mobile, Python dengan TensorFlow/Keras untuk training, dan runtime TFLite/LiteRT yang kompatibel dengan Flutter/Android untuk inferensi. Pilih paket aktif setelah memeriksa dokumentasi, dukungan platform, lisensi, dan kebutuhan SDK-nya. Kunci versi yang telah diuji; jangan menebak API paket.

Untuk state management, gunakan satu pendekatan yang sederhana dan konsisten, misalnya Riverpod. Penyimpanan dapat memakai SQLite untuk riwayat sesi dan penyimpanan preferensi sederhana untuk pengaturan. Jangan menambah arsitektur yang tidak membantu kebutuhan PBL.

Struktur yang disarankan:

```text
latihisyarat/
  README.md
  mobile/
    lib/
      app/
      features/onboarding/
      features/alphabet/
      features/live_practice/
      features/quiz/
      features/progress/
      services/camera/
      services/inference/
      services/storage/
    assets/models/
    assets/alphabet/
    test/
    integration_test/
  ml/
    notebooks/
    src/
    configs/
    reports/
    exports/
  docs/
    concept.md
    dataset_notes.md
    model_card.md
    camera_pipeline.md
    evaluation.md
```

Jangan memasukkan dataset besar, token, cache, atau hasil training yang tidak diperlukan ke repositori kode. Jika menyimpan model dalam repositori, catat versi dan hash-nya.

## 9. Dataset dan pembagian data

Sumber utama: https://www.kaggle.com/datasets/datamunge/sign-language-mnist

Menurut deskripsi dataset:

- Training bawaan: 27.455 contoh.
- Testing bawaan: 7.172 contoh.
- Total: 34.627 contoh.
- Bentuk: CSV dengan kolom label dan 784 piksel grayscale bernilai 0-255.
- Resolusi: 28 x 28.
- Kelas: 24 huruf dengan label asli dari rentang 0-25, tanpa label 9 (J) dan 25 (Z).
- Dataset mencantumkan CC0 pada halaman Kaggle; tetap cantumkan atribusi akademik dan telusuri referensi pembentuk dataset.
- Banyak contoh berasal dari augmentasi kumpulan sumber yang relatif kecil. Jangan menyamakan jumlah baris dengan jumlah pengguna independen.

Gunakan semua contoh sesuai peran, bukan seluruhnya sebagai data training:

| Bagian | Jumlah | Penggunaan |
|---|---:|---|
| Training | 21.964 | Memperbarui bobot model |
| Validasi | 5.491 | Memilih konfigurasi dan checkpoint |
| Testing bawaan | 7.172 | Evaluasi akhir setelah model dipilih |

Ambil validasi sebesar 20% dari training bawaan dengan stratifikasi kelas dan seed tetap. Jika versi dataset berbeda, periksa jumlah sebenarnya dan dokumentasikan perubahannya.

**Pemetaan label wajib eksplisit:**

```python
CLASS_NAMES = list("ABCDEFGHIKLMNOPQRSTUVWXY")
ORIGINAL_LABELS = [i for i in range(26) if i not in (9, 25)]
LABEL_TO_INDEX = {label: index for index, label in enumerate(ORIGINAL_LABELS)}
```

Output model berindeks 0-23 dan dipetakan melalui `CLASS_NAMES`. Jangan langsung mengubah indeks output menjadi `chr(65 + index)` karena hasil setelah I akan bergeser.

Periksa nilai piksel, jumlah kelas, kelas kosong, duplikat identik, dan kemiripan antarsplit. Jika identitas gambar sumber tersedia, kelompokkan seluruh turunannya dalam split yang sama. Jika tidak tersedia, laporkan keterbatasan evaluasi berbasis augmentasi dan jangan mengklaim pemisahan berdasarkan individu. Jangan menggunakan testing untuk memilih augmentasi atau arsitektur.

## 10. Rancangan model awal

Gunakan CNN sederhana berikut sebagai baseline eksperimen, bukan arsitektur final yang sudah terbukti:

```text
Input 28 x 28 x 1, float32, nilai 0-255
Rescaling 1/255 di dalam model
Conv2D 32, kernel 3 x 3, padding same, ReLU
MaxPooling2D 2 x 2
Conv2D 64, kernel 3 x 3, padding same, ReLU
MaxPooling2D 2 x 2
Conv2D 128, kernel 3 x 3, padding same, ReLU
MaxPooling2D 2 x 2
GlobalAveragePooling2D
Dense 64, ReLU
Dropout 0.3
Dense 24, Softmax
```

Konfigurasi awal:

- Optimizer Adam dengan learning rate awal 0.001.
- Loss sparse categorical crossentropy untuk indeks kelas integer.
- Batch awal 64; sesuaikan berdasarkan memori dan hasil eksperimen.
- Maksimal awal 30 epoch dengan EarlyStopping berdasarkan validation loss.
- ModelCheckpoint menyimpan checkpoint dengan validation loss terbaik.
- ReduceLROnPlateau dapat membantu ketika validasi berhenti membaik.
- Tetapkan seed dan catat versi library serta konfigurasi eksperimen.
- Mulai tanpa augmentasi tambahan, lalu uji perubahan kecil berdasarkan validasi.
- Jangan menambahkan horizontal flip, rotasi besar, atau transformasi bentuk tangan tanpa memeriksa apakah label dan makna tetap valid.

Tidak menggunakan MobileNet, EfficientNet, ResNet, atau classifier pretrained. TensorFlow/Keras tetap boleh digunakan untuk operasi dan training; "dari nol" berarti bobot classifier dilatih sendiri, bukan membuat framework deep learning sendiri.

Jangan menyebut rancangan ini sebagai arsitektur ilmiah baru tanpa penelitian pembanding. Kontribusi PBL berada pada eksperimen, integrasi, pengalaman aplikasi, dan evaluasi.

## 11. Penyimpanan dan ekspor model

Simpan checkpoint terbaik ke penyimpanan permanen selama training, misalnya Google Drive pengguna. File sementara Colab dapat hilang ketika runtime berakhir.

Hasil yang diperlukan:

```text
best_model.keras
model_float32.tflite
labels.json
model_metadata.json
training_history.csv
evaluation.json
confusion_matrix.png
model_card.md
```

Mulai dengan TFLite float32. Setelah hasilnya cocok dengan Keras, pertimbangkan float16 atau int8 jika ada manfaat nyata pada ukuran atau performa perangkat. Int8 membutuhkan representative dataset yang diambil dari training; baca skala, zero-point, dan dtype tensor dari interpreter. Jangan menganggap input int8 selalu sama dengan float32.

Metadata minimal memuat:

- Versi model dan hash file.
- Bentuk, dtype, dan rentang input.
- Urutan label lengkap.
- Posisi normalisasi: di dalam model atau di luar, tidak keduanya.
- Aturan grayscale, resize, crop, dan orientasi.
- Kebijakan ambang serta kestabilan yang dipakai aplikasi.
- Versi TensorFlow/converter dan runtime yang diuji.
- Sumber dataset, konfigurasi training, dan batas evaluasi.

Ukuran model kecil diharapkan berada di orde ratusan KB hingga beberapa MB, tetapi ukuran final harus diukur setelah ekspor. Ukuran APK, penggunaan RAM, dan waktu prediksi adalah metrik berbeda.

## 12. Pipeline kamera dan preprocessing

### 12.1 Preview dan sampling

Preview kamera berjalan terus selama sesi. Mulai dari target sekitar 3 inferensi per detik, lalu ukur di perangkat sasaran. Tidak perlu mengirim semua frame ke model atau menyamakan FPS preview dengan FPS inferensi.

Gunakan timestamp dan penanda `isProcessing` atau mekanisme setara. Lewati frame ketika interpreter masih bekerja; jangan menumpuk antrean frame lama. Jika memungkinkan, lakukan konversi piksel dan inferensi di luar UI isolate sesuai dukungan runtime. Kelola pembuatan dan penutupan interpreter pada isolate pemiliknya dengan benar.

### 12.2 Area tangan

Versi pertama memakai kotak ROI persegi dengan instruksi pengguna memasukkan satu tangan. Ini tidak sama dengan deteksi lokasi tangan otomatis. Jangan mengklaim hand detector sudah tersedia jika belum dibuat.

Pemetaan kotak layar ke koordinat frame harus mempertimbangkan aspect ratio, preview yang terpotong, orientasi sensor, dan kamera depan yang mungkin menampilkan preview bercermin. Jangan mengasumsikan piksel stream selalu sama dengan tampilan preview.

Tentukan dan dokumentasikan transformasi orientasi serta mirror yang membuat gambar input sesuai data latihan. Uji memakai contoh diketahui. Jangan melakukan flip otomatis tanpa validasi terhadap dataset dan tangan yang ditargetkan.

### 12.3 Konversi input

Urutan awal:

1. Baca format frame kamera yang sebenarnya.
2. Konversi ke RGB dengan penanganan row stride dan pixel stride yang benar untuk YUV420, atau format lain yang disediakan platform.
3. Koreksi rotasi/orientasi.
4. Ambil ROI yang sudah dipetakan dari preview ke koordinat gambar.
5. Ubah ke grayscale menggunakan formula yang konsisten di Python dan Dart.
6. Resize bilinear ke 28 x 28 dengan perilaku yang disepakati.
7. Bentuk tensor `[1, 28, 28, 1]` float32, nilai 0-255 untuk baseline dengan Rescaling di dalam model.
8. Jalankan model dan baca 24 skor output.

Jangan membagi 255 di Flutter jika normalisasi sudah ada di model. Jangan memakai preprocessing CIFAKE RGB 32 x 32; proyek ini menggunakan grayscale 28 x 28 dan 24 kelas.

Sediakan kumpulan gambar referensi kecil untuk membandingkan hasil preprocessing Python dan Flutter, termasuk orientasi, ROI, rentang nilai, dan tensor akhir. Selisih numerik kecil harus diukur dengan toleransi yang dijelaskan.

## 13. Stabilisasi, ketidakpastian, dan tangan tidak terlihat

Baseline kebijakan aplikasi dapat menggunakan:

- Riwayat lima prediksi terbaru dengan timestamp.
- Label yang sama pada sedikitnya tiga hasil terbaru sebelum diterima.
- Ambang skor awal 0.80 dan selisih skor pertama-kedua 0.15 sebagai titik eksperimen.
- Batas usia prediksi agar huruf lama tidak terus ditampilkan ketika pengguna mengubah posisi.

Angka tersebut harus disesuaikan menggunakan data validasi dan data pengembangan kamera yang terpisah dari pengujian akhir. Softmax tidak otomatis merupakan probabilitas kebenaran yang terkalibrasi.

**Masalah penting:** model yang hanya dilatih pada 24 huruf dapat memberi skor tinggi pada gambar tanpa tangan atau bentuk yang tidak dikenal. Threshold dan voting tidak menyelesaikan masalah ini secara pasti.

Karena itu:

- Jangan mengklaim aplikasi memiliki penolakan kelas asing yang andal tanpa pengujian.
- Sertakan pengujian tanpa tangan, tangan terpotong, transisi gerakan, dan bentuk yang tidak didukung.
- Pemeriksaan kualitas gambar hanya menjadi bantuan, bukan bukti bahwa tangan ada.
- Tampilkan petunjuk posisi dan jangan menilai kuis ketika kondisi belum cukup stabil.
- Jika salah deteksi pada kondisi negatif mengganggu demo, usulkan tahap tambahan yang jelas: pengumpulan data negatif untuk model/gate tambahan atau deteksi keberadaan tangan. Dataset tambahan dan perubahan kelas tidak boleh dimasukkan diam-diam.

Data tambahan untuk pengujian kamera tetap diperbolehkan dan memang diperlukan. Jika akan dimasukkan ke training, catat sebagai perubahan sumber data dan pisahkan pengujian akhir yang baru.

## 14. Pengujian dan metrik

### 14.1 Evaluasi model

- Accuracy, macro precision, macro recall, dan macro F1 pada testing bawaan.
- Confusion matrix 24 kelas dan pasangan huruf yang sering tertukar.
- Grafik training/validation loss dan accuracy.
- Hasil yang dapat direproduksi dari checkpoint dan konfigurasi tersimpan.
- Perbandingan hasil model Keras dengan model TFLite pada set yang sama.
- Jika memakai quantization, laporkan perubahan ukuran, metrik, dan kecepatan.

Jangan menjanjikan angka 95-99% tanpa hasil nyata. Jangan menggabungkan metrik dataset dan kamera menjadi satu angka tanpa penjelasan.

### 14.2 Evaluasi kamera nyata

Sebagai rencana, libatkan 3-5 peserta yang menyetujui pengujian dan tidak dipakai sebagai contoh pengembangan final. Pandu pembentukan huruf menggunakan referensi yang tervalidasi; label percobaan harus benar sebelum menilai model.

- Uji seluruh 24 huruf dengan beberapa pengulangan.
- Uji lebih dari satu kondisi pencahayaan dan latar.
- Catat kamera/perangkat, jarak, tangan yang digunakan, dan orientasi.
- Pisahkan sesi pengembangan kamera dari sesi pengujian akhir.
- Sertakan kondisi negatif dan transisi.
- Laporkan keberhasilan pengenalan stabil, huruf tertukar, kegagalan membaca, serta prediksi keliru pada kondisi negatif.

### 14.3 Evaluasi aplikasi dan performa

- Waktu preprocessing, inferensi, dan waktu sampai hasil stabil.
- Median dan persentil ke-95 waktu prediksi pada perangkat yang disebutkan.
- Preview tetap dapat digunakan ketika prediksi berjalan.
- Penggunaan RAM, ukuran model, ukuran APK, dan pengamatan suhu/baterai pada sesi latihan.
- Izin kamera ditolak, kembali dari pengaturan, pindah halaman, background/resume, serta pergantian kamera.
- Model gagal dimuat, metadata salah, dan aset materi belum tersedia.
- Seluruh fitur inti diuji dalam mode pesawat setelah pemasangan.

Target awal adalah prediksi beberapa kali per detik pada HP uji; target diterima atau direvisi setelah profiling. Jangan menyatakan semua perangkat akan memiliki performa yang sama.

### 14.4 Pengujian kode yang bernilai

- Pemetaan label termasuk lompatan I ke K.
- Preprocessing dan transformasi koordinat kamera dengan contoh yang diketahui.
- Kestabilan prediksi, kedaluwarsa hasil, serta penanganan frame berlebih.
- Penyimpanan dan penghapusan progres.
- Smoke test pemuatan model dan inferensi end-to-end pada perangkat Android.

## 15. Tahapan implementasi untuk AI agent

Kerjakan secara bertahap. Jangan menghabiskan waktu pada UI lengkap sebelum kelayakan kamera diuji.

### Tahap A: Audit dan fondasi

- Periksa repositori, instruksi lokal, SDK, perangkat uji, dan file yang sudah ada.
- Pertahankan perubahan pengguna.
- Jika ditemukan notebook CIFAKE dari diskusi lama, anggap sebagai artefak proyek berbeda; jangan menggunakannya sebagai model ASL.
- Tetapkan struktur proyek, dependensi, label, dan kontrak preprocessing.
- Siapkan notebook Colab tanpa token atau kredensial tertanam.

### Tahap B: Baseline ML

- Muat CSV, validasi label/piksel, dan buat split.
- Buat CNN dari bobot acak, training, checkpoint, dan evaluasi validasi.
- Ekspor model TFLite float32 dan metadata.
- Jika tidak memiliki GPU atau runtime untuk training, selesaikan kode yang dapat dijalankan dan laporkan batas verifikasi; jangan mengarang model terlatih atau hasil metrik.

### Tahap C: Uji integrasi kecil

- Buat aplikasi Flutter minimal yang memuat model dan memprediksi gambar referensi.
- Cocokkan hasil dengan Python.
- Tambahkan preview kamera, ROI, preprocessing, dan sampling frame.
- Uji satu perangkat Android nyata sebelum memperluas fitur.
- Tentukan apakah dataset cukup untuk live demo atau perlu perbaikan pipeline/data.

### Tahap D: Fitur pembelajaran

- Implementasikan materi yang tervalidasi, latihan kamera stabil, kuis, progres, dan pengaturan.
- Gunakan state aplikasi yang jelas untuk loading, siap, membaca, belum terbaca, error, serta pause.
- Jangan menambahkan login, leaderboard, chat, atau backend tanpa kebutuhan baru dari pengguna.

### Tahap E: Evaluasi akhir dan distribusi

- Jalankan pengujian dataset akhir setelah model dipilih.
- Jalankan pengujian kamera dengan data/peserta terpisah.
- Ukur model dan performa aplikasi.
- Build APK, uji pemasangan, izin, dan penggunaan offline.
- Siapkan README, model card, laporan hasil, serta skenario demo.

## 16. Pembagian kerja kelompok yang disarankan

Jika kelompok berisi empat orang:

1. ML dan data: dataset, eksperimen, evaluasi, ekspor model.
2. Integrasi kamera: ROI, preprocessing, TFLite, lifecycle, dan performa.
3. Mobile dan materi: navigasi, tampilan belajar, kuis, progres.
4. QA dan dokumentasi: validasi materi, pengujian perangkat/peserta, laporan, dan APK.

Semua anggota menyepakati kontrak input/output model sejak awal. Jika jumlah anggota berbeda, gabungkan tanggung jawab tanpa menghilangkan pengujian.

## 17. Rencana satu semester

Rencana awal 12 minggu kerja efektif, disesuaikan kalender kuliah:

| Minggu | Fokus | Hasil yang dapat diperiksa |
|---|---|---|
| 1 | Scope, referensi materi, dataset, perangkat uji | Spesifikasi dan audit data |
| 2-3 | Baseline CNN | Notebook, checkpoint, grafik validasi |
| 4 | Ekspor dan uji TFLite | Model serta prediksi yang konsisten |
| 5-6 | Kamera dan preprocessing | Demo live minimal pada HP |
| 7-8 | Perbaikan live, materi, kuis | Alur latihan yang dapat digunakan |
| 9 | Progres dan UX | Fitur utama lengkap |
| 10 | Evaluasi akhir | Metrik dataset, kamera, performa |
| 11 | Perbaikan dan build | APK kandidat akhir |
| 12 | Dokumentasi dan presentasi | Laporan, APK, dan demo |

Jika pengujian kamera minggu 5-6 menunjukkan model tidak dapat digunakan, prioritaskan perbaikan data dan preprocessing. Laporkan tradeoff cakupan secara jujur; jangan menyembunyikan kegagalan dengan UI yang menampilkan prediksi palsu.

## 18. Kriteria selesai dan deliverables

### Deliverables

- Source Flutter dan petunjuk setup/build.
- Notebook training ASL yang dapat dijalankan ulang.
- Model hasil training `.keras` dan model mobile `.tflite` ketika training berhasil dilakukan.
- Label, metadata, versi, dan hasil evaluasi.
- Materi alfabet dengan sumber dan izin yang tercatat.
- APK Android yang diuji pada perangkat nyata.
- Dokumentasi pipeline kamera, batasan, hasil uji, dan skenario demo.

### Kriteria selesai

- Pengguna dapat memasang dan menjalankan aplikasi di perangkat sasaran.
- Materi, latihan kamera, kuis yang dinyatakan tersedia, dan progres lokal bekerja.
- Kamera diproses otomatis tanpa pengguna memotret satu per satu.
- Prediksi berasal dari model nyata, bukan label acak atau hardcoded.
- Pemetaan 24 kelas dan preprocessing Python/Flutter sudah diverifikasi.
- Model disimpan, dimuat, dan dijalankan secara offline.
- Tidak ada klaim dukungan J/Z, BISINDO, atau penerjemahan kalimat.
- Hasil evaluasi nyata dan keterbatasannya dilaporkan.
- Kualitas live disepakati berdasarkan hasil uji pengguna, bukan hanya skor benchmark.

Jika model nyata belum tersedia, mock hanya boleh dipakai untuk pengembangan UI dengan penanda eksplisit. Mock tidak boleh masuk demo final sebagai fitur ML yang sudah selesai.

## 19. Di luar scope versi pertama

- Penerjemahan kalimat dan percakapan.
- Pengenalan J/Z atau urutan gerakan dinamis.
- BISINDO, SIBI, dan penggabungan berbagai bahasa isyarat.
- Deteksi otomatis banyak tangan atau banyak pengguna.
- Training model di HP pengguna.
- Pengiriman video ke cloud atau REST API prediksi wajib.
- Login, sinkronisasi lintas perangkat, leaderboard online, serta dashboard admin.
- Publikasi toko aplikasi sebagai syarat penyelesaian PBL.
- Jaminan akurasi atau performa untuk semua HP dan kondisi.

## 20. Instruksi pelaksanaan untuk AI agent

Anda bertindak sebagai engineer Flutter dan machine learning untuk proyek ini. Gunakan dokumen ini sebagai sumber scope dan keputusan teknis. Mulai dengan memeriksa kondisi repositori, lalu lakukan pekerjaan yang dapat dijalankan berdasarkan tahap implementasi.

Prinsip kerja:

1. Bangun solusi sesuai keputusan yang sudah disepakati, terutama ASL 24 kelas, training dari nol, kamera langsung, dan inferensi offline di HP.
2. Jangan mengganti kembali ke proyek deteksi AI vs non-AI, CIFAKE, API Hugging Face, atau classifier pretrained tanpa persetujuan pengguna.
3. Selesaikan fondasi model dan kamera sebelum memperluas fitur.
4. Gunakan dokumentasi versi paket yang benar dan catat dependensi yang diuji.
5. Ajukan pertanyaan hanya ketika informasi tersebut benar-benar menghalangi pekerjaan; lanjutkan bagian independen yang sudah jelas.
6. Jangan mengunggah data, menyewa layanan berbayar, mengaktifkan billing, atau mempublikasikan aplikasi tanpa otorisasi pengguna.
7. Jangan mengarang hasil training, ukuran model, FPS, atau akurasi. Bedakan kode yang baru diperiksa sintaksnya dengan kode yang telah dijalankan dan diuji di perangkat.
8. Jelaskan risiko material dan keputusan implementasi dalam bahasa Indonesia yang mudah dipahami.
9. Pada setiap tahap, laporkan file yang dihasilkan, pemeriksaan yang dilakukan, hasil nyata, serta pekerjaan yang masih tersisa.

**Tugas awal agent:** audit repositori dan lingkungan; buat rencana bertahap; siapkan pipeline dataset, pemetaan 24 label, CNN baseline, dan kontrak preprocessing; kemudian lanjutkan uji integrasi model di Flutter sesuai sumber daya yang tersedia.

## 21. Sumber awal

- Dataset dan deskripsi resmi: https://www.kaggle.com/datasets/datamunge/sign-language-mnist
- Referensi sumber gambar yang ditautkan pembuat dataset: https://github.com/mon95/Sign-Language-and-Static-gesture-recognition-using-sklearn
- Google Colab dan batas sumber daya: https://research.google.com/colaboratory/faq.html

Dokumentasi library Flutter, TensorFlow/Keras, dan runtime TFLite/LiteRT harus diperiksa saat implementasi karena API, kompatibilitas, dan kebutuhan versi dapat berubah. Sumber pengajaran bentuk huruf harus divalidasi terpisah; deskripsi dataset saja bukan kurikulum ASL lengkap.
