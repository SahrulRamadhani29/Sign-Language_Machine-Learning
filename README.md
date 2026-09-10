# Sign-Language Machine Learning

[![Buka di Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SahrulRamadhani29/Sign-Language_Machine-Learning/blob/main/training_colab/LatihIsyarat_Training_Colab.ipynb)

[Bandingkan hasil seed 42, 123, dan 2026 di Colab](https://colab.research.google.com/github/SahrulRamadhani29/Sign-Language_Machine-Learning/blob/main/training_colab/LatihIsyarat_Compare_Seeds_Colab.ipynb)

[Jalankan eksperimen CNN target 99,5%](https://colab.research.google.com/github/SahrulRamadhani29/Sign-Language_Machine-Learning/blob/main/training_colab/LatihIsyarat_Experimental_99_5_Colab.ipynb)

[Jalankan eksperimen besar <10 MiB, 5 approach x 3 seed](https://colab.research.google.com/github/SahrulRamadhani29/Sign-Language_Machine-Learning/blob/main/training_colab/LatihIsyarat_Experimental_Large_MultiSeed_Colab.ipynb)

[Lanjutkan dengan ensemble 3-seed dan full-data fine-tune](https://colab.research.google.com/github/SahrulRamadhani29/Sign-Language_Machine-Learning/blob/main/training_colab/LatihIsyarat_ThreeSeed_FullData_Refit_Colab.ipynb)

[Model juara final yang disiapkan untuk aplikasi PBL](MODEL_JUARA_FINAL_PBL/README.md)

Repositori LatihIsyarat untuk melatih CNN pengenal 24 huruf statis alfabet
American Sign Language (ASL). Huruf J dan Z tidak termasuk karena membutuhkan
pengenalan gerakan.

Pipeline Google Colab tersedia di
[`training_colab/`](training_colab/README.md). Folder tersebut sengaja mandiri
dan tidak menentukan struktur aplikasi Flutter. Setelah training, aplikasi
mobile cukup mengambil model `.tflite`, `labels.json`, dan
`model_metadata.json` dari folder hasil.

Dokumen konsep proyek tersedia di [`konsep.md`](konsep.md).
