# churn_transform.py (Versi Final yang Valid dan Robust)

import tensorflow as tf
import tensorflow_transform as tft

# Daftar fitur kategorikal dan numerik
CATEGORICAL_FEATURES = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]
NUMERICAL_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Nama fitur label
LABEL_KEY = 'Churn'


def _transformed_name(key):
    """
    Menghasilkan nama baru untuk fitur yang telah ditransformasi,
    dengan menambahkan akhiran '_xf'.
    """
    return key + '_xf'


def preprocessing_fn(inputs):
    """
    Fungsi pra-pemrosesan untuk data churn menggunakan TFX Transform.

    Args:
        inputs: Dictionary dari tensor-tensor mentah (raw features).

    Returns:
        Dictionary dari tensor-tensor yang telah ditransformasi.
    """
    outputs = {}

    # === Transformasi Fitur Numerik ===
    for feature in NUMERICAL_FEATURES:
        input_tensor = inputs[feature]

        # Penanganan khusus untuk fitur bertipe string (mis. 'TotalCharges')
        if input_tensor.dtype == tf.string:
            # Mengganti string kosong atau hanya spasi dengan '0.0' lalu mengonversi ke float32
            cleaned_tensor = tf.strings.regex_replace(input_tensor, r'^\s*$', '0.0')
            numeric_tensor = tf.strings.to_number(cleaned_tensor, out_type=tf.float32)
        else:
            numeric_tensor = tf.cast(input_tensor, dtype=tf.float32)

        # Standardisasi (Z-score): (x - mean) / std_dev
        outputs[_transformed_name(feature)] = tft.scale_to_z_score(numeric_tensor)

    # === Transformasi Fitur Kategorikal ===
    for feature in CATEGORICAL_FEATURES:
        # Membuat indeks integer dari nilai string berdasarkan vocabulary
        int_feature = tft.compute_and_apply_vocabulary(
            inputs[feature],
            vocab_filename=feature
        )

        # Mendapatkan ukuran vocabulary untuk menentukan 'depth' one-hot encoding
        vocab_size = tft.experimental.get_vocabulary_size_by_name(feature)
        vocab_size_int32 = tf.cast(vocab_size, dtype=tf.int32)

        # One-hot encoding pada fitur kategorikal
        outputs[_transformed_name(feature)] = tf.one_hot(
            indices=int_feature,
            depth=vocab_size_int32,
            on_value=1.0,
            off_value=0.0
        )

    # === Transformasi Label ===
    # Mengonversi nilai 'Yes' menjadi 1 dan 'No' menjadi 0
    outputs[_transformed_name(LABEL_KEY)] = tf.cast(
        tf.where(tf.equal(inputs[LABEL_KEY], 'Yes'), 1, 0),
        dtype=tf.float32
    )

    return outputs
