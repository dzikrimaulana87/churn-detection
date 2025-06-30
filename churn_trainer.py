# churn_trainer.py (Versi Final - Perbaikan OSError)

import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

# ============================================================================
# Definisi Konstanta dan Fungsi Helper
# ============================================================================

CATEGORICAL_FEATURES = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]
NUMERICAL_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']
LABEL_KEY = 'Churn'


def _transformed_name(key):
    """
    Menghasilkan nama baru untuk fitur yang telah ditransformasi,
    dengan menambahkan akhiran '_xf'.
    """
    return key + '_xf'


# ============================================================================
# Fungsi-fungsi Utama yang Digunakan oleh run_fn
# ============================================================================

def _input_fn(file_pattern: str,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 32) -> tf.data.Dataset:
    """
    Membuat dataset dari file TFRecord yang sudah ditransformasi untuk pelatihan atau evaluasi.

    Args:
        file_pattern: Pola file TFRecord (dengan kompresi GZIP).
        tf_transform_output: Objek TFTransformOutput dari hasil transformasi.
        batch_size: Ukuran batch untuk training atau evaluasi.

    Returns:
        tf.data.Dataset yang sudah dibatch dan siap digunakan.
    """
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=lambda filenames: tf.data.TFRecordDataset(filenames, compression_type='GZIP'),
        label_key=_transformed_name(LABEL_KEY)
    )
    return dataset


def _build_keras_model(tf_transform_output: tft.TFTransformOutput) -> tf.keras.Model:
    """
    Membangun model Keras dengan input dinamis dari fitur numerik dan kategorikal.

    Args:
        tf_transform_output: Objek TFTransformOutput untuk mengambil spesifikasi fitur.

    Returns:
        Objek tf.keras.Model yang sudah terkompilasi.
    """
    model_inputs = []
    processed_tensors = []

    # Input untuk fitur numerik
    for feature in NUMERICAL_FEATURES:
        input_layer = tf.keras.layers.Input(shape=(1,), name=_transformed_name(feature))
        model_inputs.append(input_layer)
        processed_tensors.append(input_layer)

    # Input untuk fitur kategorikal
    feature_spec = tf_transform_output.transformed_feature_spec()
    for feature in CATEGORICAL_FEATURES:
        spec = feature_spec[_transformed_name(feature)]
        cat_input = tf.keras.layers.Input(shape=spec.shape,
                                          name=_transformed_name(feature),
                                          dtype=spec.dtype)
        squeezed_input = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1))(cat_input)
        model_inputs.append(cat_input)
        processed_tensors.append(squeezed_input)

    # Menggabungkan semua input
    concat = tf.keras.layers.Concatenate()(processed_tensors)

    # Arsitektur Dense layer
    x = tf.keras.layers.Dense(64, activation='relu')(concat)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # Kompilasi model
    model = tf.keras.Model(inputs=model_inputs, outputs=outputs)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy']
    )

    model.summary()
    return model


def run_fn(fn_args: FnArgs):
    """
    Fungsi utama yang digunakan oleh komponen TFX Trainer untuk melatih model.

    Args:
        fn_args: Objek FnArgs yang menyediakan argumen pelatihan termasuk file,
                 direktori output, jumlah langkah, dan lainnya.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = _input_fn(fn_args.eval_files, tf_transform_output)

    model = _build_keras_model(tf_transform_output)

    model.fit(
        train_dataset,
        epochs=10,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps
    )

    # Menyimpan model tanpa signature kustom.
    # Signature akan dibuat otomatis oleh TFX agar kompatibel dengan komponen Evaluator.
    model.save(fn_args.serving_model_dir, save_format='tf')
