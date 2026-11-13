# Formative 2: Multimodal Data Processing

## Introduction

This project was completed for the Formative 2 assignment. Three types of data were processed: customer information, facial images and voice recordings. The datasets were prepared, features were extracted, machine-learning models were trained, and a local web application was developed to demonstrate multimodal transaction verification.

The project includes three prediction tasks:

1. Identify a participant from a facial image.
2. Identify a participant from a voice recording.
3. Predict a product category from customer transaction and social-profile information.

The face and voice models are small classroom experiments. They are not designed for real financial security.

## Project structure

```text
data/
  raw/                         Original customer CSV files
  processed/                   Cleaned and merged customer data
  images/raw/                  Original participant images
  images/processed/            Cropped images and extracted HOG features
  audio/raw/                   Original participant recordings
  audio/processed/             Augmented recordings and extracted audio features
models/                        Trained model files
notebooks/
  01_tabular_preprocessing.ipynb
  02_image_preprocessing.ipynb
  03_audio_preprocessing.ipynb
  04_model_training_evaluation.ipynb
src/
  preprocess_tabular.py
  train_face_model.py
  train_voice_model.py
  train_product_model.py
  transaction_cli.py
  transaction_simulation.py
```


## Standalone model scripts

Each modelling task also has its own training script:

```bash
python src/train_face_model.py
python src/train_voice_model.py
python src/train_product_model.py
```

These scripts use the processed feature tables created by notebooks 01-03 and save the trained models in `models/`. The training notebook remains available for the full comparison and evaluation results.

## Command-line transaction simulation

The assignment's command-line requirement is supported by `src/transaction_cli.py`. The command accepts a claimed participant name, an image path and a WAV path:

```bash
python src/transaction_cli.py --name Nicole --face data/images/raw/nicole_neutral.jpg --voice data/audio/raw/nicole_yes_approve.wav
```

The file paths above are examples and should be replaced with the actual filenames in the local project. A confidence threshold can be supplied with `--threshold`; the CLI default demonstration threshold is 0.75. Predictions below the threshold are treated as unknown. Both the face and voice predictions must match the claimed participant before the transaction is accepted. After successful verification, the CLI automatically selects the demo customer record assigned to that participant, retrieves the customer's transaction features and displays a product-category prediction with the top category scores.

For this classroom demonstration, the CLI uses the following fixed participant-to-customer assignments: Ella → 100, King → 101, Nicole → 102 and Sarah → 103. These are manually assigned demo links to existing customer IDs, not verified real-world identity matches or evidence that those customer records belong to the named participants. The mapping is defined in `CUSTOMER_IDS` in `src/transaction_cli.py` and can be changed if the group later verifies a different mapping.

The threshold is a classroom demonstration setting, not a security-grade threshold. The biometric dataset is too small to establish reliable confidence calibration or real-world unknown-person rejection.

The Flask localhost interface in `src/transaction_simulation.py` is retained as an additional demonstration interface. It applies the same default 0.60 confidence threshold.

## Multimodal decision logic

The claimed name is mapped to a participant ID. The uploaded face is converted into HOG features and the uploaded voice recording is converted into MFCC, spectral roll-off, RMS-energy and duration features. Separate models predict the participant identity from each modality.

The decision follows an AND rule:

1. The face model must return a known participant with confidence at or above the threshold.
2. The voice model must return a known participant with confidence at or above the threshold.
3. Both predicted participant IDs must match the claimed account.
4. Only after both checks pass may the product-category model run.

A failed or low-confidence face or voice result rejects the transaction. A successful face check cannot override a failed voice check, and vice versa.

## Data preparation

### Customer transaction and social-profile data

The first notebook checks the two customer tables, including their dimensions, column types, missing values, duplicates and product-category distribution. It also investigates how the customer IDs in the two files might match.

The social-profile file uses IDs with an `A` prefix, while the transaction file uses numeric IDs. Removing the prefix is tested as a possible way to match the records. This assumption should be checked against the original dataset documentation.

Some customers appear more than once in the social-profile table. Social-profile records are grouped by customer before the transaction table is merged. This avoids creating extra transaction rows through a many-to-many join.

The processed files are saved in `data/processed/`:

- `transactions_clean.csv`
- `social_profiles_customer_level.csv`
- `customer_transactions_social_merged.csv`
- `tabular_model_features.csv`

The standalone preprocessing script can be run from the project root:

```bash
pip install -r requirements.txt
python src/preprocess_tabular.py
```

### Facial image data

The image notebook uses the group's collected images. Each participant has neutral, smiling and surprised expressions. The files are checked, participant and expression labels are identified, faces are detected, and face crops are resized to 128 × 128 pixels.

Image augmentation is performed using small rotations, horizontal flipping and contrast normalisation. HOG features are then extracted and saved in `data/images/processed/image_features.csv`.

The participant labels used in the project are:

| Participant | ID |
|---|---|
| Ella | participant_01 |
| King | participant_02 |
| Nicole | participant_03 |
| Sarah | participant_04 |

There were 12 original images. Face detection succeeded for 11 images, while one image was skipped. The augmented feature table contained 55 rows. Augmented copies are variations of the original images, not additional independent samples.

### Voice recordings

The audio notebook processes two phrases from each participant: “Yes, approve” and “Confirm transaction”. The recordings are checked, converted to mono and resampled to 16 kHz.

Pitch shifting, time stretching and low-level noise are applied to create augmented versions of the recordings. The feature extraction step calculates MFCC statistics, spectral roll-off and RMS energy. The results are saved in `data/audio/processed/audio_features.csv`.

Eight original WAV recordings were found, with two recordings for each of the four participants. The augmented audio feature table contained 48 rows. These augmented rows are not additional independent voice samples.

## Model training and evaluation

The fourth notebook trains and compares models for all three tasks:

- **Facial identity:** Linear SVM and Random Forest using HOG features.
- **Voice identification:** RBF SVM and Random Forest using audio features.
- **Product category:** Logistic Regression and Random Forest using customer features.

The evaluation includes accuracy, precision, recall, F1-score and log loss where probability estimates are available. Classification reports and confusion matrices are also produced.

The face and voice test sets are very small, with only four held-out original samples each. Both models achieved perfect scores on these small test sets, but this does not demonstrate reliable performance with new people or different recording conditions.

The product-category models achieved 22.58% accuracy on a five-class task. This is only slightly above the 20% random-choice baseline, so the product predictions are weak and should be treated as experimental.

The trained models are saved in `models/`, while evaluation tables are saved in `data/processed/model_evaluation/`.

## Local transaction verification application

The transaction simulation is a small Flask web application that runs locally. It provides a browser-based form instead of requiring sample selection through the command line.

The application asks the operator to:

1. Enter the account holder's name: Ella, King, Nicole or Sarah.
2. Upload a facial image in JPG or PNG format.
3. Upload a voice recording in WAV format.
4. Select **Verify and process transaction**.

The uploaded image is processed using the same face-detection and HOG feature extraction approach used during preprocessing. The voice recording is converted to mono, resampled to 16 kHz and processed using the same MFCC, spectral roll-off and RMS-energy feature calculations used in the audio notebook.

The face model and voice model independently predict the participant's identity. Both predictions must match the entered account name for the transaction to be authorised. If either check fails, the transaction is rejected. When both checks pass, the product-category model produces an experimental category suggestion using a held-out example transaction from the customer dataset.

The uploaded image and audio are processed in memory and are not saved by the application.

### Starting the local application

Run the notebooks in the order listed below first so that the feature files and trained models are available. Then run the following command from the project root:

```bash
python src/transaction_simulation.py
```

Open the local address shown in the terminal:

```text
http://127.0.0.1:5000
```

The application is intended for local demonstration on the computer where the project and trained model files are stored.

For the demonstration, test one authorised attempt using the matching participant's face and voice, then test unauthorised attempts by uploading a face or voice sample belonging to a different participant.

The product-category suggestion is a demonstration output based on a held-out customer example. It is not a personalised recommendation based on the uploaded biometric files.

This simulation has no liveness detection or anti-spoofing. The models were trained on very small datasets, so the application must not be used to authorise real transactions.

## Main limitations

The biggest limitation is the small amount of biometric data. There are only four participants, three original images per participant and two original voice recordings per participant. Augmentation increases the number of training rows, but it does not replace collecting more genuine samples.

The product-category model also performed poorly. More customer data, verification of the customer-ID matching assumption and further feature experiments would be needed to improve it.

The results should therefore be understood as a demonstration of the preprocessing and modelling workflow rather than a finished biometric or recommendation system.

## Running the notebooks

Install the packages listed in `requirements.txt`, then run the notebooks in this order:

1. `01_tabular_preprocessing.ipynb`
2. `02_image_preprocessing.ipynb`
3. `03_audio_preprocessing.ipynb`
4. `04_model_training_evaluation.ipynb`

The local transaction verification application can be started after the processed datasets and trained models have been generated.

## Data and consent

The facial images and voice recordings were collected from group participants who agreed to their use for this assignment. These files contain biometric information, so access should remain limited to the people who need them for the coursework. They should not be reused for another purpose without permission.
