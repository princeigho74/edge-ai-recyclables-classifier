"""
Edge AI Recyclables Classifier
Complete implementation for training, conversion, and deployment
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time
import os

# ==================== DATA PREPARATION ====================

def create_sample_dataset():
    """
    Create a sample dataset for demonstration
    In practice, use real recyclables dataset like TrashNet or custom collected data
    """
    # Using TensorFlow's image_dataset_from_directory for real implementation
    # This is a placeholder structure
    
    img_height, img_width = 224, 224
    batch_size = 32
    
    # Example: Load from directory structure:
    # data/
    #   plastic_bottle/
    #   glass_bottle/
    #   aluminum_can/
    #   paper_cardboard/
    #   non_recyclable/
    
    train_ds = keras.utils.image_dataset_from_directory(
        'data/train',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    val_ds = keras.utils.image_dataset_from_directory(
        'data/train',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    # Optimize dataset performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    class_names = train_ds.class_names
    return train_ds, val_ds, class_names


def augment_data(image, label):
    """Data augmentation for better generalization"""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    return image, label


# ==================== MODEL TRAINING ====================

def build_model(num_classes=5, input_shape=(224, 224, 3)):
    """
    Build a lightweight model using MobileNetV2 as base
    Optimized for Edge AI deployment
    """
    # Load pre-trained MobileNetV2
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        alpha=1.0  # Width multiplier for model size
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        
        # Preprocessing
        layers.Rescaling(1./255),
        
        # Base model
        base_model,
        
        # Classification layers
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def train_model(model, train_ds, val_ds, epochs=15):
    """Train the model with callbacks for optimization"""
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history


# ==================== MODEL EVALUATION ====================

def evaluate_model(model, val_ds, class_names):
    """Comprehensive model evaluation"""
    
    # Get predictions
    y_true = []
    y_pred = []
    
    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
    
    # Classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(
        y_true, y_pred, 
        target_names=class_names,
        digits=3
    ))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()
    
    # Calculate per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(per_class_acc):
        print(f"{class_names[i]}: {acc*100:.2f}% accuracy")
    
    return cm, per_class_acc


# ==================== TENSORFLOW LITE CONVERSION ====================

def convert_to_tflite(model, quantize=True):
    """
    Convert Keras model to TensorFlow Lite format
    Includes post-training quantization for size reduction
    """
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # Post-training quantization (INT8)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Optional: Full integer quantization (requires representative dataset)
        # def representative_dataset():
        #     for data in train_ds.take(100):
        #         yield [tf.cast(data[0], tf.float32)]
        # converter.representative_dataset = representative_dataset
        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    tflite_filename = 'recyclables_classifier.tflite'
    with open(tflite_filename, 'wb') as f:
        f.write(tflite_model)
    
    # Compare sizes
    keras_size = os.path.getsize('best_model.h5') / (1024 * 1024)
    tflite_size = os.path.getsize(tflite_filename) / (1024 * 1024)
    
    print(f"\n{'='*60}")
    print("MODEL SIZE COMPARISON")
    print(f"{'='*60}")
    print(f"Keras Model: {keras_size:.2f} MB")
    print(f"TFLite Model: {tflite_size:.2f} MB")
    print(f"Compression: {(1 - tflite_size/keras_size)*100:.1f}% smaller")
    
    return tflite_model, tflite_filename


# ==================== EDGE DEPLOYMENT ====================

class EdgeInference:
    """TensorFlow Lite inference engine for edge devices"""
    
    def __init__(self, model_path, class_names):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.class_names = class_names
        
        # Get input shape
        self.input_shape = self.input_details[0]['shape'][1:3]
        
    def preprocess_image(self, image_path):
        """Load and preprocess image for inference"""
        img = keras.preprocessing.image.load_img(
            image_path, 
            target_size=self.input_shape
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, 0) / 255.0
        return img_array.astype(np.float32)
    
    def predict(self, image_path):
        """Run inference on a single image"""
        # Preprocess
        img_array = self.preprocess_image(image_path)
        
        # Set input tensor
        self.interpreter.set_tensor(
            self.input_details[0]['index'], 
            img_array
        )
        
        # Run inference
        start_time = time.time()
        self.interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Get output
        predictions = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )[0]
        
        # Get predicted class and confidence
        predicted_idx = np.argmax(predictions)
        predicted_class = self.class_names[predicted_idx]
        confidence = predictions[predicted_idx]
        
        return {
            'class': predicted_class,
            'confidence': float(confidence),
            'all_probabilities': {
                self.class_names[i]: float(predictions[i]) 
                for i in range(len(self.class_names))
            },
            'inference_time_ms': inference_time
        }
    
    def benchmark(self, image_path, num_runs=100):
        """Benchmark inference performance"""
        img_array = self.preprocess_image(image_path)
        
        times = []
        for _ in range(num_runs):
            start = time.time()
            self.interpreter.set_tensor(
                self.input_details[0]['index'], 
                img_array
            )
            self.interpreter.invoke()
            times.append((time.time() - start) * 1000)
        
        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'median_ms': np.median(times)
        }


# ==================== MAIN EXECUTION ====================

def main():
    """Complete training and deployment pipeline"""
    
    print("="*60)
    print("EDGE AI RECYCLABLES CLASSIFIER")
    print("="*60)
    
    # 1. Load data
    print("\n[1/6] Loading dataset...")
    # train_ds, val_ds, class_names = create_sample_dataset()
    # For demo purposes, we'll simulate this
    class_names = ['Plastic Bottle', 'Glass Bottle', 'Aluminum Can', 
                   'Paper/Cardboard', 'Non-Recyclable']
    
    # 2. Build model
    print("\n[2/6] Building model...")
    model = build_model(num_classes=len(class_names))
    model.summary()
    
    # 3. Train model
    print("\n[3/6] Training model...")
    # history = train_model(model, train_ds, val_ds, epochs=15)
    
    # 4. Evaluate model
    print("\n[4/6] Evaluating model...")
    # cm, per_class_acc = evaluate_model(model, val_ds, class_names)
    
    # 5. Convert to TFLite
    print("\n[5/6] Converting to TensorFlow Lite...")
    # tflite_model, tflite_filename = convert_to_tflite(model, quantize=True)
    
    # 6. Test edge inference
    print("\n[6/6] Testing edge inference...")
    # edge_engine = EdgeInference(tflite_filename, class_names)
    # result = edge_engine.predict('test_image.jpg')
    # print(f"\nPrediction: {result['class']}")
    # print(f"Confidence: {result['confidence']*100:.2f}%")
    # print(f"Inference Time: {result['inference_time_ms']:.2f} ms")
    
    # Benchmark
    # bench_results = edge_engine.benchmark('test_image.jpg', num_runs=100)
    # print(f"\nBenchmark Results (100 runs):")
    # print(f"  Mean: {bench_results['mean_ms']:.2f} ms")
    # print(f"  Std:  {bench_results['std_ms']:.2f} ms")
    # print(f"  Min:  {bench_results['min_ms']:.2f} ms")
    # print(f"  Max:  {bench_results['max_ms']:.2f} ms")
    
    print("\n" + "="*60)
    print("DEPLOYMENT COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()


# ==================== RASPBERRY PI DEPLOYMENT ====================

"""
RASPBERRY PI SETUP INSTRUCTIONS:

1. Install dependencies:
   sudo apt-get update
   sudo apt-get install python3-pip python3-opencv
   pip3 install tensorflow-lite tflite-runtime numpy pillow

2. Copy model file:
   scp recyclables_classifier.tflite pi@raspberrypi.local:~/

3. Camera integration (save as camera_classifier.py):

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

class LiveClassifier:
    def __init__(self, model_path):
        self.interpreter = Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
    def classify_frame(self, frame):
        # Resize and preprocess
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0).astype(np.float32) / 255.0
        
        # Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        return predictions

# Initialize
classifier = LiveClassifier('recyclables_classifier.tflite')
cap = cv2.VideoCapture(0)

classes = ['Plastic', 'Glass', 'Aluminum', 'Paper', 'Non-Recyclable']

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    predictions = classifier.classify_frame(frame)
    predicted_class = classes[np.argmax(predictions)]
    confidence = np.max(predictions)
    
    # Display
    cv2.putText(frame, f"{predicted_class}: {confidence*100:.1f}%", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Recyclables Classifier', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

4. Run on startup:
   Add to /etc/rc.local:
   python3 /home/pi/camera_classifier.py &
"""
