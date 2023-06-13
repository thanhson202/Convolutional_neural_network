import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

# Đường dẫn đến file mô hình đã được huấn luyện
model_path = "cnn_model.h5"

# Kích thước ảnh đầu vào
image_width, image_height = 150, 150

# Load mô hình đã được huấn luyện
model = tf.keras.models.load_model(model_path)

# Chúng ta cần khai báo lại class_labels để tương ứng với số lớp
class_labels = {0: "conuoc_matnap_conhan", 1: "day_du", 2: "daynuoc_matnap_matnhan",
                3: "daynuoc_conap_matnhan", 4: "matnuoc_conap_conhan",
                5: "matnuoc_matnap_conhan",6: "thieunuoc_conap_matnhan", 7: "thieunuoc_matnap_matnhan"}

# Khởi tạo camera
camera = cv2.VideoCapture(0)

while True:
    # Đọc khung hình từ camera
    ret, frame = camera.read()

    if not ret:
        break

    # Chuyển đổi khung hình thành ảnh RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Thay đổi kích thước ảnh
    image = cv2.resize(image, (image_width, image_height))

    # Chuẩn hóa ảnh
    image = image / 255.0

    # Mở rộng kích thước ảnh thành batch size = 1
    image = np.expand_dims(image, axis=0)

    # Dự đoán lớp của ảnh
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_class_index]

    # Hiển thị kết quả dự đoán lên khung hình
    cv2.putText(frame, predicted_class, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Hiển thị khung hình
    cv2.imshow("Camera", frame)

    # Thoát khỏi vòng lặp nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ hiển thị
camera.release()
cv2.destroyAllWindows()
