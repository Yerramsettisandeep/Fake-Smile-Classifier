import cv2
import joblib
import numpy as np

def real_time_detection():
    # Load the trained model
    model = joblib.load('models/smile_classifier.pkl')

    # Start webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize image to make it faster to process
        resized_gray = cv2.resize(gray, (64, 64))
        
        # Extract histogram features
        hist = cv2.calcHist([resized_gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # Make a prediction
        prediction = model.predict([hist])[0]
        label = "Real Smile" if prediction == 0 else "Fake Smile"

        # Display the label
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Smile Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_detection()
