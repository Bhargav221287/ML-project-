import cv2
import numpy as np

class YOLOObjectDetector:
    def __init__(self, weights_path, config_path, classes_path):
        """
        Initialize YOLO object detector
        
        Args:
        weights_path (str): Path to pre-trained weights file
        config_path (str): Path to network configuration file
        classes_path (str): Path to classes names file
        """
        # Load class names
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Load YOLO network
        self.net = cv2.dnn.readNet(weights_path, config_path)
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Define colors for bounding boxes
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3))

    def detect_objects(self, frame, confidence_threshold=0.5, nms_threshold=0.4):
        """
        Detect objects in a frame
        
        Args:
        frame (numpy.ndarray): Input image frame
        confidence_threshold (float): Minimum confidence to consider detection
        nms_threshold (float): Non-maximum suppression threshold
        
        Returns:
        tuple: Processed frame, list of detections
        """
        height, width, _ = frame.shape
        
        # Create blob from frame
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        # Set input to network
        self.net.setInput(blob)
        
        # Forward pass through the network
        outs = self.net.forward(self.output_layers)
        
        # Lists to store detected objects
        class_ids = []
        confidences = []
        boxes = []
        
        # Process network output
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression to remove overlapping bounding boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        
        # Draw bounding boxes and labels
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = self.colors[class_ids[i]]
                
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), color.tolist(), 2)
                
                # Create label text with class and confidence
                text = f"{label}: {confidence:.2f}"
                cv2.putText(frame, text, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)
        
        return frame

def main():
    # Paths to YOLO files (you'll need to download these)
    weights_path = 'yolov3.weights'
    config_path = 'yolov3.cfg'
    classes_path = 'coco.names'
    
    try:
        # Initialize detector
        detector = YOLOObjectDetector(weights_path, config_path, classes_path)
        
        # Open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            # Detect and draw objects
            frame_with_objects = detector.detect_objects(frame)
            
            # Display the frame
            cv2.imshow('Object Detection', frame_with_objects)
            
            # Break loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
