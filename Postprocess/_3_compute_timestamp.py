import cv2
import time
from pyzbar import pyzbar

def run_qr_scanner(camera_index=0):
    """
    Start a QR code scanner that reads time stamps from QR codes displayed on another device.
    """
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        print("Error: Unable to open camera. Please check if the camera is being used by another program.")
        return

    print("Camera started. Please align with the time QR code on another device...")
    print("Press 'q' to exit the program.")

    last_decoded_data = None
    
    while True:
        scanner_time_ms = int(time.time() * 1000)
        ret, frame = camera.read()
        if not ret:
            print("Error: Unable to read video frame.")
            break

        decoded_objects = pyzbar.decode(frame)

        for obj in decoded_objects:
            qr_data_str = obj.data.decode('utf-8')
            
            if qr_data_str == last_decoded_data:
                continue
            last_decoded_data = qr_data_str

            try:
                generator_time_ms = int(qr_data_str)
                
                time_difference = scanner_time_ms - generator_time_ms
                
                # --- 在控制台打印结果 ---
                print("----------------------------------------")
                print(f"Decoded QR Code Success")
                print(f"  Generator TimeStamp (A): {generator_time_ms} ms")
                print(f"  Scanner TimeStamp (B): {scanner_time_ms} ms")
                print(f"  Time Difference (B - A): {time_difference} ms")
                print("----------------------------------------\n")

                points = obj.polygon
                if len(points) > 3:
                    pts = [(p.x, p.y) for p in points]
                    cv2.polylines(frame, [tuple(pts)], True, (0, 255, 0), 3)

                text = f"Delay: {time_difference} ms"
                
                rect = obj.rect
                cv2.putText(frame, text, (rect.left, rect.top - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            except ValueError:
                print(f"Decoded non-timestamp QR Code: {qr_data_str}")
                pass
        
        cv2.imshow("Python QR Code Time Scanner", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Closing...")
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_qr_scanner()