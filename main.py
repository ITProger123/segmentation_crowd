import cv2
from ultralytics import YOLO
import numpy as np

def get_video(path: str) -> cv2.VideoCapture:
    """
    Считывает видео по заданному пути

    Args:
        path (str): путь до видео

    Returns:
        cv2.VideoCapture: объект полученного видеопотока
    
    Raises:
        FileNotFoundError: Выдает исключение, если видео по заданному пути нет
    """
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        raise FileNotFoundError(f"The video on the path {path} is not found")
    return video

def setup_result_videofile(video: cv2.VideoCapture, output: str) -> cv2.VideoWriter:
    """
    Настраивает параметры выходного видео с сегментацией

    Args:
        video (cv2.VideoCapture): объект исходного видеопотока
        output (str): путь, в котором будет сохранено сегментированное видео

    Returns:
        cv2.VideoWriter: Объект для записи видео.
    """
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    format_video = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output, format_video, fps, (width, height))

def process_frame(frame: np.ndarray, model: YOLO, alpha=0.3) -> np.ndarray:
    """
    Обрабатывает один кадр, на котором выполняется сегментация, добавление имени класса и уровня уверенности

    Args:
        frame (np.ndarray): Входной кадр для обработки в формате BGR
        model (YOLO): Модель, которая будет выполнять сегментацию кадра
        alpha : Прозрачность маски (по умолчанию 0.3)

    Returns:
        np.ndarray: Обработанный кадр видео с добавлением маски и подписями класса и уровнями уверенности
    """
    results = model(frame)
    for result in results:
        for box, mask in zip(result.boxes, result.masks or []):
            if int(box.cls) != 0:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf.item()
            label = f"Person: {confidence:.2f}"
            mask_data = mask.data[0].cpu().numpy()
            mask_data = cv2.resize(mask_data, (frame.shape[1], frame.shape[0]),interpolation=cv2.INTER_NEAREST)
            overlay = frame.copy()
            overlay[mask_data > 0] = (0, 255, 0)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            contours, _ = cv2.findContours(mask_data.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame

PATH = '/app/data/crowd.mp4'
OUTPUT_PATH = '/app/data/segmented_crowd.mp4'


def main():
    """
    Основная логика программы: чтение каждого кадра, выполнение сегментации, запись изменений в выходное видео
    """
    model = YOLO("yolov8l-seg.pt")
    video = get_video(PATH)
    output_video = setup_result_videofile(video, OUTPUT_PATH)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        processed_frame = process_frame(frame, model)
        output_video.write(processed_frame)
    video.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
