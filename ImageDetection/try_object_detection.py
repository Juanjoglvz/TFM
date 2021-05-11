from imageai.Detection import ObjectDetection
import os

PATH_TO_IMAGES_DIR = "F:\\MultiStanceCat-IberEval-training-20180404\\photos"
PATH_TO_OUTPUT_IMAGES_DIR = "F:\\MultiStanceCat-IberEval-training-20180404\\output_photos"
PATH_TO_OUTPUT_LABELS_DIR = "F:\\MultiStanceCat-IberEval-training-20180404\\output_labels"


def main():
    already_visited = []
    for root, dirs, files in os.walk(PATH_TO_OUTPUT_IMAGES_DIR):
        for file in files:
            already_visited.append(file.split("_")[0])

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join("data", "resnet50_coco_best_v2.1.0.h5"))
    detector.loadModel()

    for root, dirs, files in os.walk(PATH_TO_IMAGES_DIR):
        for file in files:
            file_id, file_ext = file.split(".")
            if file_id not in already_visited:
                print(os.path.join(root, file))
                detections = detector.detectObjectsFromImage(input_image=os.path.join(root, file),
                                                             output_image_path=os.path.join(PATH_TO_OUTPUT_IMAGES_DIR,
                                                                                            f"{file_id}_output.{file_ext}"),
                                                             minimum_percentage_probability=50)
                with open(os.path.join(PATH_TO_OUTPUT_LABELS_DIR, f"{file_id}_label.txt"), "w+") as f:
                    for eachObject in detections:
                        f.write(f"{eachObject['name']}:{eachObject['percentage_probability']}\n")


if __name__ == "__main__":
    main()
