import torch
import glob
import cv2

from transformers import AutoImageProcessor, AutoModelForImageClassification


class VITforImageClassification:
    """
    This model is mainly used for evaluation porpuoses.
    When generating an image, we need to validate how well
    the generated image is similar to a real one
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and processor from HuggingFace
        self.processor = AutoImageProcessor.from_pretrained(
            "farleyknight/mnist-digit-classification-2022-09-04"
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            "farleyknight/mnist-digit-classification-2022-09-04"
        ).to(self.device)

    def run_inference(self, input_processed):
        logits = self.model(**input_processed).logits

        return torch.argmax(logits, dim=1).cpu().numpy()

    def process_images(self, image):
        processed_image = self.processor(image, return_tensors="pt")

        return processed_image


if __name__ == "__main__":
    test_images = glob.glob("data/mnist_data/image2audio/test/*.jpg")
    images = [cv2.imread(image_file) for image_file in test_images][:50]

    print("[ INFO ] Loading model ...")
    model = VITforImageClassification()
    print("[ INFO ] Model Loaded")

    print("[ INFO ] Running inference..")
    model_inputs = model.process_image(images)
    results = model.run_inference(model_inputs.to(model.device))
    print("[ INFO ] Finished inference")
    print(f"[ INFO ] Classification: {results}")
