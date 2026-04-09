const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const result = document.getElementById("result");

imageInput.addEventListener("change", function () {
  const file = this.files[0];
  if (file) {
    previewImage.src = URL.createObjectURL(file);
    previewImage.style.display = "block";
    result.innerHTML = "";
  }
});

function classifyImage() {
  if (!previewImage.src) {
    alert("Please upload an image first!");
    return;
  }

  // Dummy prediction (replace later with real ML API)
  const predictions = ["Cat 🐱", "Dog 🐶", "Car 🚗", "Flower 🌸"];
  const randomPrediction = predictions[Math.floor(Math.random() * predictions.length)];

  result.innerHTML = "Prediction: " + randomPrediction;
}