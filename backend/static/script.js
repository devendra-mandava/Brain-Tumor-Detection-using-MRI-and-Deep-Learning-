function uploadImage() {
    let input = document.getElementById("imageInput");
    if (input.files.length === 0) {
        alert("Please select an image");
        return;
    }

    let formData = new FormData();
    formData.append("file", input.files[0]);

    let button = document.querySelector("button");
    button.innerText = "Processing...";
    button.disabled = true;

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        button.innerText = "Upload & Analyze";
        button.disabled = false;

        if (data.error) {
            alert("Error: " + data.error);
            console.error("Server Error:", data.error);
        } else {
            // Display uploaded image
            let uploadedImage = document.getElementById("uploadedImage");
            uploadedImage.src = URL.createObjectURL(input.files[0]);
            uploadedImage.classList.remove("hidden");

            // Force reload processed image by appending a timestamp
            let outputImage = document.getElementById("outputImage");
            outputImage.src = data.output_image + "?t=" + new Date().getTime();
            outputImage.classList.remove("hidden");
        }
    })
    .catch(error => {
        console.error("Fetch Error:", error);
        alert("Failed to upload image. Please check the server.");
        button.innerText = "Upload & Analyze";
        button.disabled = false;
    });
}
