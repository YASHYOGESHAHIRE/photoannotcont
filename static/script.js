const uploadForm = document.getElementById("uploadForm");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
let image = null;
let box = { x: 50, y: 50, width: 100, height: 100 }; // Initial box
let dragging = false;

// Handle file upload
uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData(uploadForm);
    const response = await fetch("/upload", {
        method: "POST",
        body: formData,
    });
    const data = await response.json();
    loadImage(data.image_url);
});

// Load the uploaded image
function loadImage(url) {
    image = new Image();
    image.src = url;
    image.onload = () => {
        canvas.width = image.width;
        canvas.height = image.height;
        draw();
    };
}

// Draw the image and the box
function draw() {
    if (image) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = "green";
        ctx.lineWidth = 2;
        ctx.strokeRect(box.x, box.y, box.width, box.height);
    }
}

// Mouse events for dragging/resizing
canvas.addEventListener("mousedown", (e) => {
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    if (
        mouseX >= box.x &&
        mouseX <= box.x + box.width &&
        mouseY >= box.y &&
        mouseY <= box.y + box.height
    ) {
        dragging = true;
    }
});

canvas.addEventListener("mousemove", (e) => {
    if (dragging) {
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        box.width = mouseX - box.x;
        box.height = mouseY - box.y;
        draw();
    }
});

canvas.addEventListener("mouseup", () => {
    dragging = false;
});

// Save annotation
const annotationForm = document.getElementById("annotationForm");
annotationForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const label = document.getElementById("labelInput").value;
    const response = await fetch("/save_annotations", {
        method: "POST",
        body: new URLSearchParams({
            label: label,
            x: box.x,
            y: box.y,
            width: box.width,
            height: box.height,
        }),
    });
    const data = await response.json();
    alert(data.message);
});
